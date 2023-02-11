import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import logging

from transformers import BartTokenizer, BartConfig,T5Tokenizer,T5Config
from transformers import T5EncoderModel
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import (
    AdamW,
    Adafactor,
    get_scheduler,
    is_torch_available,
)
from Diffusion_T5.dataloader.fewshot_gym_singletask_t5large_full  import NLPFewshotGymSingleTaskData

from Diffusion_T5.utils_t5 import freeze_embeds, trim_batch
from Diffusion_T5.modeling_t5 import T5ForConditionalGeneration

from Gpt.distributed import get_rank, get_world_size, is_main_proc, synchronize

from tqdm import tqdm

import re
from glob import glob


def make_args():
    parser = argparse.ArgumentParser()

    ## Basic parameters
    #parser.add_argument("--task_dir", default="/data3/private/yyn/data_eval_100_task/glue-sst2", required=False)
    parser.add_argument("--task_dir", default="/data/private/yeyining/data_eval_100_task/glue-sst2", required=False)
    parser.add_argument("--train_file", default="data", required=False)
    #parser.add_argument("--dev_file", default="/data3/private/yyn/data_eval_100_task/glue-sst2/glue-sst2_dev.tsv", required=False)
    parser.add_argument("--dev_file", default="/data/private/yeyining/data_eval_100_task/glue-sst2/glue-sst2_dev.tsv", required=False)
    parser.add_argument("--test_file", default="data", required=False)
    parser.add_argument("--dataset", default="nlp_forest_single", required=False)
    # parser.add_argument("--model", default="/data3/private/yyn/t5_pretrained_models/t5.1.1.lm100k.base", required=False)
    parser.add_argument("--model", default="/data/private/yeyining/t5_pretrained_models/t5.1.1.lm100k.base", required=False)
    
    # parser.add_argument("--output_dir", default="/data3/private/yyn/t5_delta_result_for_100tasks/t5_test_out", type=str, required=False)
    parser.add_argument("--output_dir", default="/data3/private/yyn/t5_delta_result_for_100tasks/t5_test_out", type=str, required=False)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_predict", action='store_true')
    #parser.add_argument("--predict_checkpoint", type=str, default="/data3/private/yyn/t5_delta_result_for_100tasks/diffusion_full_data_prompt/singletask-glue-sst2/1/ckpt_1300.pt")
    parser.add_argument("--predict_checkpoint", type=str, default="/data/private/yeyining/diffusion_full_data_prompt_layernorm_unified/singletask-glue-sst2/5/ckpt_2100.pt")

    ## Model parameters
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--do_lowercase", action='store_true', default=False)
    parser.add_argument("--freeze_embeds", action='store_true', default=False)

    # Preprocessing/decoding-related parameters
    parser.add_argument('--max_input_length', type=int, default=512)
    parser.add_argument('--max_output_length', type=int, default=64)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument("--append_another_bos", action='store_true', default=False)

    # Training-related parameters
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--predict_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=0.1, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=100000.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--total_steps", default=100000, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--wait_step', type=int, default=10000000000)

    # Other parameters
    parser.add_argument("--quiet", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--eval_period', type=int, default=2000,
                        help="Evaluate & save model")
    parser.add_argument('--prefix', type=str, default='',
                        help="Prefix for saving predictions")
    parser.add_argument('--debug', action='store_true',
                        help="Use a subset of data for debugging")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # to tune
    parser.add_argument("--learning_rate_list", nargs="*", type=float, default=[])
    parser.add_argument("--bsz_list", nargs="*", type=int, default=[])

    # to prompt tuning
    parser.add_argument("--prompt_num", type=int, default=10)
    parser.add_argument("--do_prompt", action='store_true', help="prompt tuning or not", default=True)
    parser.add_argument("--do_inherit_prompt", action='store_true', help="inherit prompt or not")
    parser.add_argument("--inherit_prompt_path", type=str)

    args = parser.parse_args("")

    #args = parser.parse_args()
    return args

class sst2_data_loader(Dataset):
    def __init__(self,path=None,split="train",tasks=[]):
        if path == None:
            self.tasks = {t:k for k,t in enumerate(tasks)}
            self.id2task = tasks
            self.former_memory = [0] * len(self.tasks)
            self.make_data(tasks)
        else:
            self.encode_data = dict()
            if tasks == []:

                tasks = find_tasks()

            self.tasks = {t:k for k,t in enumerate(tasks)}
            self.id2task = tasks
            self.former_memory = [0] * len(self.tasks)
            self.memory_order = [0] * 100000
            for task in tasks:
                self.encode_data[task] = torch.load(os.path.join(path,f"{task}_embedding_average_pooling.pt"))

                # for id,cont in tqdm(enumerate(self.encode_data[task])):
                #     self.encode_data[task][id]["embedding"] = torch.mean(self.encode_data[task][id]["embedding"],dim=0)
                if split == "train":
                    length = len(self.encode_data[task])
                    self.encode_data[task] = self.encode_data[task][:int(length*0.95)]
                elif split == "test":
                    length = len(self.encode_data[task])
                    self.encode_data[task] = self.encode_data[task][int(length*0.95):]
                else:
                    raise NotImplementedError
    @classmethod
    def make_average(self, path, tasks=[]):
        for task in tasks:
            encode_data=None
            for file in list(glob(path+"/*")):
                if task in file:
                    encode_data = torch.load(file)
                    break
            assert encode_data != None
            for id,cont in tqdm(enumerate(encode_data)):
                encode_data[id]["embedding"] = torch.mean(encode_data[id]["embedding"],dim=0)
            torch.save(encode_data,f"{task}_embedding_average_pooling.pt")
    def map_data(self,task, sentence):
        sentence = sentence.strip()

        if task == "glue-sst2":
            x = sentence.split("	")
            return "".join(x[:-1]) + ". This sentence is " + x[-1]
        elif task == "glue-mnli":

            result = re.match(r"premise: (.+) \[SEP\] hypothesis: (.+)	(.+)",sentence)
            assert result
            fact = result.groups(1)
            hypothesis = result.groups(2)
            label = result.groups(3)
            template = f"{fact} And we hypothesis: {hypothesis}. This is {label}."
            return template
        elif task == "financial_phrasebank":
            x = sentence.split("	")
            return "".join(x[:-1])[:200] + ". This sentence is " + x[-1]
        elif task == "imdb":
            x = sentence.split("	")
            return "".join(x[:-1])[:200] + ". This paragraph is " + x[-1]
        elif task == "yelp_polarity":
            x = sentence.split("	")
            return "".join(x[:-1])[:200] + ". This paragraph is " + x[-1]
        elif task == "poem_sentiment":
            x = sentence.split("	")
            return "".join(x[:-1]) + ". This poem is " + x[-1]
        elif task == "amazon_polarity":
            result = re.match(r"title: (.+) \[SEP\] content: (.+)	(.+)",sentence)
            assert result
            title = result.groups(1)
            content = result.groups(2)[:200]
            label = result.groups(3)
            template = f"Tell me the polarity of this news: {title}. {content} This is {label}"
            return template
        elif task == "rotten_tomatoes":
            x = sentence.split("	")
            return "".join(x[:-1]) + ". This sentence is " + x[-1]
        elif task in ["ethos-gender","ethos-race"]:
            result = re.match(r"(.+)	(.+)",sentence)
            assert result
            template = f"Tell me if this is a hate speech: {result.group(1)[:200]}. The answer is {result.group(2)}."
            return template
        else:
            print(task)
            raise  NotImplementedError

    def make_data(self, tasks=[]):
        prefix = "/data/private/yeyining/data_full_new_100_task/"
        self.encode_data = dict()
        for task in tasks:
            save_path=f"/data/private/yeyining/t5_pretrain_embeddings/{task}_embedding_average_pooling.pt"
            if os.path.exists(save_path):
                print(f"skip task {task}")
                continue
            print(f"doing task {task}")
            self.encode_data[task]= []
            f = open(os.path.join(prefix, task,f"{task}_train.tsv"),"r",encoding="utf-8")
            self.data = [line.strip() for line in f.readlines()]

            for id, x in enumerate(self.data):
                self.data[id] = self.map_data(task,x)
            # self.data = np.loadtxt("/data2/private/yeyining/data_full_new_100_task/glue-sst2/glue-sst2_train.tsv",dtype=np.str,delimiter=None)
            print(f"len of dataset {len(self.data)}")

            instruction_tokenizer = T5Tokenizer.from_pretrained('t5-base')

            t5_encoder = T5EncoderModel.from_pretrained('t5-base').to("cuda")

            pos = 0
            with torch.no_grad():
                with tqdm(total = len(self.data) // 1024) as pbar:
                    while len(self.encode_data[task]) < len(self.data):
                        batch = self.data[pos:pos+1024]

                        pos += 1024
                        x = instruction_tokenizer(batch,return_tensors="pt",max_length=256,padding="max_length",truncation=True).input_ids.to(t5_encoder.device)

                        end = (x != 0)
                        end = torch.sum(end, dim = -1)
                        # print(end)
                        # exit()
                        encode_instruction = t5_encoder(x
                                                        ).last_hidden_state
                        #encode_instruction.to("cpu")

                        

                        for id, (sentence,cont) in enumerate(zip(batch,encode_instruction)):
                            self.encode_data[task].append({"instructions":sentence, "embedding": torch.mean(cont[:end[id]],dim=0)})
                        if len(self.encode_data[task]) > 50000:
                            break

                        del encode_instruction
                        del x
                        torch.cuda.empty_cache()
                        pbar.update(1)

            torch.save(self.encode_data[task],save_path)
        
    def __getitem__(self, index,order=None,random=True):
        index = index % len(self.encode_data)

        embed_id = self.former_memory[index] #正常和上次一样
        if order != None:
            embed_id = self.memory_order[order]

        if random:
            embed_id = torch.randint(0,len(self.encode_data[self.id2task[index]]),size=()).item()
            self.former_memory[index] = embed_id
            if order != None:
                self.memory_order[order] = embed_id
        return self.encode_data[self.id2task[index]][embed_id]
    
    def get_instruction_embeddings_by_task(self,task_list,random=True):
        task_map = {f"singletask-{key}" : key for key in self.tasks.keys()}
        for i in range(len(task_list)):
            if task_map.get(task_list[i]):
                task_list[i] = task_map[task_list[i]]
        # task_list = [task_map[t] for t in task_list]

        cont = [self.__getitem__(self.tasks[task],k,random) for k,task in enumerate(task_list)]

        embeddings = [c["embedding"].cuda() for c in cont]
        instructions = [c["instructions"] for c in cont]
        embeddings  = torch.stack(embeddings)
        return {"embedding":embeddings, "instructions":instructions}

    def __len__(self):
        return len(self.encode_data)

def get_sst2_data(batch_size,split):
    tmp = sst2_data_loader(path="/data/private/yeyining/t5_pretrain_embeddings",tasks=[], split=split)
    return tmp
    #tmp = sst2_data_loader()
    # exit()
    loader = DataLoader(dataset=tmp,batch_size=batch_size,shuffle=True,drop_last=True)
    # x = next(enumerate(loader))
    if split == "train":
        while True:
            for id, cont in enumerate(loader):
                # while True:
                yield cont
    elif split == "test":
        for id, cont in enumerate(loader):
            while True:
                yield cont
    else:
        raise NotImplementedError

# def get_datas(batch_size,loader):
#     pass

@torch.inference_mode()
def inference(model, dev_data, save_predictions=False, verbose=False,leave=False):
    predictions = []
    valid_losses = []
    #bos_token_id = dev_data.tokenizer.bos_token_id
    pbar = enumerate(dev_data.dataloader)
    if is_main_proc():
        pbar = tqdm(enumerate(dev_data.dataloader),total=len(dev_data.dataloader),leave=leave)
    for i, batch in pbar:
        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]
        #print(len(batch))
        pad_token_id = dev_data.tokenizer.pad_token_id
        #batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
        outputs = model.generate(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 #num_beams=dev_data.args.num_beams,
                                 max_length=dev_data.args.max_output_length,
                                 #decoder_start_token_id=model.config.bos_token_id,
                                 early_stopping=True,)
        for input_, output in zip(batch[0], outputs):
            pred = dev_data.decode(output)
            #print("pred:",pred)
            predictions.append(pred)

        outputs = model(input_ids=batch[0], attention_mask=batch[1],
                        labels=batch[2], decoder_attention_mask=batch[3])
        loss = outputs[0]
        valid_losses.append(loss.detach().cpu())

    if save_predictions:
        dev_data.save_predictions(predictions)
    evaluate_results = dev_data.evaluate(predictions,verbose=verbose)
    val = list(evaluate_results.items())[0][1]
    #print("val:",val)
    #print(valid_losses)
    return val, float(np.mean(valid_losses))

def find_tasks():
    tasks = []
    file_names = glob("/data/private/yeyining/t5_pretrain_embeddings/*")
    for file_name in file_names:
        result = re.match(r".*/(.*)_embedding_average_pooling.pt",file_name)
        if result:
            tasks.append(result.group(1))
    return tasks

def sst2_data_fn():
    args = make_args()
    logger = logging.getLogger(__name__)
    tokenizer = T5Tokenizer.from_pretrained(args.model)
    dev_data = {}
    for task_name in find_tasks():
        file_name = os.path.join("/data/private/yeyining/data_eval_100_task/",task_name,f"{task_name}_dev.tsv")
        dev_data[task_name] = NLPFewshotGymSingleTaskData(logger, args, file_name, data_type="dev", is_training=False)
        dev_data[task_name].load_dataset(tokenizer)
        dev_data[task_name].load_dataloader()
    
    # print(len(next(enumerate(dev_data.dataloader))[1]))
    # exit()
    
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    # model.cuda()
    
    return model, dev_data

@torch.inference_mode()
def sst2_test_single_model(t5_model, dev_data, task_name,prompt_fake_model,leave=False):
    # return 1.0
    cur_device = torch.cuda.current_device()
    t5_model.cuda(device=cur_device)
    # print("the code has successfully reached sst2test model stage")
    x1,x2 =  prompt_fake_model.to_prompt()
    t5_model.base_model.encoder.prompt_embeddings.weight.data = (x1).to(t5_model.device)
    t5_model.base_model.encoder.prompt_layer_norm.weight.data = (x2).to(t5_model.device)
    per, loss = inference(t5_model, dev_data[task_name], save_predictions=False, verbose=False,leave=leave)
    t5_model.to("cpu")
    torch.cuda.empty_cache()
    return loss

@torch.inference_mode()
def sst2_test_fn(t5_model, dev_data, prompt_fake_models,tasks_name,leave=False):
    if type(prompt_fake_models) != list:
        return sst2_test_single_model(t5_model, dev_data, tasks_name,prompt_fake_models)
    else:
        result = []
        for prompt_fake_model,task_name in zip(prompt_fake_models,tasks_name):
            result.append(sst2_test_single_model(t5_model, dev_data, task_name,prompt_fake_model,leave = leave))
        return result



if __name__ == "__main__":
    sst2_data_loader(tasks=["imdb","amazon_polarity","financial_phrasebank","poem_sentiment","yelp_polarity","ethos-gender","ethos-race"])
    exit()
    data = sst2_data_loader(path="/data/private/yeyining/t5_pretrain_embeddings",tasks=["rotten_tomatoes","glue-sst2"])
    emb, ins = data.get_instruction_embeddings_by_task(["singletask-rotten_tomatoes","singletask-glue-sst2"])
    print(emb.shape)
    exit()
    # x = get_sst2_data(16)
    # print(next(x))
    # # while True:
    # #     next(x)
    # exit()
    args = make_args()
    logger = logging.getLogger(__name__)


    tokenizer = T5Tokenizer.from_pretrained(args.model)
    dev_data = NLPFewshotGymSingleTaskData(logger, args, args.dev_file, data_type="dev", is_training=False)
    dev_data.load_dataset(tokenizer)
    dev_data.load_dataloader()
    
    # print(next(enumerate(dev_data.dataloader)))
    # exit()
    
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    model.cuda()
    # print("model:",model.encoder.prompt_embeddings)
    
    init_prompt_weight = torch.load(args.predict_checkpoint)
    for cont in init_prompt_weight.keys():
        print(f"{cont}: {init_prompt_weight[cont].size()}")
        print(torch.max(init_prompt_weight[cont]))
        print(torch.min(init_prompt_weight[cont]))
        print(torch.mean(abs(init_prompt_weight[cont])))
    # exit()
    #init_prompt_weight["encoder.prompt_embeddings.weight"] = torch.nn.functional.layer_norm(init_prompt_weight["encoder.prompt_embeddings.weight"],normalized_shape=init_prompt_weight["encoder.prompt_embeddings.weight"].size())
    # print(init_prompt_weight["encoder.prompt_embeddings.weight"].shape)
    # exit()
    #model.base_model.encoder.prompt_embeddings.weight.data = init_prompt_weight["encoder.prompt_embeddings.weight"].cuda()
    model.base_model.encoder.prompt_layer_norm.weight.data = init_prompt_weight["encoder.prompt_layer_norm.weight"].cuda()
    model.base_model.encoder.prompt_embeddings.weight.data = init_prompt_weight["encoder.prompt_embeddings.weight"].cuda()
    # weight_after_laynorm = model.base_model.encoder.prompt_layer_norm(model.base_model.encoder.prompt_embeddings)
    # print(weight_after_laynorm)
    # exit()
    per, loss = inference(model, dev_data, save_predictions=False, verbose=False)
    print(per)
    print(loss)
    exit()
    model.base_model.encoder.prompt_layer_norm.weight.data = init_prompt_weight["encoder.prompt_layer_norm.weight"].cuda() * 2
    model.base_model.encoder.prompt_embeddings.weight.data = init_prompt_weight["encoder.prompt_embeddings.weight"].cuda() / 2
    # model.base_model.encoder.prompt_layer_norm.weight.data = torch.clamp(init_prompt_weight["encoder.prompt_layer_norm.weight"], -500,500).cuda()
    # model.base_model.encoder.prompt_embeddings.weight.data = torch.clamp(init_prompt_weight["encoder.prompt_embeddings.weight"], -500,500).cuda()
    # weight_after_laynorm = model.base_model.encoder.prompt_layer_norm(model.base_model.encoder.prompt_embeddings)
    # print(weight_after_laynorm)
    # exit()
    per, loss = inference(model, dev_data, save_predictions=False, verbose=False)
    print(per)
    print(loss)