"""
This script filters raw checkpoints to remove those with NaN/inf/large weights.
It also estimates the variance of parameters in order to pre-process them for G.pt training.
"""

# try:
#     import isaacgym
# except ImportError:
#     print("WARNING: Isaac Gym not imported")


from Gpt.vis import moduleify
from Diffusion_T5.my_t5_sst_test import sst2_data_fn,sst2_test_fn
from Gpt.tasks import TASK_METADATA
import torch
from tqdm import tqdm
from Gpt.data.dataset_lmdb import ParameterDataset
from Gpt.diffusion import create_diffusion
import shutil
import re
import numpy as np
import lmdb
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_utils")

def compute_align_without_t(x_0,x_1, data_fn, p_0,p_1,mata,task_name):
    lambdas = torch.rand(size=(1,)).item()
    fake_p = p_0 * (1-lambdas) + p_1 * lambdas
    guess_loss = x_0 * (1-lambdas) + x_1 * lambdas

    fake_model = moduleify(fake_p.unsqueeze(0), mata['constructor'], lambda x: x,task_name).cuda()
    model_0 = moduleify(p_0.unsqueeze(0), mata['constructor'], lambda x: x,task_name).cuda()
    real_loss = sst2_test_fn(*data_fn,fake_model,task_name,leave = False)
    real_loss_0 = sst2_test_fn(*data_fn,model_0,task_name,leave = False)

    guess_loss_resize = guess_loss * (real_loss_0 / x_0)

    return abs(guess_loss_resize - real_loss) / real_loss, lambdas, real_loss, guess_loss_resize,real_loss_0

def compute_align(t0,t1,th,x0,x1, data_fn, p_0,p_1,mata):
    task_name = "rotten_tomatoes"
    fake_p = p_0 + (th - t0)/(t1-t0) *(p_1 - p_0) 
    fake_model = moduleify(fake_p.unsqueeze(0), mata['constructor'], lambda x: x,task_name).cuda()
    p1_model = moduleify(p_1.unsqueeze(0), mata['constructor'], lambda x: x,task_name).cuda()
    fake_xh = np.log(x0) + (th - t0)/(t1-t0) *(np.log(x1) - np.log(x0)) 
    fake_xh = np.exp(fake_xh)
    fake_xh_lenear = x0 + (th - t0)/(t1-t0) *(x1 - x0) 

    real_xh = sst2_test_fn(*data_fn,fake_model,task_name,leave = False)
    real_x1 = sst2_test_fn(*data_fn,fake_model,task_name,leave = False)

    fake_xh_resize = fake_xh * (real_x1 / x1)
    # print(f"metric 1: {real_x1} {x1} ")
    # # print(t0,th,t1)
    # print(fake_xh)
    # print(fake_xh_lenear)
    # print(fake_xh_resize )
    # print(real_xh)
    # exit()
    return abs(fake_xh_resize - real_xh) / fake_xh_resize
    pass

def checkpoint_key_to_iter(checkpoint_key: str):
    x = re.findall(r"run\[(\d+)\]_step\[(\d+)\]",checkpoint_key) 
    if x != None:
        return int(x[0][-1])

if __name__ == "__main__":
    torch.set_grad_enabled(False)

    # Example using Cartpole:
    dataset_name = "t5_multitask_test_loss"
    # dataset_dir = "/data3/private/yyn/diffusion_data/Gpt_pretrain_data/checkpoint_datasets/t5_sst2"
    dataset_dir = "/data/private/yeyining/diffusion_data/Gpt_pretrain_dataset/t5_multitask"
    train_metric = "avg_test_loss"
    num_test_runs = 2

    mata = TASK_METADATA[dataset_name]

    check_for_bad_runs = False
    check_delta_alignment = False
    check_different_run_alignment = True
    compute_variance = True
    compute_diffusion_prior = True


    if check_delta_alignment:
        dataset = ParameterDataset(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            train_metric=train_metric,
            num_test_runs=0,
            normalizer_name="none",
            max_step_spacing = 10,
        )
        data_fn = sst2_data_fn()
        num_checkpoints_per_run = 5
        num = 50
        #torch.manual_seed(10)
        for distance in range(2,10):

            rand_runs = torch.randint(low=0, high=dataset.num_runs, size=(num,)).tolist()
            deltas = []
            for run_num in tqdm(rand_runs,leave=False):
                # 统计一次差值的对齐损失,相对值
                for i in range(num_checkpoints_per_run):
                    item = dataset.__getitem__(run_num,distance)
                    # print(item)
                    iter_0 = checkpoint_key_to_iter(item["checkpoint_key_0"])
                    iter_1 = checkpoint_key_to_iter(item["checkpoint_key_1"])
                    iter_half = checkpoint_key_to_iter(item["checkpoint_key_half"])
                    delta = compute_align(iter_0,iter_1,iter_half,item[train_metric+"_0"],item[train_metric+"_1"],data_fn,item["parameters_0"],item["parameters_1"],mata)
                    deltas.append(delta)
            mean_delta = np.mean(deltas)
            std_delta = np.std(deltas)
            print(f"distance={distance}, mean={mean_delta} std={std_delta}")
        exit()

    if check_different_run_alignment:
        dataset = ParameterDataset(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            train_metric=train_metric,
            num_test_runs=2,
            normalizer_name="none",
            max_step_spacing = 10,
        )
        data_fn = sst2_data_fn()
        num_checkpoints_per_run = 100
        num = 50
        for k,task in enumerate(list(dataset.tasks)):
            if task != "glue-sst2":
                continue
            print(f"now checking {task}")
            for i in range(num_checkpoints_per_run):
                x = dataset[k]
                # print(x)
                delta,lambdas,real_loss,guess_loss,real_loss_0 = compute_align_without_t(x[f"{train_metric}_0"],x[f"{train_metric}_1"],data_fn,x["parameters_0"],x["parameters_1"],mata,task)
                print(f"{delta},{lambdas},{x[f'{train_metric}_0']},{real_loss_0},{real_loss},{guess_loss},{x[f'{train_metric}_1']}")
                # exit()
    print("finish computing align")

    if check_for_bad_runs:

        dataset = ParameterDataset(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            train_metric=train_metric,
            num_test_runs=0,
            normalizer_name="none"
        )

        bad_runs = []
        for run in tqdm(range(dataset.num_runs), total=dataset.num_runs):
            #print("Run = {}".format(dataset.runs[run]))
            for iters in range(200):
                try:
                    net = dataset.get_run_network(run, iters)
                except (lmdb.CorruptedError, lmdb.PageNotFoundError):
                    print(f'Bad Run Found (iter={iters}): {dataset.runs[run]}')
                    bad_runs.append(dataset.runs[run])
                    continue
                if not isinstance(net, (torch.FloatTensor, torch.Tensor)):
                    print(f'Bad Run Found (iter={iters}): {dataset.runs[run]}')
                    bad_runs.append(dataset.runs[run])
                    continue
                big_weights = (net.abs().amax() > 10).item()
                illegal_weights = torch.isfinite(net).all().logical_not_().item()
                if big_weights or illegal_weights:
                    print(f'Bad Run Found (iter={iters}): {dataset.runs[run]}')
                    bad_runs.append(dataset.runs[run])
        bad_runs = set(bad_runs)
        print(f"Deleting following bad runs: {list(bad_runs)}")
        for bad_run in bad_runs:
            shutil.rmtree(f"{dataset_dir}/{bad_run}")
        print('Done checking for bad runs.')

        del dataset

    if compute_variance:

        dataset = ParameterDataset(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            train_metric=train_metric,
            num_test_runs=num_test_runs,
            normalizer_name="none",
        )
        for k, task in enumerate(dataset.tasks):

            num_checkpoints_per_run = 200
            num = 25000
            torch.manual_seed(10)
            rand_runs = torch.randint(low=0, high=len(dataset.run_lmdb_path[task]), size=(num,)).tolist()
            # print(rand_runs)
            # rand_iters = []
            # for run in rand_runs:
            #     max_len = len(dataset.run_jsons[run])
            #     print(max_len)
            #     exit()
            # rand_iters = torch.randint(low=0, high=num_checkpoints_per_run, size=(num,)).tolist()
            # runs_and_iters = zip(rand_runs, rand_iters)
            # nets = [dataset.get_run_network(run, iteration) for run, iteration in tqdm(runs_and_iters, total=num)]
            nets = [dataset.get_random_network(k, run) for run in tqdm(rand_runs, total=num)]

            nets = torch.stack(nets)
            stdev = nets.flatten().std(unbiased=True).item()
            oai_coeff = 0.538 / stdev   # 0.538 is the variance of ImageNet pixels scaled to [-1, 1]
            print(f'{task}, Standard Deviation: {stdev}')
            print(f'{task}, OpenAI Coefficient: {oai_coeff}')

        if compute_diffusion_prior:
            diffusion = create_diffusion(
                learn_sigma=False, predict_xstart=True,
                noise_schedule='linear', steps=1000
            )
            prior_kl = diffusion._prior_bpd(nets.cuda() * oai_coeff)
            print(f'{task}, Prior KL: {prior_kl.mean().item()}')
