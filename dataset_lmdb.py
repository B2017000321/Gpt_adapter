"""
The main Dataset class for training/testing on checkpoint data (parameters, metrics, etc.).
"""
import json
import os
import random

from dataclasses import dataclass
from glob import glob
from typing import Any, Dict

import torch
import numpy as np
from torch.utils.data import Dataset

from .database import Database
from .augment import random_permute_flat
from .normalization import get_normalizer
from Gpt.tasks import TASK_METADATA
from Gpt.download import find_checkpoints


@dataclass
class ParameterDataset(Dataset):
    dataset_dir: str = "/data3/private/yyn/diffusion_data/Gpt_pretrain_data/checkpoint_datasets/mnist"  # Path to the checkpoint dataset
    dataset_name: str = "mnist_loss"                # Name of the dataset in tasks.py
    split: str = "train"                            # Split of the checkpoint dataset to use ("train" or "test")
    max_train_runs: int = 1000000                   # Maximum number of runs to train on (default: all)
    num_test_runs: int = 500                        # Number of runs in the test split
    target_epoch_size: int = 806400                 # Amount of data to train on per-"epoch" (806400 is arbitrary)
    train_metric: str = "avg_test_loss"             # Conditional metric for G.pt
    min_step_spacing: int = 1                       # Minimum spacing between starting and future checkpoints
    max_step_spacing: int = None                    # Maximum spacing between starting and future checkpoints
    normalizer_name: str = "openai"                 # Parameter normalization algorithm
    openai_coeff: float = 4.185                      # Scaling coefficient for the "openai" DALL-E 2 normalizer
    single_run_debug: bool = False                  # Option to debug with a single run
    min_val: float = None                            # Minimum value of parameters (usually passed to test dataset)
    max_val: float = None                            # Maximum value of parameters (usually passed to test dataset)
    permute_augment: bool = False                   # if True, applies permutation augmentation to parameters
    verify_done: bool = False                       # if True, filters runs that don't have a DONE.txt file
    download: bool = False                           # If True, auto-downloads and caches the checkpoint dataset

    def __post_init__(self):

        # Auto-download the dataset:
        if self.download:
            find_checkpoints(self.dataset_dir)

        # Find all the LDMB directories:
        lmdb_paths = sorted(list(glob(f'{self.dataset_dir}/*')))
        if len(lmdb_paths) == 0:
            raise FileNotFoundError

        # Filter out runs missing DONE.txt files if needed:
        if self.verify_done:
            lmdb_paths = [p for p in lmdb_paths if os.path.exists(os.path.join(p, "DONE.txt"))]


        # Build run to lmdb path map:
        run_lmdb_path = dict()
        for lmdb_path in lmdb_paths:
            task_name = os.path.basename(lmdb_path)
            run_lmdb_path[task_name] = dict()
            for single_task_path in sorted(list(glob(f"{lmdb_path}/*"))):
                # print(single_task_path)
                lmdb_runs = os.path.basename(single_task_path).split("_")

                for run in lmdb_runs:
                    run_lmdb_path[task_name][run] = single_task_path
        # print(run_lmdb_path)
        # exit()
        self.tasks = list(run_lmdb_path.keys())
        self.task2id = {task: k for k,task in enumerate(self.tasks)}
        self.runs = list(run_lmdb_path.keys())

        # Perform the train/test split:

        assert self.split in ["train", "test"]
        for task in self.tasks:
            assert self.num_test_runs < len(run_lmdb_path[task]), f"{task}, {len(run_lmdb_path[task])}"

        self.run_lmdb_path = dict()
        self.run_index2lmbd_id = dict()
        for task in self.tasks:
            task_run_keys = list(run_lmdb_path[task].keys())
            if self.split == "train":
                task_run_keys = task_run_keys[:-self.num_test_runs] if self.num_test_runs > 0 else task_run_keys
                task_run_keys = task_run_keys[:self.max_train_runs]
            else:
                task_run_keys = task_run_keys[-self.num_test_runs:] if self.num_test_runs > 0 else []
        # Delete the runs we aren't using:
            self.run_lmdb_path[task] = {run: run_lmdb_path[task][run] for run in task_run_keys}
            self.run_index2lmbd_id[task] = list(self.run_lmdb_path[task].keys())
            # print(self.run_lmdb_path[task])
            # lmdb_paths = list(self.run_lmdb_path.values())
            print(f"(split={self.split}),task={task} number of runs: {len(self.run_lmdb_path[task])}")

        # Read run jsons containing various metadata:
        self.run_jsons = dict()
        for task in self.tasks:
            self.run_jsons[task] = []
            for run in self.run_index2lmbd_id[task]:
                lmdb_path = self.run_lmdb_path[task][run]
                json_path = os.path.join(lmdb_path, "runs.json")
                with open(json_path, "r") as f:
                    json_data = json.load(f)
                # Extract run json in case of fused jsons
                if run in json_data:

                    json_data = json_data[run]
                self.run_jsons[task].append(json_data)

        # assert len(self.runs) == len(self.run_jsons[task])

        # Build map of LMDB paths to LMDB objects:

        self.databases = dict()
        for task in self.tasks:
            self.databases[task] = {
                lmdb_path_item[1]: Database(path=lmdb_path_item[1], map_size=10485760) for lmdb_path_item in self.run_lmdb_path[task].items()
            }
        
        self.num_runs = len(self.tasks)
        # self.lmdb_paths = lmdb_paths
        self.architecture = TASK_METADATA[self.dataset_name]['constructor']()
        self.parameter_sizes = self.get_database(0,0)[\
            f'{list(self.run_jsons[self.tasks[0]][0]["checkpoints"].keys())[0]}_arch'].long().tolist()
        print(self.parameter_sizes)
        self.parameter_names = list(self.architecture.state_dict().keys())
        print(self.parameter_names)
        assert len(self.parameter_names) == len(self.parameter_sizes)

        self.num_checkpoints = dict()
        for task in self.tasks:
            self.num_checkpoints[task] = [len(run_json['checkpoints']) for run_json in self.run_jsons[task]]

        self.generator = None

        # Setup parameter normalization:
        reduce_fn = min if TASK_METADATA[self.dataset_name]['minimize'] else max
        self.optimal_test_loss = self.reduce_metadata(
            f'optimal_{self.train_metric}'.replace('_avg', ''), reduce_fn=reduce_fn
        )
        self.min_val, self.max_val = self.get_range(normalize=False)
        self.normalizer = {task: get_normalizer(self.normalizer_name, openai_coeff=self.openai_coeff,
                                         min_val=self.min_val[task], max_val=self.max_val[task], dataset=self) for task in self.tasks}
        for task in self.tasks:
            print(f"(split={self.split}),task={task} using normalizer={self.normalizer[task].message()}")
            print(f'(split={self.split}),task={task} max-val: {self.max_val[task]}, min-val: {self.min_val[task]}')
            print(f'(split={self.split}),task={task} optimal-{self.train_metric}: {self.optimal_test_loss[task]}')

        # Setup parameter augmentation if needed:
        self.make_aug_fn()

        # Ensure that epoch_size is perfectly divisible by the number of runs:
        assert self.target_epoch_size >= self.num_runs, \
            f"target_epoch_size ({self.target_epoch_size}) " \
            f"is less than the number of runs ({self.num_runs})"
        self.epoch_size = self.num_runs * (self.target_epoch_size // self.num_runs)

        self.cache_run = [0 for i in self.tasks]

    def get_database(self, task_index, run_index=None):
        # print(f"{run_index}: {self.run_lmdb_path[self.runs[run_index]]}")
        if run_index == None:
            return self.databases[self.tasks[task_index]]
        else:
            task_name = self.tasks[task_index]
            real_run_index = self.run_lmdb_path[task_name][ self.run_index2lmbd_id[task_name][run_index] ]
        
            return self.databases[task_name][real_run_index]

    def make_aug_fn(self):
        task_dict = TASK_METADATA[self.dataset_name]
        self.use_augment = self.permute_augment and 'aug_fn' in task_dict
        if self.use_augment:
            self.task_aug_fn = task_dict['aug_fn']
            print(f'(split={self.split}) Using augmentation')
        else:
            print(f'(split={self.split}) NOT using augmentation')

    def aug_fn(self, p, seed=None):
        if self.use_augment:
            return random_permute_flat(p, self.architecture, seed, self.task_aug_fn)
        else:
            return p

    def get_range(self, normalize=True):
        if self.min_val is None and self.max_val is None:
            min_val = self.reduce_metadata('min_parameter_val', reduce_fn=min)
            max_val = self.reduce_metadata('max_parameter_val', reduce_fn=max)
        else:
            min_val, max_val = self.min_val, self.max_val
        if normalize:
            # If normalize=True, this returns the range of normalized parameter values
            assert hasattr(self, "normalizer"), "normalizer hasn't been instantiated yet"
            for key in min_val.keys():
                min_val[key], max_val[key] = self.normalizer[key].get_range(min_val[key], max_val[key])
        return min_val, max_val

    def normalize(self, task_name,weights):
        return self.normalizer[task_name].normalize(weights)

    def unnormalize(self, task_name,normalized_weights):
        if type(task_name) == list:
            result = []
            for tn,nw in zip(task_name,normalized_weights):
                if not tn in self.normalizer.keys():
                    tn = list(self.normalizer.keys())[0]
                    result.append(self.normalizer[tn].unnormalize(nw))
            return torch.stack(result)
        else:
            if not task_name in self.normalizer.keys():
                task_name = list(self.normalizer.keys())[0]
            return self.normalizer[task_name].unnormalize(normalized_weights)



    def reduce_metadata(self, key, reduce_fn=max):
        # Applies a reduction function over all runs in this split
        re = {}
        for task in self.tasks:
            re[task] = reduce_fn(run_json['metadata'][key] for run_json in self.run_jsons[task])
        return re

    def get_run_losses(self, task_index,run_index: int):
        if self.single_run_debug:
            run_index = 0
        metadata = self.run_jsons[self.tasks[task_index]][run_index]['metadata']
        test_losses = torch.tensor(metadata[self.train_metric])
        return test_losses

    def get_random_network(self,task_index,run_index):
        rand_x = random.randint(0,len(self.run_jsons[self.tasks[task_index]][run_index]))
        return self.get_run_network(task_index, run_index, rand_x)
    def get_optimal_network(self,task_index):
        run_checkpoint_names = sorted(list(self.run_jsons[self.tasks[task_index]][0]['checkpoints'].keys()))
        iters = run_checkpoint_names[-1]
        print(f"get {self.tasks[task_index]} {iters}")
        return self.get_run_network(task_index, 0, -1)

    def get_run_network(self, task_index, run_index, iter=0, normalize=True, augment=False):
        if self.single_run_debug:
            run_index = 0
        run_checkpoint_names = sorted(list(self.run_jsons[self.tasks[task_index]][run_index]['checkpoints'].keys()))
        checkpoint_name = run_checkpoint_names[iter]
        database = self.get_database(task_index,run_index)
        parameters = database[checkpoint_name]
        if normalize:
            parameters = self.normalize(self.tasks[task_index],parameters)
        if augment:
            parameters = self.aug_fn(parameters)
        return parameters

    def linear_two_parameter(self, p1,p2,v1,v2):
        a = torch.rand(
            size=(),
            generator=self.generator,
        ).item()

        p = a * p1 + (1-a) * p2
        v = a * np.log(v1) + (1 - a ) * np.log(v2)
        v = np.exp(v)
        return p, v, a

    def __getitem__(self, task_index: int, distance: int = 5):
        task_index = task_index % len(self.tasks)
        task_name = self.tasks[task_index]
        runs_in_task = len(self.run_lmdb_path[task_name])
        #run_index = torch.randint(0,runs_in_task,size=(),generator= self.generator).item()
        run_index = self.cache_run[task_index]
        self.cache_run[task_index] = (self.cache_run[task_index] + 1 ) % runs_in_task
        if self.single_run_debug:
            run_index = 0

        run_checkpoints = self.run_jsons[task_name][run_index]['checkpoints']

        num_checkpoints = self.num_checkpoints[task_name][run_index]
        max_step_spacing = (num_checkpoints - 1) if self.max_step_spacing is None else self.max_step_spacing
        checkpoint_names = list(self.run_jsons[task_name][run_index]['checkpoints'].keys())

        step_spacing = torch.randint(
            low = 2,
            high = max_step_spacing,
            size=(),
            generator=self.generator,
        ).item()
        step_spacing_half = torch.randint(
            low=1,
            high=min(step_spacing,distance),
            size=(),
            generator=self.generator,
        ).item()

        checkpoint_index_0 = torch.randint(
            low=0,
            high=num_checkpoints - step_spacing,
            size=(),
            generator=self.generator,
        ).item()
        if self.single_run_debug:
            checkpoint_index_0 = 0
        checkpoint_index_1 = checkpoint_index_0 + step_spacing
        checkpoint_index_half = checkpoint_index_1 - step_spacing_half
        # print(f"checkpoint_index_1={checkpoint_index_1}, checkpoint_index_0={checkpoint_index_0} ,checkpoint_index_half={checkpoint_index_half}")

        run_checkpoint_name_0 = checkpoint_names[checkpoint_index_0]
        run_checkpoint_name_1 = checkpoint_names[checkpoint_index_1]
        run_checkpoint_name_half = checkpoint_names[checkpoint_index_half]

        run_metrics_0 = run_checkpoints[run_checkpoint_name_0]
        run_metrics_1 = run_checkpoints[run_checkpoint_name_1]
        run_metrics_half = run_checkpoints[run_checkpoint_name_half]

        database = self.get_database(task_index,run_index)
        parameters_0 = database[run_checkpoint_name_0]
        parameters_1 = database[run_checkpoint_name_1]
        parameters_half = database[run_checkpoint_name_half]

        # print(parameters_0)
        # exit()

        parameters_0 = self.normalize(task_name,parameters_0)
        parameters_1 = self.normalize(task_name,parameters_1)
        parameters_half = self.normalize(task_name,parameters_half)



        parameters_0, parameters_1, parameters_half = self.aug_fn((parameters_0, parameters_1,parameters_half), seed=None)



        outputs = {
            "parameters_0": parameters_0,
            "parameters_1": parameters_1,
            "parameters_half": parameters_half,
            "checkpoint_key_0": run_checkpoint_name_0,
            "checkpoint_key_1": run_checkpoint_name_1,
            "checkpoint_key_half": run_checkpoint_name_half,
            "step_spacing": step_spacing,
            "task_name": task_name,
        }

        for metric in [self.train_metric]:

            outputs[f"{metric}_0"] = run_metrics_0[metric]
            outputs[f"{metric}_1"] = run_metrics_1[metric]
            outputs[f"{metric}_half"] = run_metrics_half[metric]

        p,v,a = self.linear_two_parameter(parameters_half,parameters_1, outputs[f"{self.train_metric}_half"], outputs[f"{self.train_metric}_1"])
        outputs[f"{self.train_metric}_1"] = v
        outputs[f"parameters_1"] = p

        return outputs

    def __getitem__different_run(self, task_index: int) -> Dict[str, Any]:
        task_index = task_index % len(self.tasks)
        if self.single_run_debug:
            task_index = 0

        task_name = self.tasks[task_index]

        random_two_runs = torch.randint(
            low=0,
            high=len(self.run_lmdb_path[task_name]),
            size=(2,),
            generator=self.generator,
        ) #也可以是一次run


        run_checkpoints_0 = self.run_jsons[task_name][random_two_runs[0]]['checkpoints']
        run_checkpoints_1 = self.run_jsons[task_name][random_two_runs[1]]['checkpoints']

        num_checkpoints_0 = self.num_checkpoints[task_name][random_two_runs[0]]
        num_checkpoints_1 = self.num_checkpoints[task_name][random_two_runs[1]]

        # max_step_spacing = (num_checkpoints - 1) if self.max_step_spacing is None else self.max_step_spacing
        checkpoint_names_0 = list(self.run_jsons[task_name][random_two_runs[0]]['checkpoints'].keys())
        checkpoint_names_1 = list(self.run_jsons[task_name][random_two_runs[1]]['checkpoints'].keys())

        checkpoint_index_0 = torch.randint(
            low=0,
            high=num_checkpoints_0,
            size=(),
            # generator=self.generator,
        ).item()
        checkpoint_index_1 = torch.randint(
            low=0,
            high=num_checkpoints_1,
            size=(),
            # generator=self.generator,
        ).item()

        if self.single_run_debug:
            checkpoint_index_0 = 0

        run_checkpoint_name_0 = checkpoint_names_0[checkpoint_index_0]
        run_checkpoint_name_1 = checkpoint_names_1[checkpoint_index_1]

        run_metrics_0 = run_checkpoints_0[run_checkpoint_name_0]
        run_metrics_1 = run_checkpoints_1[run_checkpoint_name_1]

        database_0 = self.get_database(task_index, random_two_runs[0])
        database_1 = self.get_database(task_index, random_two_runs[1])
        parameters_0 = database_0[run_checkpoint_name_0]
        parameters_1 = database_1[run_checkpoint_name_1]

        # print(parameters_0)
        # exit()
        parameters_0 = self.normalize(self.tasks[task_index],parameters_0)
        parameters_1 = self.normalize(self.tasks[task_index],parameters_1)

        parameters_0, parameters_1 = self.aug_fn((parameters_0, parameters_1), seed=None)

        metric_0, metric_1 = run_metrics_0[self.train_metric], run_metrics_1[self.train_metric]

        if (metric_0 < metric_1 and TASK_METADATA[self.dataset_name]['minimize']) or (metric_0 > metric_1 and not TASK_METADATA[self.dataset_name]['minimize']):
            parameters_0,parameters_1 = parameters_1,parameters_0
            run_checkpoint_name_0,run_checkpoint_name_1 = run_checkpoint_name_1, run_checkpoint_name_0
            random_two_runs[0], random_two_runs[1] =  random_two_runs[1], random_two_runs[0]
            metric_0,metric_1 = metric_1,metric_0

        outputs = {
            "parameters_0": parameters_0,
            "parameters_1": parameters_1,
            "checkpoint_key_0": run_checkpoint_name_0,
            "checkpoint_key_1": run_checkpoint_name_1,
            "run_name": random_two_runs,
            "task_name": task_name,
        }

        for metric in [self.train_metric]:
            outputs[f"{metric}_0"] = metric_0
            outputs[f"{metric}_1"] = metric_1

        return outputs

    def __getitem__old(self, run_index: int) -> Dict[str, Any]:
        run_index = run_index % len(self.runs)
        if self.single_run_debug:
            run_index = 0

        run_name = self.runs[run_index]
        run_checkpoints = self.run_jsons[run_index]['checkpoints']

        num_checkpoints = self.num_checkpoints[run_index]
        max_step_spacing = (num_checkpoints - 1) if self.max_step_spacing is None else self.max_step_spacing
        checkpoint_names = list(self.run_jsons[run_index]['checkpoints'].keys())

        step_spacing = torch.randint(
            low=self.min_step_spacing,
            high=max_step_spacing + 1,
            size=(),
            generator=self.generator,
        ).item()

        checkpoint_index_0 = torch.randint(
            low=0,
            high=num_checkpoints - step_spacing,
            size=(),
            generator=self.generator,
        ).item()
        if self.single_run_debug:
            checkpoint_index_0 = 0
        checkpoint_index_1 = checkpoint_index_0 + step_spacing

        run_checkpoint_name_0 = checkpoint_names[checkpoint_index_0]
        run_checkpoint_name_1 = checkpoint_names[checkpoint_index_1]

        run_metrics_0 = run_checkpoints[run_checkpoint_name_0]
        run_metrics_1 = run_checkpoints[run_checkpoint_name_1]

        database = self.get_database(run_index)
        parameters_0 = database[run_checkpoint_name_0]
        parameters_1 = database[run_checkpoint_name_1]

        # print(parameters_0)
        # exit()
        parameters_0 = self.normalize(parameters_0)
        parameters_1 = self.normalize(parameters_1)

        parameters_0, parameters_1 = self.aug_fn((parameters_0, parameters_1), seed=None)


        outputs = {
            "parameters_0": parameters_0,
            "parameters_1": parameters_1,
            "checkpoint_key_0": run_checkpoint_name_0,
            "checkpoint_key_1": run_checkpoint_name_1,
            "run_name": run_name,
            "step_spacing": step_spacing,
        }

        for metric in [self.train_metric]:
            outputs[f"{metric}_0"] = run_metrics_0[metric]
            outputs[f"{metric}_1"] = run_metrics_1[metric]
        assert abs(outputs[f"{metric}_0"]) < 1e5, "0 ji"
        assert abs(outputs[f"{metric}_1"]) < 1e5, "1 ji"
        return outputs

    def __len__(self) -> int:
        return self.epoch_size
