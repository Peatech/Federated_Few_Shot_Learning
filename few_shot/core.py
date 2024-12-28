# few_shot/core.py

import torch
from torch.utils.data import Sampler
from typing import List, Iterable, Callable, Tuple
import numpy as np
import pandas as pd


# few_shot/core.py

import torch
from torch.utils.data import Sampler, Subset
import pandas as pd
import numpy as np
from typing import List, Iterable, Callable, Tuple


class NShotTaskSampler(Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 episodes_per_epoch: int = None,
                 n: int = None,
                 k: int = None,
                 q: int = None,
                 num_tasks: int = 1,
                 fixed_tasks: List[Iterable[int]] = None):
        """
        PyTorch Sampler subclass that generates n-shot, k-way, q-query tasks.

        # Arguments
            dataset: A dataset or subset. Must have a .df attribute 
                     (if it's a Subset, its .dataset must have a .df).
            episodes_per_epoch: Number of tasks (episodes) per epoch.
            n: n-shot (support samples per class)
            k: k-way (# classes)
            q: q-query (# query samples per class)
            num_tasks: How many tasks in each yielded batch.
            fixed_tasks: Predefined tasks, if any.
        """
        super(NShotTaskSampler, self).__init__(dataset)

        # 1. Unwrap the dataset if it's a Subset
        self.wrapped_dataset = dataset
        if isinstance(dataset, Subset):
            # Subset => unwrap underlying dataset
            self.dataset = dataset.dataset
            self.indices = dataset.indices
        else:
            # Not a subset => entire dataset is used
            self.dataset = dataset
            self.indices = range(len(dataset))

        # 2. Validate dataset has 'df'
        if not hasattr(self.dataset, 'df') or not isinstance(self.dataset.df, pd.DataFrame):
            raise ValueError("The underlying dataset must have a 'df' (pandas.DataFrame).")

        # Basic parameters
        self.episodes_per_epoch = episodes_per_epoch
        self.n = n
        self.k = k
        self.q = q
        self.num_tasks = num_tasks
        self.fixed_tasks = fixed_tasks

        self.i_task = 0

        # Optional check if dataset is large enough for n-shot, k-way, q-query
        if self.k * (self.n + self.q) > len(self.indices):
            raise ValueError(f"Not enough data in this subset to sample {self.k}-way "
                             f"{self.n}-shot tasks. Subset size = {len(self.indices)}.")

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = []

            for _ in range(self.num_tasks):
                # If fixed_tasks is None => random classes
                if self.fixed_tasks is None:
                    # We pick from the underlying dataset's df to get possible classes
                    unique_classes = self.dataset.df['class_id'].unique()
                    if len(unique_classes) < self.k:
                        raise ValueError("Not enough unique classes in this subset to sample k-way tasks.")
                    episode_classes = np.random.choice(unique_classes, size=self.k, replace=False)
                else:
                    # Use a fixed task
                    episode_classes = self.fixed_tasks[self.i_task % len(self.fixed_tasks)]
                    self.i_task += 1

                # df => entire dataset's DataFrame
                # We only want rows that correspond to our subset's indices
                # So we filter df by .isin(self.indices)
                df = self.dataset.df[self.dataset.df.index.isin(self.indices)]
                df = df[df['class_id'].isin(episode_classes)]

                support_k = {cls: None for cls in episode_classes}
                for cls in episode_classes:
                    # sample n support examples
                    support = df[df['class_id'] == cls].sample(self.n)
                    support_k[cls] = support
                    for _, row in support.iterrows():
                        batch.append(row['id'])

                for cls in episode_classes:
                    # sample q query examples
                    query = df[(df['class_id'] == cls) & (~df['id'].isin(support_k[cls]['id']))].sample(self.q)
                    for _, row in query.iterrows():
                        batch.append(row['id'])

            yield np.stack(batch)

    
def prepare_nshot_task(n: int, k: int, q: int, device: torch.device = torch.device('cuda')):
    """
    Returns a function that processes n-shot tasks. 
    Instead of hard-coded .cuda(), we pass a 'device' argument.
    """
    def _prepare_nshot_task(batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        x = x.to(device, dtype=torch.double)
        y = create_nshot_task_label(k, q).to(device)
        return x, y
    return _prepare_nshot_task

def create_nshot_task_label(k: int, q: int) -> torch.Tensor:
    """Label structure: [0]*q + [1]*q + ... + [k-1]*q"""
    y = torch.arange(0, k, 1 / q).long()
    return y
