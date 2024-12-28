# few_shot/core.py

import torch
from torch.utils.data import Sampler
from typing import List, Iterable, Callable, Tuple
import numpy as np
import pandas as pd


class NShotTaskSampler(Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 episodes_per_epoch: int = None,
                 n: int = None,
                 k: int = None,
                 q: int = None,
                 num_tasks: int = 1,
                 fixed_tasks: List[Iterable[int]] = None):
        """PyTorch Sampler subclass for n-shot, k-way, q-query tasks."""
        super(NShotTaskSampler, self).__init__(dataset)

        # Handle Subset datasets
        if isinstance(dataset, torch.utils.data.Subset):
            self.dataset_indices = dataset.indices
        else:
            self.dataset_indices = range(len(dataset))

        self.dataset = dataset
        self.episodes_per_epoch = episodes_per_epoch
        self.num_tasks = num_tasks
        self.k = k
        self.n = n
        self.q = q
        self.fixed_tasks = fixed_tasks

        if self.k * (self.n + self.q) > len(self.dataset_indices):
            raise ValueError(f"Dataset is too small for {self.k}-way {self.n}-shot tasks.")

        self.i_task = 0

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = []

            for _ in range(self.num_tasks):
                if self.fixed_tasks is None:
                    # Randomly sample k classes
                    episode_classes = np.random.choice(
                        self.dataset.df['class_id'].unique(), size=self.k, replace=False
                    )
                else:
                    # Use fixed tasks
                    episode_classes = self.fixed_tasks[self.i_task % len(self.fixed_tasks)]
                    self.i_task += 1

                df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]
                support_k = {k: None for k in episode_classes}

                for k in episode_classes:
                    support = df[df['class_id'] == k].sample(self.n)
                    support_k[k] = support

                    for _, s in support.iterrows():
                        batch.append(s['id'])

                for k in episode_classes:
                    query = df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))].sample(self.q)
                    for _, q in query.iterrows():
                        batch.append(q['id'])

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
