# few_shot/core.py

import torch
from torch.utils.data import Sampler, Subset  # <-- Add Subset here
import pandas as pd
import numpy as np
from typing import List, Iterable, Callable, Tuple

class NShotTaskSampler(Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 episodes_per_epoch: int,
                 n: int,
                 k: int,
                 q: int,
                 num_tasks: int = 1,
                 fixed_tasks: list = None):
        super().__init__(dataset)

        if isinstance(dataset, Subset):
            self.dataset = dataset.dataset
            self.indices = dataset.indices
        else:
            self.dataset = dataset
            self.indices = range(len(dataset))

        # Check that the underlying dataset has a .df
        if not hasattr(self.dataset, 'df') or not isinstance(self.dataset.df, pd.DataFrame):
            raise ValueError("The underlying dataset must have a 'df' attribute of type pandas.DataFrame.")

        self.episodes_per_epoch = episodes_per_epoch
        self.n = n
        self.k = k
        self.q = q
        self.num_tasks = num_tasks
        self.fixed_tasks = fixed_tasks
        self.i_task = 0

        # OPTIONAL: check total subset size
        if self.k * (self.n + self.q) > len(self.indices):
            raise ValueError(f"Subset size={len(self.indices)} too small for k={self.k}, n={self.n}, q={self.q} tasks.")

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        # Filter the dataset's df by self.indices
        df_subset = self.dataset.df[self.dataset.df.index.isin(self.indices)]

        for _ in range(self.episodes_per_epoch):
            batch = []

            for _ in range(self.num_tasks):
                # 1) Choose classes
                if self.fixed_tasks is None:
                    unique_classes = df_subset['class_id'].unique()
                    if len(unique_classes) < self.k:
                        raise ValueError(
                            "Not enough unique classes in this subset for k-way tasks."
                        )
                    episode_classes = np.random.choice(unique_classes, size=self.k, replace=False)
                else:
                    episode_classes = self.fixed_tasks[self.i_task % len(self.fixed_tasks)]
                    self.i_task += 1

                # 2) For each class, pick n support & q query
                support_k = {}
                for cls in episode_classes:
                    df_class = df_subset[df_subset['class_id'] == cls]
                    if len(df_class) < self.n:
                        raise ValueError(
                            f"Class {cls} doesn't have enough samples in this subset to pick n={self.n}. "
                            f"Available: {len(df_class)}"
                        )
                    support = df_class.sample(self.n, replace=False)
                    support_k[cls] = support

                    for _, row in support.iterrows():
                        batch.append(row['id'])

                for cls in episode_classes:
                    df_class = df_subset[df_subset['class_id'] == cls]
                    # Exclude already used support
                    exclude_ids = support_k[cls]['id'].values
                    df_query = df_class[~df_class['id'].isin(exclude_ids)]
                    if len(df_query) < self.q:
                        raise ValueError(
                            f"Class {cls} doesn't have enough remaining samples "
                            f"to pick q={self.q} after picking support. "
                            f"Remaining: {len(df_query)}"
                        )
                    query = df_query.sample(self.q, replace=False)
                    for _, row in query.iterrows():
                        batch.append(row['id'])

            yield np.array(batch)




    
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
