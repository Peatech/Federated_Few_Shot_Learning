# few_shot/core.py

import torch
from torch.utils.data import Sampler, Subset  # <-- Add Subset here
import pandas as pd
import numpy as np
from typing import List, Iterable, Callable, Tuple

class NShotTaskSampler(Sampler):
    def __init__(self, dataset, episodes_per_epoch, n, k, q):
        self.dataset = dataset
        self.episodes_per_epoch = episodes_per_epoch
        self.n = n
        self.k = k
        self.q = q

        # Check if dataset has 'df' attribute
        if hasattr(dataset, 'df') and isinstance(dataset.df, pd.DataFrame):
            self.indices = dataset.df.index.values
        else:
            raise ValueError("Dataset must have a 'df' attribute of type DataFrame.")

        # Ensure there are enough samples for n, k, q
        if len(self.indices) < k * (n + q):
            raise ValueError(f"Dataset size={len(self.indices)} too small for k={k}, n={n}, q={q} tasks.")

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            # Sample 'k' classes
            sampled_classes = np.random.choice(self.dataset.df['class_id'].unique(), size=self.k, replace=False)

            support_indices = []
            query_indices = []

            for cls in sampled_classes:
                class_samples = self.dataset.df[self.dataset.df['class_id'] == cls].sample(self.n + self.q).index.values
                support_indices.extend(class_samples[:self.n])
                query_indices.extend(class_samples[self.n:])

            yield support_indices + query_indices

    def __len__(self):
        return self.episodes_per_epoch


    
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
