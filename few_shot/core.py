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
        """Generates batches of n-shot, k-way, q-query tasks."""
        super().__init__(dataset)
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset
        if num_tasks < 1:
            raise ValueError('num_tasks must be > 1.')

        self.num_tasks = num_tasks
        self.k = k
        self.n = n
        self.q = q
        self.fixed_tasks = fixed_tasks
        self.i_task = 0

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        # Ensure we get the actual dataset if it's a Subset
        dataset = self.dataset.dataset if isinstance(self.dataset, torch.utils.data.Subset) else self.dataset
    
        # Validate dataset compatibility
        if not hasattr(dataset, 'df') or not isinstance(dataset.df, pd.DataFrame):
            raise ValueError("Dataset passed to NShotTaskSampler must have a 'df' attribute of type pandas.DataFrame")
    
        for _ in range(self.episodes_per_epoch):
            batch = []
    
            for task in range(self.num_tasks):
                if self.fixed_tasks is None:
                    # Get random classes
                    episode_classes = np.random.choice(dataset.df['class_id'].unique(), size=self.k, replace=False)
                else:
                    # Loop through classes in fixed_tasks
                    episode_classes = self.fixed_tasks[self.i_task % len(self.fixed_tasks)]
                    self.i_task += 1
    
                df = dataset.df[dataset.df['class_id'].isin(episode_classes)]
    
                support_k = {k: None for k in episode_classes}
                for k in episode_classes:
                    # Select support examples
                    support = df[df['class_id'] == k].sample(self.n)
                    support_k[k] = support
    
                    for i, s in support.iterrows():
                        batch.append(s['id'])
    
                for k in episode_classes:
                    query = df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))].sample(self.q)
                    for i, q in query.iterrows():
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
