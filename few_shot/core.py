# few_shot/core.py

import torch
from torch.utils.data import Sampler
from typing import List, Iterable, Callable, Tuple
import numpy as np

from few_shot.metrics import categorical_accuracy
from few_shot.callbacks import Callback

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
        for _ in range(self.episodes_per_epoch):
            batch = []

            for task in range(self.num_tasks):
                if self.fixed_tasks is None:
                    # random classes
                    episode_classes = np.random.choice(
                        self.dataset.df['class_id'].unique(),
                        size=self.k, replace=False
                    )
                else:
                    # use fixed_tasks
                    episode_classes = self.fixed_tasks[self.i_task % len(self.fixed_tasks)]
                    self.i_task += 1

                df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]

                support_k = {cls: None for cls in episode_classes}
                for cls in episode_classes:
                    # Select support
                    support = df[df['class_id'] == cls].sample(self.n)
                    support_k[cls] = support

                    for i, s in support.iterrows():
                        batch.append(s['id'])

                for cls in episode_classes:
                    query = df[
                        (df['class_id'] == cls) &
                        (~df['id'].isin(support_k[cls]['id']))
                    ].sample(self.q)
                    for i, q_ in query.iterrows():
                        batch.append(q_['id'])

            yield np.stack(batch)

class EvaluateFewShot(Callback):
    """
    Evaluate a network on an n-shot, k-way classification task after every epoch.
    In federated learning, you could adapt it to evaluate after each round.
    """
    def __init__(self,
                 eval_fn: Callable,
                 num_tasks: int,
                 n_shot: int,
                 k_way: int,
                 q_queries: int,
                 taskloader: torch.utils.data.DataLoader,
                 prepare_batch: Callable,
                 prefix: str = 'val_',
                 **kwargs):
        super().__init__()
        self.eval_fn = eval_fn
        self.num_tasks = num_tasks
        self.n_shot = n_shot
        self.k_way = k_way
        self.q_queries = q_queries
        self.taskloader = taskloader
        self.prepare_batch = prepare_batch
        self.prefix = prefix
        self.kwargs = kwargs
        self.metric_name = f'{self.prefix}{self.n_shot}-shot_{self.k_way}-way_acc'

    def on_train_begin(self, logs=None):
        self.loss_fn = self.params['loss_fn']
        self.optimiser = self.params['optimiser']

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        seen = 0
        totals = {'loss': 0, self.metric_name: 0}

        for batch_index, batch in enumerate(self.taskloader):
            x, y = self.prepare_batch(batch)
            # Note: If you want to handle CPU clients, replace .cuda() with .to(device)
            # or unify device. We'll keep this as is for demonstration.

            # evaluate
            loss, y_pred = self.eval_fn(
                self.model,
                self.optimiser,
                self.loss_fn,
                x,
                y,
                n_shot=self.n_shot,
                k_way=self.k_way,
                q_queries=self.q_queries,
                train=False,
                **self.kwargs
            )
            seen += y_pred.shape[0]
            totals['loss'] += loss.item() * y_pred.shape[0]
            totals[self.metric_name] += categorical_accuracy(y, y_pred) * y_pred.shape[0]

        logs[f'{self.prefix}loss'] = totals['loss'] / seen
        logs[self.metric_name] = totals[self.metric_name] / seen

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
