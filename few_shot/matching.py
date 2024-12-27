# few_shot/matching.py

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.nn import Module
from torch.nn.modules.loss import _Loss as Loss

from config import EPSILON
from few_shot.core import create_nshot_task_label
from few_shot.utils import pairwise_distances


def matching_net_episode(model: Module,
                         optimiser: Optimizer,
                         loss_fn: Loss,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         n_shot: int,
                         k_way: int,
                         q_queries: int,
                         distance: str,
                         fce: bool,
                         train: bool):
    """Performs a single training episode for a Matching Network.

    # Arguments
        model: Matching Network to be trained.
        optimiser: Optimiser to calculate gradient step from loss
        loss_fn: Loss function to calculate between predictions and outputs
        x: Input samples of few shot classification task
        y: Input labels of few shot classification task
        n_shot: Number of examples per class in the support set
        k_way: Number of classes in the few shot classification task
        q_queries: Number of examples per class in the query set
        distance: Distance metric to use ("cosine", "l2", "dot")
        fce: Whether or not to use fully conditional embeddings
        train: Whether (True) or not (False) to perform a parameter update

    # Returns
        loss: Loss of the Matching Network on this task
        y_pred: Predicted class probabilities for the query set
    """
    if train:
        model.train()
        optimiser.zero_grad()
    else:
        model.eval()

    # 1) Embed all samples
    embeddings = model.encoder(x)

    # 2) Split into support/query
    support = embeddings[:n_shot * k_way]
    queries = embeddings[n_shot * k_way:]

    # 3) Optionally apply FCE (Bidirectional LSTM + Attention LSTM)
    if fce:
        # shape = (k_way * n_shot, embedding_dim) -> add dimension for LSTM
        support, _, _ = model.g(support.unsqueeze(1))
        support = support.squeeze(1)

        queries = model.f(support, queries)

    # 4) Compute pairwise distances
    distances = pairwise_distances(queries, support, distance)
    # Convert distances to attention weights
    attention = (-distances).softmax(dim=1)

    # 5) Weighted sum of support labels
    y_pred = matching_net_predictions(attention, n_shot, k_way, q_queries)

    # 6) Compute negative log-likelihood loss
    clipped_y_pred = y_pred.clamp(EPSILON, 1 - EPSILON)
    loss = loss_fn(clipped_y_pred.log(), y)

    if train:
        # Backprop
        loss.backward()
        # Clip gradients
        clip_grad_norm_(model.parameters(), 1)
        optimiser.step()

    return loss, y_pred


def matching_net_predictions(attention: torch.Tensor, n: int, k: int, q: int) -> torch.Tensor:
    """Calculates Matching Network predictions based on equation (1) of the paper."""
    if attention.shape != (q * k, k * n):
        raise ValueError(
            f'Expecting attention Tensor of shape (q * k, k * n) = ({q * k}, {k * n}), got {attention.shape}')

    # One-hot labels for the support set
    y_onehot = torch.zeros(k * n, k)
    y = create_nshot_task_label(k, n).unsqueeze(-1)
    y_onehot = y_onehot.scatter(1, y, 1)

    # Weighted sum
    y_pred = torch.mm(attention, y_onehot.to(attention.device, dtype=torch.double))

    return y_pred
