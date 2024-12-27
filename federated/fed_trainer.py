###############################################
# federated/fed_trainer.py
###############################################
import copy
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Subset

# We'll import the matching_net_episode for episodic updates
from few_shot.matching import matching_net_episode
# We'll import NShotTaskSampler for local episodic batches
from few_shot.core import NShotTaskSampler
# We'll import your datasets
from few_shot.datasets import OmniglotDataset, MiniImageNet
# We'll import the data splitting logic
from federated.fed_data_splitting import get_user_groups

def run_federated_few_shot(args, global_model):
    """
    Orchestrates a FedAVG approach for few-shot tasks (Omniglot or miniImageNet).
    """
    device = next(global_model.parameters()).device
    print(f"[FED] Starting federated few-shot training on {args.dataset} (device={device})")

    # 1) Load dataset
    if args.dataset.lower() == 'omniglot':
        dataset = OmniglotDataset('background')
    elif args.dataset.lower() == 'miniimagenet':
        dataset = MiniImageNet('background')
    else:
        raise ValueError("Unsupported dataset in fed_trainer")

    # 2) Split among users
    user_groups = get_user_groups(dataset, args.num_users, iid=(args.iid==1))

    # 3) FedAVG main loop
    global_weights = global_model.state_dict()

    for round_idx in range(args.epochs):
        print(f"\n--- [FED] Global Round {round_idx+1}/{args.epochs} ---")

        selected_users = _select_clients(args.num_users, args.frac)
        local_weights = []
        local_losses = []

        for user_id in selected_users:
            w, loss = local_update_few_shot(
                args, copy.deepcopy(global_model), dataset, user_groups[user_id], device
            )
            local_weights.append(w)
            local_losses.append(loss)

        # Aggregate
        global_weights = _fed_avg_aggregate(local_weights)
        global_model.load_state_dict(global_weights)

        avg_loss = sum(local_losses) / len(local_losses) if len(local_losses) > 0 else 0
        print(f"[FED] Round {round_idx+1} average local loss: {avg_loss:.4f}")

    # (Optional) Evaluate final global model
    print("[FED] Training complete. Evaluate or save the global_model as needed.")


def _select_clients(num_users, frac):
    """
    Select a fraction of users each round (simple random approach).
    """
    m = max(int(frac * num_users), 1)
    selected = np.random.choice(range(num_users), m, replace=False)
    print(f"[FED] Selected users: {selected}")
    return selected


def local_update_few_shot(args, local_model, dataset, user_idxs, device):
    """
    Simulates local training on a single client with episodic few-shot approach.
    """
    optimizer = torch.optim.SGD(local_model.parameters(), lr=args.lr, momentum=args.momentum)
    local_losses = []

    # Create a Subset for this client's data
    user_subset = Subset(dataset, user_idxs)

    # We'll do local training for 'args.local_ep' episodes,
    # each drawn from the NShotTaskSampler
    sampler = NShotTaskSampler(
        dataset=user_subset,
        episodes_per_epoch=args.local_ep,  # e.g. if local_ep=3, we do 3 episodes
        n=args.n_train,  # number of support examples
        k=args.k_train,  # number of classes in each episode
        q=args.q_train,  # number of query samples
        num_tasks=1
    )
    loader = DataLoader(user_subset, batch_sampler=sampler, num_workers=0)

    local_model.train()
    loss_fn = nn.NLLLoss().to(device)

    for episode_idx, batch in enumerate(loader):
        x, y = batch  # x,y shaped for n-shot
        x, y = x.to(device), y.to(device)

        # calling matching_net_episode for a single forward/backward pass
        loss, y_pred = matching_net_episode(
            model=local_model,
            optimiser=optimizer,
            loss_fn=loss_fn,
            x=x,
            y=y,
            n_shot=args.n_train,
            k_way=args.k_train,
            q_queries=args.q_train,
            distance=args.distance,
            fce=args.fce,
            train=True  # triggers backprop
        )
        local_losses.append(loss.item())

    # Return updated weights and average local loss
    new_weights = copy.deepcopy(local_model.state_dict())
    avg_loss = sum(local_losses)/len(local_losses) if len(local_losses)>0 else 0
    return new_weights, avg_loss


def _fed_avg_aggregate(local_weights):
    """
    Basic FedAVG aggregator to average model weights.
    """
    if len(local_weights) == 0:
        return None
    w_avg = copy.deepcopy(local_weights[0])
    for key in w_avg.keys():
        for i in range(1, len(local_weights)):
            w_avg[key] += local_weights[i][key]
        w_avg[key] = torch.div(w_avg[key], len(local_weights))
    return w_avg
