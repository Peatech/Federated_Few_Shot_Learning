###############################################
# federated/fed_trainer.py
# Implements FedAVG for Few-Shot training
###############################################
import copy
import torch
import numpy as np

from torch import nn
from torch.utils.data import DataLoader, Subset

# If you have a script like "few_shot.matching" that has matching_net_episode:
from few_shot.matching import matching_net_episode
# If you want to do local episodes with NShotTaskSampler
from few_shot.core import NShotTaskSampler
# If you want to load Omniglot or MiniImageNet
from few_shot.datasets import OmniglotDataset, MiniImageNet

# For demonstration, we define all code here. You might break it into 
# separate files for data splitting, local update logic, etc.

def run_federated_few_shot(args, global_model):
    """
    Orchestrates a canonical FedAVG approach for a few-shot model.
    
    Steps:
      1) Load dataset, do user data splits (IID or non-IID).
      2) For each global round:
         a) Sample fraction of clients
         b) For each client: local training (few-shot episodes or simple steps)
         c) Aggregate updates into global_model
      3) Optionally test final model
    """
    device = next(global_model.parameters()).device
    print(f"[FED] Starting federated few-shot training on device={device}")

    # 1) Load dataset
    if args.dataset.lower() == 'omniglot':
        full_dataset = OmniglotDataset('background')  # or 'evaluation'
    elif args.dataset.lower() == 'miniimagenet':
        full_dataset = MiniImageNet('background')
    else:
        raise ValueError("Unsupported dataset in fed_trainer")

    # 2) Create user data splits
    #    If you want true IID or non-IID logic, you'd do so here.
    #    We'll do a simplistic IID example: each user gets an equal chunk 
    #    of the dataset. For a real approach, see your old sampling code.
    num_items = len(full_dataset) // args.num_users
    all_indices = np.arange(len(full_dataset))
    np.random.shuffle(all_indices)

    user_groups = {}
    start = 0
    for user_id in range(args.num_users):
        user_groups[user_id] = all_indices[start:start+num_items]
        start += num_items

    # 3) Federated Training Rounds
    global_weights = global_model.state_dict()

    for round_idx in range(args.epochs):
        print(f"\n--- [FED] Global Round {round_idx+1}/{args.epochs} ---")
        local_weights = []
        local_losses = []

        # Select fraction of users
        m = max(int(args.frac * args.num_users), 1)
        selected_users = np.random.choice(range(args.num_users), m, replace=False)
        print(f"[FED] Selected users: {selected_users}")

        # Each selected client does local update
        for user_id in selected_users:
            w, loss = local_update_few_shot(
                args,
                copy.deepcopy(global_model),
                full_dataset,
                user_groups[user_id],
                device
            )
            local_weights.append(w)
            local_losses.append(loss)

        # Aggregate (FedAVG)
        global_weights = fed_avg_aggregate(local_weights)
        global_model.load_state_dict(global_weights)

        avg_loss = sum(local_losses) / len(local_losses)
        print(f"[FED] Round {round_idx+1} average local loss: {avg_loss:.4f}")

    # 4) (Optional) Evaluate or save final global model
    print("[FED] Training complete. Evaluate or save your global_model here if you wish.")


def local_update_few_shot(args, global_model, dataset, user_idxs, device):
    """
    Simulates local training on a single client, using a few-shot approach 
    (n-way, k-shot episodes) if desired. We'll adapt 'matching_net_episode' 
    for local training.
    """
    # 1) Create a local optimizer
    #    Typically each client uses the same LR, etc.
    optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr, momentum=args.momentum)

    # 2) Build a small DataLoader subset for this user's data
    user_subset = Subset(dataset, user_idxs)
    # Here you could do episodic sampling with NShotTaskSampler 
    # if you want actual n-shot episodes for local training:
    sampler = NShotTaskSampler(
        dataset=user_subset,
        episodes_per_epoch=args.local_ep,  # e.g., "local_ep" episodes
        n=args.n_train,  # how many support per class
        k=args.k_train,  # how many classes
        q=args.q_train,  # how many query
        num_tasks=1
    )
    local_loader = DataLoader(user_subset, batch_sampler=sampler, num_workers=0)

    # 3) Perform local updates for 'args.local_ep' episodes
    local_losses = []
    global_model.train()

    for episode_idx, batch in enumerate(local_loader):
        # batch is [x, y] but shaped for n-shot scenario
        # We'll call matching_net_episode
        x, y = batch
        x, y = x.to(device), y.to(device)

        # matching_net_episode requires some extra kwargs
        # We'll define them below. We rely on your "distance" param, plus fce
        # Also note 'train=True' triggers the backward pass inside matching_net_episode
        loss, y_pred = matching_net_episode(
            model=global_model,
            optimiser=optimizer,
            loss_fn=nn.NLLLoss().to(device),
            x=x,
            y=y,
            n_shot=args.n_train,
            k_way=args.k_train,
            q_queries=args.q_train,
            distance=args.distance,
            fce=args.fce,
            train=True
        )
        local_losses.append(loss.item())

    # Return updated weights and average loss
    w = copy.deepcopy(global_model.state_dict())
    return w, sum(local_losses) / len(local_losses)


def fed_avg_aggregate(local_weights):
    """
    Basic FedAVG aggregator. Averages the weights from multiple clients.
    """
    w_avg = copy.deepcopy(local_weights[0])
    for key in w_avg.keys():
        for i in range(1, len(local_weights)):
            w_avg[key] += local_weights[i][key]
        w_avg[key] = torch.div(w_avg[key], len(local_weights))
    return w_avg
