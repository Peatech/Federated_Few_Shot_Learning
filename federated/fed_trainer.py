###############################################################
# federated/fed_trainer.py
# Implements FedAVG for a Few-Shot scenario with local episodes
###############################################################
import copy
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Subset

# Few-shot logic
from few_shot.matching import matching_net_episode
from few_shot.core import NShotTaskSampler, prepare_nshot_task
# Dataset classes
from few_shot.datasets import OmniglotDataset, MiniImageNet
# Data splitting for fed
from federated.fed_data_splitting import get_user_groups


def run_federated_few_shot(args, global_model):
    """
    Orchestrates a canonical FedAVG approach for few-shot tasks.
    Steps:
      1) Load train/eval datasets
      2) Split train data among clients
      3) For each round:
         a) Select fraction of users
         b) Local updates
         c) Aggregate
         d) Evaluate
    """

    # Determine device
    device = next(global_model.parameters()).device
    print(f"[FED] Starting Federated Few-Shot Training. device={device}")

    ###############################################################
    # 1) LOAD TRAIN/TEST DATASETS
    ###############################################################
    # Train dataset
    if args.dataset.lower() == 'omniglot':
        train_dataset = OmniglotDataset('background')
    elif args.dataset.lower() == 'miniimagenet':
        train_dataset = MiniImageNet('background')
    else:
        raise ValueError(f"Unsupported dataset {args.dataset}")

    # Eval dataset
    if args.dataset.lower() == 'omniglot':
        eval_dataset = OmniglotDataset('evaluation')
    else:
        eval_dataset = MiniImageNet('evaluation')

    # 2) Split data among clients
    user_groups = get_user_groups(
        dataset=train_dataset,
        num_users=args.num_users
    )

    ###############################################################
    # 3) BUILD EVAL DATALOADER
    ###############################################################
    eval_episodes = getattr(args, 'eval_episodes', 100)
    eval_sampler = NShotTaskSampler(
        dataset=eval_dataset,
        episodes_per_epoch=eval_episodes,
        n=args.n_test,
        k=args.k_test,
        q=args.q_test,
        num_tasks=1
    )
    eval_loader = DataLoader(eval_dataset, batch_sampler=eval_sampler, num_workers=0)
    eval_prepare_fn = prepare_nshot_task(args.n_test, args.k_test, args.q_test, device=device)

    ###############################################################
    # 4) FEDAVG MAIN LOOP
    ###############################################################
    global_weights = global_model.state_dict()

    for round_idx in range(args.epochs):
        print(f"\n--- [FED] Global Round {round_idx+1}/{args.epochs} ---")

        # a) SELECT CLIENTS
        selected_users = _select_clients(args.num_users, args.frac)
        local_weights, local_losses = [], []

        # b) LOCAL UPDATES
        for user_id in selected_users:
            updated_w, local_loss = _local_update_few_shot(
                args=args,
                local_model=copy.deepcopy(global_model),
                dataset=train_dataset,
                user_idxs=user_groups[user_id],
                device=device
            )
            local_weights.append(updated_w)
            local_losses.append(local_loss)

        # c) AGGREGATE (FEDAVG)
        global_weights = _fed_avg_aggregate(local_weights)
        global_model.load_state_dict(global_weights)

        # Summarize local losses
        avg_local_loss = sum(local_losses) / len(local_losses) if local_losses else 0.0
        print(f"[FED] Round {round_idx+1}: Avg Local Loss = {avg_local_loss:.4f}")

        # d) EVALUATE the aggregated global model on few-shot tasks
        eval_loss, eval_acc = _evaluate_global_model(
            model=global_model,
            dataloader=eval_loader,
            prepare_batch=eval_prepare_fn,
            device=device,
            n_shot=args.n_test,
            k_way=args.k_test,
            q_queries=args.q_test,
            distance=args.distance,
            fce=args.fce
        )
        print(f"[FED] Round {round_idx+1} EVAL: Loss={eval_loss:.4f}  Acc={eval_acc:.4f}")

    print("\n[FED] Federated Few-Shot Training complete.\n")


def _local_update_few_shot(args, local_model, dataset, user_idxs, device):
    """
    Performs local episodic training for a single client in a few-shot manner.
    """
    # 1) Create Subset for the user's data
    user_data = Subset(dataset, user_idxs)

    # Validation: ensure user_data.dataset has a 'df'
    if isinstance(user_data, Subset) and not hasattr(user_data.dataset, 'df'):
        raise ValueError("Dataset wrapped in Subset must have a 'df' attribute for n-shot sampling.")

    # 2) Build local DataLoader with NShotTaskSampler
    sampler = NShotTaskSampler(
        dataset=user_data,
        episodes_per_epoch=args.local_ep,  # how many episodes this client runs
        n=args.n_train,
        k=args.k_train,
        q=args.q_train,
        num_tasks=1
    )
    loader = DataLoader(user_data, batch_sampler=sampler, num_workers=0)

    # 3) Prepare function for device casting
    prepare_fn = prepare_nshot_task(args.n_train, args.k_train, args.q_train, device=device)

    # 4) Local training
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=args.lr, momentum=args.momentum)
    loss_fn = nn.NLLLoss().to(device)
    local_losses = []

    for episode_batch in loader:
        x, y = prepare_fn(episode_batch)

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
            train=True
        )
        local_losses.append(loss.item())

    updated_weights = copy.deepcopy(local_model.state_dict())
    avg_loss = sum(local_losses) / len(local_losses) if local_losses else 0.0
    return updated_weights, avg_loss


def _evaluate_global_model(model,
                           dataloader,
                           prepare_batch,
                           device,
                           n_shot,
                           k_way,
                           q_queries,
                           distance,
                           fce):
    """
    Evaluates the global model on few-shot tasks.
    """
    model.eval()
    total_loss, total_correct, total_seen = 0.0, 0, 0
    loss_fn = nn.NLLLoss().to(device)

    dummy_optim = torch.optim.SGD(model.parameters(), lr=0.001)

    with torch.no_grad():
        for batch in dataloader:
            x, y = prepare_batch(batch)
            loss, y_pred = matching_net_episode(
                model=model,
                optimiser=dummy_optim,
                loss_fn=loss_fn,
                x=x,
                y=y,
                n_shot=n_shot,
                k_way=k_way,
                q_queries=q_queries,
                distance=distance,
                fce=fce,
                train=False
            )
            batch_size = y_pred.shape[0]
            total_loss += loss.item() * batch_size

            preds = y_pred.argmax(dim=1)
            correct = (preds == y).sum().item()
            total_correct += correct
            total_seen += batch_size

    avg_loss = total_loss / total_seen if total_seen else 0.0
    avg_acc = total_correct / total_seen if total_seen else 0.0
    return avg_loss, avg_acc


def _select_clients(num_users, frac):
    """
    Randomly select a fraction of users each round.
    """
    m = max(int(frac * num_users), 1)
    selected = np.random.choice(range(num_users), m, replace=False)
    print(f"[FED] Selected users: {selected}")
    return selected


def _fed_avg_aggregate(local_weights):
    """
    Basic FedAVG aggregator: average local weights.
    """
    if not local_weights:
        return None
    w_avg = copy.deepcopy(local_weights[0])
    for key in w_avg.keys():
        for i in range(1, len(local_weights)):
            w_avg[key] += local_weights[i][key]
        w_avg[key] = torch.div(w_avg[key], len(local_weights))
    return w_avg
