############################################
# experiments/matching_nets.py (from scratch)
############################################

import argparse
import torch

# Because we want a self-contained approach, we define or import our 
# local/federated logic in the same repo:
# 
#  - "few_shot.models" has the MatchingNetwork class
#  - "federated.fed_trainer" is a script we will create that handles 
#     server orchestration (FedAVG) and local training (few-shot episodes).
#  - "config" holds PATH, EPSILON, etc.
#
# If you haven't created "fed_trainer" yet, we'll just reference it below as if it exists.

from config import PATH, EPSILON
from few_shot.models import MatchingNetwork
# from federated.fed_trainer import run_federated_few_shot  # We'll demonstrate how you'd call it.

def build_parser():
    """
    Merges few-shot hyperparameters with 'federated' hyperparameters in a single parser.
    This is inspired by the old 'options.py' for federated plus the old arguments in 
    'experiments/matching_nets.py' for few-shot.

    You can expand or modify these as needed.
    """
    parser = argparse.ArgumentParser(description="Federated Few-Shot Training with Matching Networks")

    # --------------------------------------------------------------------
    # Federated arguments (inspired by your original fed scripts)
    # --------------------------------------------------------------------
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of global training rounds")
    parser.add_argument('--num_users', type=int, default=5,
                        help="Number of total clients (users)")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='Fraction of clients selected each round')
    parser.add_argument('--local_ep', type=int, default=3,
                        help="Number of local epochs per client update")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="Local batch size")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for local optimizers')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum')

    # Additional model/training options
    parser.add_argument('--model', type=str, default='matchingnet', 
                        help='Model type (here it is matchingnet by default)')
    parser.add_argument('--iid', type=int, default=0,
                        help='Whether data distribution is IID (1) or not (0)')

    # GPU
    parser.add_argument('--gpu', type=int, default=None, 
                        help='Which GPU to use. None for CPU')

    # --------------------------------------------------------------------
    # Few-Shot arguments
    # --------------------------------------------------------------------
    parser.add_argument('--fce', type=lambda x: x.lower()[0] == 't', default=True,
                        help='Use fully conditional embeddings (FCE)?')
    parser.add_argument('--distance', default='cosine',
                        help='Distance metric for the Matching Network: "cosine", "l2", or "dot"')
    parser.add_argument('--n_train', default=1, type=int,
                        help='Number of support examples per class (n) for training tasks')
    parser.add_argument('--n_test', default=1, type=int,
                        help='Number of support examples per class (n) for test tasks')
    parser.add_argument('--k_train', default=5, type=int,
                        help='Number of classes (k) in each training episode')
    parser.add_argument('--k_test', default=5, type=int,
                        help='Number of classes (k) in each test episode')
    parser.add_argument('--q_train', default=15, type=int,
                        help='Number of query samples per class for training tasks')
    parser.add_argument('--q_test', default=1, type=int,
                        help='Number of query samples per class for test tasks')
    parser.add_argument('--lstm_layers', default=1, type=int,
                        help='Number of LSTM layers if using FCE')
    parser.add_argument('--unrolling_steps', default=2, type=int,
                        help='Unrolling steps in the Attentional LSTM for FCE')

    # Dataset selection (omniglot, miniImageNet, etc.)
    parser.add_argument('--dataset', type=str, default='omniglot', 
                        help='Which dataset to use in the few-shot tasks')

    return parser

def main():
    """
    Main entry point for federated few-shot learning with Matching Networks.
    This script:
      1) Parses arguments
      2) Creates the Matching Network model as a "global model"
      3) Hands control to a future "fed_trainer" to orchestrate training
    """
    parser = build_parser()
    args = parser.parse_args()

    # Device selection. 
    device = 'cuda' if args.gpu is not None and torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    # Decide dataset-based hyperparams for the Matching Network
    if args.dataset.lower() == 'omniglot':
        num_input_channels = 1
        lstm_input_size = 64
    elif args.dataset.lower() == 'miniimagenet':
        num_input_channels = 3
        lstm_input_size = 1600
    else:
        raise ValueError("Unsupported dataset. Choose 'omniglot' or 'miniImageNet'")

    # Build the Matching Network
    global_model = MatchingNetwork(
        n=args.n_train,
        k=args.k_train,
        q=args.q_train,
        fce=args.fce,
        num_input_channels=num_input_channels,
        lstm_layers=args.lstm_layers,
        lstm_input_size=lstm_input_size,
        unrolling_steps=args.unrolling_steps,
        device=device
    ).to(device, dtype=torch.double)

    # Log a bit of info
    print("===============================================")
    print("[INFO] Created Matching Network with config:")
    print(f"       fce={args.fce}, distance={args.distance}")
    print(f"       n_train={args.n_train}, k_train={args.k_train}, q_train={args.q_train}")
    print(f"       lstm_layers={args.lstm_layers}, unrolling_steps={args.unrolling_steps}")
    print(f"       Dataset: {args.dataset}")
    print("===============================================")

    # THIS is where a typical FedAVG main script orchestrates data loading, client splits, etc.
    # Since we're building from scratch, let's just call a function that you'd create 
    # in your new "fed_trainer.py". For example:

    # from federated.fed_trainer import run_federated_few_shot
    # run_federated_few_shot(args, global_model)
    #
    # The run_federated_few_shot function would:
    #   1) Load the dataset(s) and produce local splits (iid or non-iid).
    #   2) For each round in 1..args.epochs:
    #       a) select fraction of users
    #       b) each user does local few-shot training 
    #          (possibly using "matching_net_episode" from few_shot.matching)
    #       c) server aggregates
    #   3) Evaluate global_model, etc.
    
    print("[INFO] Done initializing. In a real system, now we would call the federated trainer.")
    # E.g.:
    # run_federated_few_shot(args, global_model)

if __name__ == '__main__':
    main()
