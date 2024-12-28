# experiments/matching_nets.py

import argparse
import torch

from config import PATH, EPSILON
from few_shot.models import MatchingNetwork
from federated.fed_trainer import run_federated_few_shot


def main():
    parser = argparse.ArgumentParser()
    # -------------------- Federated arguments --------------------
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of global federated training rounds")
    parser.add_argument('--num_users', type=int, default=4,
                        help="Number of federated clients (max 4!)")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='Fraction of clients selected each round')
    parser.add_argument('--local_ep', type=int, default=1,
                        help="Number of local episodes (tasks) per round")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for local updates')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--iid', type=int, default=1,
                        help='(Legacy) Whether to do some form of split. We do class-based now.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU ID (None for CPU)')

    # -------------------- Few-Shot arguments ---------------------
    parser.add_argument('--fce', type=lambda x: x.lower()[0] == 't', default=True,
                        help='Use fully conditional embeddings (FCE)?')
    parser.add_argument('--distance', default='cosine',
                        choices=['l2','cosine','dot'],
                        help='Distance metric for matching: l2, cosine, or dot')
    parser.add_argument('--n_train', default=5, type=int,
                        help='Number of support samples (n) per class for training tasks')
    parser.add_argument('--k_train', default=5, type=int,
                        help='Number of classes (k) per training task')
    parser.add_argument('--q_train', default=5, type=int,
                        help='Number of query samples (q) per class for training tasks')
    parser.add_argument('--n_test', default=5, type=int,
                        help='Number of support samples for evaluation tasks')
    parser.add_argument('--k_test', default=5, type=int,
                        help='Number of classes for evaluation tasks')
    parser.add_argument('--q_test', default=5, type=int,
                        help='Number of query samples for evaluation tasks')
    parser.add_argument('--lstm_layers', type=int, default=1,
                        help='Number of LSTM layers if using FCE')
    parser.add_argument('--unrolling_steps', type=int, default=2,
                        help='Unrolling steps if using attentional LSTM')
    parser.add_argument('--dataset', type=str, default='omniglot',
                        choices=['omniglot','miniImageNet'],
                        help='Which dataset to use')

    args = parser.parse_args()

    # ----------------- Enforce max 4 users -----------------
    if args.num_users > 4:
        raise ValueError("For Omniglot and miniImageNet, the maximum number of users is 4!")

    # Decide device
    device = 'cuda' if (args.gpu is not None and torch.cuda.is_available()) else 'cpu'
    print(f"[INFO] Using device: {device}")

    # Decide dataset-based parameters
    if args.dataset.lower() == 'omniglot':
        num_input_channels = 1
        lstm_input_size = 64
    else:  # miniImageNet
        num_input_channels = 3
        lstm_input_size = 1600

    # Build the global Matching Network
    global_model = MatchingNetwork(
        n=args.n_train,
        k=args.k_train,
        q=args.q_train,
        fce=args.fce,
        num_input_channels=num_input_channels,
        lstm_layers=args.lstm_layers,
        lstm_input_size=lstm_input_size,
        unrolling_steps=args.unrolling_steps,
        device=torch.device(device)
    ).to(device, dtype=torch.double if args.dataset=='omniglot' else torch.float)

    print("============================================")
    print("[INFO] Created MatchingNetwork for Fed-Few-Shot")
    print(f"     dataset={args.dataset}, fce={args.fce}, distance={args.distance}")
    print(f"     n_train={args.n_train}, k_train={args.k_train}, q_train={args.q_train}")
    print("============================================")

    # Now we pass control to the federated trainer
    run_federated_few_shot(args, global_model)


if __name__ == '__main__':
    main()
