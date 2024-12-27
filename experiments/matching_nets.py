###############################################
# Entry point for Federated Few-Shot training
###############################################
import argparse
import torch

from config import PATH, EPSILON
from few_shot.models import MatchingNetwork

# We import our new federated trainer function:
from federated.fed_trainer import run_federated_few_shot


def build_parser():
    """
    Merges federated arguments (e.g., num_users, local_ep, etc.)
    with few-shot arguments (n, k, q, etc.).
    """
    parser = argparse.ArgumentParser(description="Federated Few-Shot with Matching Networks")

    # -------------------- Federated arguments --------------------
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of global training rounds for federated learning")
    parser.add_argument('--num_users', type=int, default=5,
                        help="Number of total clients (users)")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='Fraction of clients selected in each round')
    parser.add_argument('--local_ep', type=int, default=3,
                        help="Number of local epochs per client update")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="Local batch size for each client")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for local client optimizers')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='Momentum for SGD (if using SGD)')

    parser.add_argument('--iid', type=int, default=1,
                        help='Whether to sample data IID (1) or non-IID (0)')

    # GPU config
    parser.add_argument('--gpu', type=int, default=None,
                        help='Which GPU to use. None for CPU')

    # -------------------- Few-Shot arguments ---------------------
    parser.add_argument('--fce', type=lambda x: x.lower()[0] == 't', default=True,
                        help='Use fully conditional embeddings (FCE)?')
    parser.add_argument('--distance', default='cosine',
                        help='Distance metric: "cosine", "l2", or "dot"')
    parser.add_argument('--n_train', type=int, default=1,
                        help='Number of support examples per class for training tasks (n)')
    parser.add_argument('--k_train', type=int, default=5,
                        help='Number of classes for training tasks (k)')
    parser.add_argument('--q_train', type=int, default=15,
                        help='Number of query examples per class for training tasks (q)')
    parser.add_argument('--n_test', type=int, default=1,
                        help='Number of support examples per class for test tasks (n)')
    parser.add_argument('--k_test', type=int, default=5,
                        help='Number of classes for test tasks (k)')
    parser.add_argument('--q_test', type=int, default=1,
                        help='Number of query examples per class for test tasks (q)')
    parser.add_argument('--lstm_layers', type=int, default=1,
                        help='Number of LSTM layers if using FCE')
    parser.add_argument('--unrolling_steps', type=int, default=2,
                        help='Attentional LSTM unrolling steps (if FCE)')
    parser.add_argument('--dataset', type=str, default='omniglot',
                        help='Which dataset to use: "omniglot" or "miniImageNet"')

    return parser


def main():
    # Parse arguments
    parser = build_parser()
    args = parser.parse_args()

    # Decide device
    device = 'cuda' if args.gpu is not None and torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    # Decide dataset-based parameters for the MatchingNetwork
    if args.dataset.lower() == 'omniglot':
        num_input_channels = 1
        lstm_input_size = 64
    elif args.dataset.lower() == 'miniimagenet':
        num_input_channels = 3
        lstm_input_size = 1600
    else:
        raise ValueError("Unsupported dataset. Use 'omniglot' or 'miniImageNet'.")

    # Build the global Matching Network model
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

    print("===========================================")
    print("[INFO] Created MatchingNetwork for Fed-Few-Shot")
    print(f"     fce={args.fce}, distance={args.distance}")
    print(f"     n={args.n_train}, k={args.k_train}, q={args.q_train}")
    print(f"     dataset={args.dataset}, device={device}")
    print("===========================================")

    # Now we pass control to the federated training function
    run_federated_few_shot(args, global_model)


if __name__ == '__main__':
    main()
