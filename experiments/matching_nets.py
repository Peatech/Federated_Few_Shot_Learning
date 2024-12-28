###############################################
# experiments/matching_nets.py
###############################################
import argparse
import torch

from config import PATH, EPSILON
from few_shot.models import MatchingNetwork
# Import the federated trainer function:
from federated.fed_trainer import run_federated_few_shot

def build_parser():
    parser = argparse.ArgumentParser(description="Federated Few-Shot with Matching Networks")

    # ---------- Federated Arguments ----------
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of global training rounds")
    parser.add_argument('--num_users', type=int, default=5,
                        help="Number of total clients (users)")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='Fraction of clients selected in each round')
    parser.add_argument('--local_ep', type=int, default=3,
                        help="Number of local epochs per client update")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="Local batch size per client")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for local optimizers')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum')
    parser.add_argument('--iid', type=int, default=1,
                        help='1 for IID, 0 for non-IID')

    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU ID (None for CPU)')

    # ---------- Few-Shot Arguments ----------
    parser.add_argument('--fce', type=lambda x: x.lower()[0] == 't', default=True,
                        help='Use Fully Conditional Embeddings (FCE)?')
    parser.add_argument('--distance', default='cosine',
                        help='Distance metric: "cosine", "l2", or "dot"')
    parser.add_argument('--n_train', default=1, type=int,
                        help='Number of support examples (n) per class for training tasks')
    parser.add_argument('--k_train', default=5, type=int,
                        help='Number of classes (k) for training tasks')
    parser.add_argument('--q_train', default=15, type=int,
                        help='Number of query examples per class for training tasks')
    parser.add_argument('--n_test', default=1, type=int,
                        help='Support examples for test tasks')
    parser.add_argument('--k_test', default=5, type=int,
                        help='Classes for test tasks')
    parser.add_argument('--q_test', default=1, type=int,
                        help='Query examples for test tasks')
    parser.add_argument('--lstm_layers', default=1, type=int,
                        help='Number of LSTM layers if using FCE')
    parser.add_argument('--unrolling_steps', default=2, type=int,
                        help='Attentional LSTM unrolling steps')
    parser.add_argument('--dataset', type=str, default='omniglot',
                        help='Which dataset: "omniglot" or "miniimagenet"')
    parser.add_argument('--eval_episodes', type=int, default=100,
                    help='Number of episodes to evaluate the model per epoch')

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    device = 'cuda' if args.gpu is not None and torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    # Decide dataset-based hyperparams for the MatchingNetwork
    if args.dataset.lower() == 'omniglot':
        num_input_channels = 1
        lstm_input_size = 64
    elif args.dataset.lower() == 'miniimagenet':
        num_input_channels = 3
        lstm_input_size = 1600
    else:
        raise ValueError("Unsupported dataset. Must be 'omniglot' or 'miniImageNet'.")

    # Create the global Matching Network model
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

    print("============================================")
    print("[INFO] Created MatchingNetwork for Fed-Few-Shot")
    print(f"     dataset={args.dataset}, fce={args.fce}, distance={args.distance}")
    print(f"     n_train={args.n_train}, k_train={args.k_train}, q_train={args.q_train}")
    print("============================================")

    # Hand over to the federated trainer
    run_federated_few_shot(args, global_model)

if __name__ == '__main__':
    main()
