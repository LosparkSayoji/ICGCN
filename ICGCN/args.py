import argparse


def parameter_parser():
    """
    Parse command line arguments for the model.
    """
    parser = argparse.ArgumentParser(description="Run Multi-view GNN Training")

    # --- Dataset and Experiment Settings ---
    parser.add_argument('--dataset', type=str, default="100leaves",
                        help='Name of the dataset to use.')
    parser.add_argument('--rep_num', type=int, default=5,
                        help='Number of independent repetitions.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility.')
    parser.add_argument('--res_path', type=str, default="./results/",
                        help='Path to save experimental results.')

    # --- Model Architecture ---
    parser.add_argument("--layer_num", type=int, default=2,
                        help="Number of graph convolutional layers.")
    parser.add_argument('--nhid', type=int, default=128,
                        help='Dimension of the hidden layers.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_norm', type=int, default=1, choices=[0, 1],
                        help='Whether to use batch normalization (1 for True, 0 for False).')

    # --- Training Hyperparameters ---
    parser.add_argument('--epoch', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='hyper-parameter in ICGCN.')

    # --- Graph Construction & Data Splitting ---
    parser.add_argument('--k', type=int, default=10,
                        help='Number of neighbors for kneighbors_graph construction.')
    parser.add_argument('--data_split_mode', type=str, default='Ratio',
                        choices=['Ratio', 'Num'], help='Data splitting mode.')
    parser.add_argument('--train_ratio', type=float, default=0.1,
                        help='Ratio of samples used for training (used if mode is Ratio).')

    # --- Hardware and Execution ---
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--device', type=str, default="0",
                        help='GPU device ID.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')

    args = parser.parse_args()

    return args
