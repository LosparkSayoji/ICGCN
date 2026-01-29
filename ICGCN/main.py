from __future__ import division
from __future__ import print_function

import os
import random
import datetime
import argparse
import numpy as np
import torch
import scipy.io as sio
from warnings import simplefilter

# Local imports
from train import train, test
from args import parameter_parser

# Ignore FutureWarnings for cleaner output
simplefilter(action='ignore', category=FutureWarning)

# --- Initialization ---
args = parameter_parser()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Set device based on args
device = torch.device('cpu' if args.device == 'cpu' else f'cuda:{args.device}')

# --- Global Seeding ---
# Strictly maintain reproducibility across numpy, torch, and random
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.enabled = True

# --- Main Experiment Loop ---
# Datasets list: '100leaves', 'animals', 'Caltech101-20', 'flower17', 'MNIST10k', 'scene15', 'HW', 'Youtube'
dataset_list = ['100leaves']

for data_name in dataset_list:
    args.dataset = data_name

    acc_records = []
    f1_records = []

    print(f"Configuration: {args}")

    for i in range(args.rep_num):
        print(f"Starting repetition: {i + 1}/{args.rep_num}")

        # Train model
        model, features, labels, adj, idx_test, output = train(args, device)

        # Test model
        acc_test, f1_score_test = test(args, model, features, labels, adj, idx_test)

        acc_records.append(acc_test)
        f1_records.append(f1_score_test)

        # Explicit memory management
        del model
        if args.cuda:
            torch.cuda.empty_cache()

    # --- Summary Statistics ---
    print("\nOptimization Finished!")

    acc_array = np.array(acc_records)
    f1_array = np.array(f1_records)

    print(f"Dataset: {args.dataset}")
    print(f"Accuracy: mean = {acc_array.mean():.4f}, std = {acc_array.std():.4f}")
    print(f"F1 Score: mean = {f1_array.mean():.4f}, std = {f1_array.std():.4f}")

    # --- Result Logging ---
    if not os.path.exists(args.res_path):
        os.makedirs(args.res_path)

    log_file_path = os.path.join(args.res_path, f"{args.dataset}.txt")

    with open(log_file_path, 'a', encoding='utf-8') as f:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')

        # Log Hyperparameters
        f.write(f"{timestamp}\n")
        f.write(f"dataset:{args.dataset} | layer_num:{args.layer_num} | rep_num:{args.rep_num} | "
                f"Ratio:{args.train_ratio} | knn:{args.k}\n")
        f.write(f"dropout:{args.dropout} | epochs:{args.epoch} | lr:{args.lr} | "
                f"wd:{args.weight_decay} | hidden:{args.nhid} | gamma:{args.gamma}\n")

        # Log Metrics
        f.write(f"ACC_mean:{acc_array.mean():.4f} | ACC_std:{acc_array.std():.4f} | ACC_max:{acc_array.max():.4f}\n")
        f.write(f"F1_mean:{f1_array.mean():.4f} | F1_std:{f1_array.std():.4f} | F1_max:{f1_array.max():.4f}\n")
        f.write("-" * 76 + "\n")
