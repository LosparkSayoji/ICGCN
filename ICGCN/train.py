from __future__ import division
from __future__ import print_function

import os
import time
import torch
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import normalize as sk_normalize, StandardScaler

# Local imports
from DataLoader import load_mat_data, generate_splits
from utils import calculate_accuracy, calculate_f1
from model import ICGCN


def train(args, device):
    """
    Train the Multi-view GCN model.
    """
    # TODO: Replace with your local dataset path
    data_path = 'your/path/to/dataset/'
    adj, features, labels, nfeats, num_view, num_class = load_mat_data(
        args.dataset, args.k, data_path
    )

    features = [StandardScaler().fit_transform(matrix) for matrix in features]

    # Split data into train/test
    idx_train, idx_test = generate_splits(labels, args)

    # Initialize model
    model = ICGCN(
        n_nodes=adj[0].shape[0],
        n_classes=num_class,
        input_dims=nfeats,
        nhid=args.nhid,
        layer_num=args.layer_num,
        dropout=args.dropout,
        gamma=args.gamma
    )

    # Log parameter count
    total_para = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in network: {total_para / 1e6:.4f}M")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Data Preparation ---
    for i in range(num_view):
        # 1. Apply sklearn normalization (Row-wise)
        features[i] = sk_normalize(features[i])
        # 2. Convert to tensor and move to device
        features[i] = torch.FloatTensor(features[i]).to(device)
        # 3. Convert sparse adj to dense and move to device
        adj[i] = adj[i].to_dense().float().to(device)

    if args.cuda:
        model = model.to(device)
        labels = labels.to(device)
        idx_train = idx_train.to(device)
        idx_test = idx_test.to(device)

    start_time = time.time()

    # --- Training Loop ---
    with tqdm(total=args.epoch) as pbar:
        pbar.set_description('Training Progress')

        for epoch in range(args.epoch):
            epoch_start = time.time()
            model.train()
            optimizer.zero_grad()

            # Forward pass
            output, W = model(features, adj)
            output = F.log_softmax(output, dim=1)

            # Loss and metrics for training set
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            acc_train = calculate_accuracy(output[idx_train], labels[idx_train])

            # Backward pass
            loss_train.backward()
            optimizer.step()

            # Evaluation for test metrics within the loop
            if not args.fastmode:
                model.eval()
                with torch.no_grad():
                    output, W = model(features, adj)
                    output = F.log_softmax(output, dim=1)

            loss_test = F.nll_loss(output[idx_test], labels[idx_test])
            acc_test = calculate_accuracy(output[idx_test], labels[idx_test])
            f1_test_val = calculate_f1(output[idx_test], labels[idx_test])

            # Update Progress Bar
            out_msg = (f"Epoch: {epoch + 1:04d} | "
                       f"Loss_Tr: {loss_train.item():.4f} | "
                       f"Acc_Tr: {acc_train.item():.4f} | "
                       f"Loss_Te: {loss_test.item():.4f} | "
                       f"Acc_Te: {acc_test.item():.4f} | "
                       f"F1_Te: {f1_test_val:.4f}")

            pbar.set_postfix_str(out_msg)
            pbar.update(1)

    print(f'Total Training Time: {time.time() - start_time:.4f}s')
    return model, features, labels, adj, idx_test, output


def test(args, model, features, labels, adj, idx_test):
    """
    Final evaluation on the test set.
    """
    model.eval()
    with torch.no_grad():
        output, W = model(features, adj)
        output = F.log_softmax(output, dim=1)  # Ensure consistency with NLLLoss

    # Compute final metrics
    acc_test = calculate_accuracy(output[idx_test], labels[idx_test])
    f1_score = calculate_f1(output[idx_test], labels[idx_test])

    print(f"\n--- Final Results for {args.dataset} (Gamma: {args.gamma}) ---")
    print(f"Test Accuracy: {100 * acc_test.item():.2f}%")
    print(f"Test F1 Score: {100 * f1_score:.2f}%")

    return 100 * acc_test.item(), 100 * f1_score