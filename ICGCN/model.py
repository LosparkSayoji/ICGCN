import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class GraphConvolution(nn.Module):
    """
    Graph Convolutional Layer with Orthogonal Weights via Cayley Transform.
    """

    def __init__(self, in_features, out_features, gamma, bias=True):
        super(GraphConvolution, self).__init__()
        self.gamma = gamma
        self.weight = nn.Parameter(torch.empty(in_features, out_features))

        # Identity matrix for Cayley Transform, stored as a buffer (moves with model to GPU)
        self.register_buffer('identity', torch.eye(in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Use standard Xavier Uniform initialization for weights
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x, adj, x_0):
        # 1. Construct skew-symmetric matrix S = W - W^T
        s = self.weight - self.weight.t()

        # 2. Cayley Transform (optimized): W_ortho = (I - S)(I + S)^-1
        # torch.linalg.solve is numerically more stable than torch.inverse
        m_left = self.identity - s
        m_right = self.identity + s
        ortho_weight = torch.linalg.solve(m_right, m_left.t()).t()

        # 3. Message Passing: A * (X * W_ortho) * gamma + X_residual
        support = torch.mm(x, ortho_weight)
        output = torch.spmm(adj, support) * self.gamma + x_0

        if self.bias is not None:
            output = output + self.bias
        return output


class ICGCN(nn.Module):
    """
    Graph Convolutional Network for multi-view data.
    """

    def __init__(self, n_nodes, n_classes, input_dims, nhid, layer_num, dropout, gamma):
        super(ICGCN, self).__init__()

        # View transformation layers (Project raw features to hidden dimension)
        self.mlps = nn.ModuleList([
            nn.Linear(dim, nhid) for dim in input_dims
        ])

        # View-specific GCN stacks
        self.ortho_gcns = nn.ModuleList([
            nn.ModuleList([GraphConvolution(nhid, nhid, gamma=gamma)
                           for _ in range(layer_num)])
            for _ in input_dims
        ])

        # Normalization layers: One LayerNorm per view for training stability
        self.lns = nn.ModuleList([nn.LayerNorm(nhid) for _ in input_dims])

        # Learnable view importance weights (initialized uniformly)
        self.view_weight = nn.Parameter(torch.ones(len(input_dims), 1) / len(input_dims))

        # Final classification head
        self.classifier = nn.Linear(nhid, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_list, adj_list):
        view_outputs = []

        # Process each view through its respective MLP and GCN stack
        for i, (mlp, gcn_stack, ln, x, adj) in enumerate(zip(self.mlps, self.ortho_gcns, self.lns, x_list, adj_list)):
            # Initial feature mapping
            h = mlp(x)
            h_0 = h  # Initial hidden state for skip connections

            # Iterative graph convolution with ELU activation and LayerNorm
            for gcn in gcn_stack:
                h = gcn(h, adj, h_0)
                h = F.elu(h)  # note the ELU here
                h = ln(h)

            view_outputs.append(h)

        # Multi-view Fusion: weighted summation of view hidden states
        combined = torch.stack(view_outputs, dim=1)  # Shape: (Nodes, Views, Hidden)
        weights = F.softmax(self.view_weight, dim=0)
        merged = (weights * combined).sum(dim=1)  # Aggregate views

        # Output layer with Dropout for regularization
        merged = self.dropout(merged)
        logits = self.classifier(merged)

        return logits, weights
