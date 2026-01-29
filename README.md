# Information-controlled Graph Convolutional Network for Multi-view Semi-supervised Classification

[![Paper](https://img.shields.io/badge/Paper-Neural_Networks-blue.svg)](https://doi.org/10.1016/j.neunet.2024.107102)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.neunet.2024.107102-green.svg)](https://doi.org/10.1016/j.neunet.2024.107102)

This is the official PyTorch implementation for our paper: **"Information-controlled Graph Convolutional Network for Multi-view Semi-supervised classification"**, published in *Neural Networks*, 2024.

---

## 🌟 Introduction
Multi-view learning using Graph Convolutional Networks (GCNs) often suffers from the **over-smoothing problem**, which limits their ability to capture long-range dependencies. While decoupling operations can mitigate this, it often leads to the loss of feature transformation modules and reduced model expressiveness.

**ICGCN** (Information-controlled Graph Convolutional Network) addresses these challenges by:
* **Feature Transformation with Orthogonality**: We maintain the node embedding paradigm during propagation by imposing orthogonality constraints on the feature transformation module.
* **Alleviating Over-smoothing**: By introducing a damping factor based on residual connections, we theoretically demonstrate that our model alleviates over-smoothing while retaining expressive feature transformations.
* **Numerical Stability**: We prove that our model stabilizes both forward inference and backward propagation, ensuring robust training for multi-view semi-supervised classification.

## 📂 Project Structure
* `model.py`: The core architecture of ICGCN.
* `DataLoader.py`: Scripts for data loading and multi-view graph construction.
* `train.py`: Main training loop.
* `main.py`: Entry point for experiments.
* `args.py`: Configuration and hyper-parameters.
* `utils.py`: Auxiliary functions for evaluation.

## 🛠️ Installation
1. Install the core dependencies:
   ```bash
   pip install -r requirements.txt
  
## 🚀 Usage
To train and evaluate the ICGCN model on the default dataset:
```bash
python main.py
