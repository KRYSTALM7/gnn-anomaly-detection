# GNN Anomaly Detection in Sensor Networks

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.x-blueviolet?style=flat-square)](https://pyg.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

A **Graph Neural Network (GCN) autoencoder** for unsupervised anomaly detection in multivariate sensor data. Sensors are modelled as graph nodes; edges encode correlation-based relationships between them. The model learns to reconstruct normal behaviour — timesteps with high reconstruction error are flagged as anomalies.

---

## Overview

Traditional anomaly detection treats sensors independently. This approach instead models **inter-sensor relationships** as a graph, enabling the GCN to exploit correlation structure when learning what "normal" looks like.

```
Sensor Data (T × N)
       │
       ▼
Correlation Matrix
       │
  Threshold (0.4)
       │
       ▼
  Graph G = (V, E)       ← N sensor nodes, E correlation edges
       │
       ▼
┌──────────────────┐
│   GCN Encoder    │     1 → 16 → 8  (latent embedding)
│   GCN Decoder    │     8 → 16 → 1  (reconstruction)
└──────────────────┘
       │
  Reconstruction
     Error (MSE)
       │
  > 95th pct threshold?
       │
     Anomaly
```

---

## Results (Synthetic Benchmark)

| Metric    | Value  |
|-----------|--------|
| Precision | ~0.78  |
| Recall    | ~0.82  |
| F1 Score  | ~0.80  |
| ROC-AUC   | ~0.91  |

> Results will vary with random seed and data generation. Replace with real dataset metrics once applied to SWAT / MSL / WADI.

---

## Project Structure

```
gnn-anomaly-detection/
├── notebooks/
│   └── gnn_anomaly_detection.ipynb   # Full pipeline notebook
├── assets/
│   ├── results_dashboard.png         # Evaluation plots
│   └── graph_topology.png            # Sensor graph visualisation
├── outputs/
│   └── models/
│       └── gcn_autoencoder.pt        # Saved model + threshold
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/KRYSTALM7/gnn-anomaly-detection.git
cd gnn-anomaly-detection
pip install -r requirements.txt
```

### 2. Run the notebook

```bash
jupyter notebook notebooks/gnn_anomaly_detection.ipynb
```

The notebook is self-contained — it generates synthetic data, trains the model, and produces all evaluation plots.

---

## Requirements

```
torch>=2.0.0
torch_geometric>=2.3.0
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
networkx>=3.1
```

Install with:
```bash
pip install -r requirements.txt
```

For PyTorch Geometric, follow the [official install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) to match your CUDA version.

---

## How It Works

**Graph Construction**  
For each dataset, we compute the Pearson correlation matrix across sensor channels. Pairs of sensors with `|corr| > 0.4` are connected by an edge. This graph is fixed throughout training and inference.

**GCN Autoencoder**  
At each timestep, sensor readings form the node feature matrix `X` (shape: `N × 1`). The encoder maps this to a low-dimensional latent embedding via two GCN layers; the decoder reconstructs the original readings. Training minimises MSE on **normal data only**.

**Anomaly Scoring**  
At inference, each timestep receives a score equal to the mean squared reconstruction error across all nodes. Scores exceeding the 95th percentile of training scores are classified as anomalies.

---

## Extensions & Next Steps

- Swap `GCNConv` for `GATConv` (Graph Attention) for learned edge weighting
- Add temporal edges between consecutive snapshots for spatial-temporal GNNs
- Apply to real datasets: [SWAT](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/), [MSL/SMAP](https://github.com/khundman/telemanom)
- Tune threshold using a labelled validation split instead of a fixed percentile
- Experiment with dynamic graph construction (recalculate edges on rolling windows)

---

## License

MIT — see [LICENSE](LICENSE).
