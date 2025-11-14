# Copilot Instructions for AI Coding Agents

## Project Overview
This project focuses on fraud detection using tabular transaction data, leveraging both traditional data science workflows and graph-based deep learning (PyTorch Geometric). The main data sources are CSVs in the `dataset/` directory, with code in `code1.py` and exploratory/feature engineering work in `code.ipynb`.

## Key Components
- **Data**: All raw data is in `dataset/` (train/test transaction & identity CSVs). `creditcard.csv` is a separate dataset.
- **Graph Construction**: `code1.py` builds a heterogeneous graph using `card`, `device`, and `transaction` nodes, with edges representing usage relationships. Node features are placeholder embeddings or transaction amounts.
- **Model**: `FraudHGNN` in `code1.py` is a Heterogeneous Graph Neural Network using SAGEConv layers for fraud prediction.
- **Notebooks**: `code.ipynb` is used for EDA, preprocessing, and feature engineering, not for model training.

## Developer Workflows
- **Data Loading**: Always use relative paths (e.g., `dataset/train_transaction.csv`).
- **Graph Model**: Run `code1.py` to construct the graph and instantiate the model. Training logic may need to be added.
- **Notebook Workflow**: Use `code.ipynb` for data exploration and preprocessing. Avoid duplicating model code here.
- **Dependencies**: Requires `torch`, `torch-geometric`, and related packages. See install commands in comments at the top of `code1.py`.

## Project Conventions
- **Merging**: Always merge `train_transaction` and `train_identity` on `TransactionID` for training data.
- **Node Mapping**: Use `factorize` to map categorical columns to node IDs, handling NaNs as a special "unknown" node.
- **Edge Construction**: Edges are always directed from `card`/`device` to `transaction`.
- **Undirected Graphs**: Use `ToUndirected()` from PyG after edge construction.
- **Masks**: Use an 80/20 random split for train/test masks on transactions.

## Examples
- See `code1.py` for graph construction and model definition patterns.
- See `code.ipynb` for EDA and missing value analysis.

## Integration Points
- No external APIs; all data is local.
- Model and data pipelines are separate: keep EDA in notebooks, modeling in scripts.

## Tips for AI Agents
- When adding new node/edge types, follow the `to_id` and `to_edge` patterns in `code1.py`.
- For new features, add them as node attributes in the graph.
- If adding training logic, use PyTorch best practices and keep it in `code1.py` or a new script.

---
For questions or unclear conventions, review `code1.py` and `code.ipynb` for concrete examples, or ask for clarification.
