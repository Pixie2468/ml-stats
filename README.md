# ML-STATS

A collection of machine learning algorithms implemented both from scratch and using industry-standard libraries.

The goal of this repository is to deeply understand how machine learning algorithms work internally while also learning how they are applied in real-world projects.

## Repository Structure

```text
├── docs/             # Learning notes and explanations
├── notebooks/        # Jupyter notebooks for prototyping and visualization
├── src/              # Source code (First-principles implementations & utils)
│   ├── algorithms/   # From-scratch implementations
│   └── utils/        # Shared utility functions
├── practical/        # Implementations using established ML libraries
├── .gitignore
├── .python-version
├── main.py           # Application entry point
├── pyproject.toml    # Project metadata and dependencies
├── README.md
└── uv.lock           # Lockfile for deterministic builds
```

## Progress Snapshot

*(For the comprehensive checklist, see `roadmap.md`)*

### Mathematical Foundations

* [ ] Linear Regression (OLS & GD)
* [ ] Regularized Regression (Ridge, Lasso)
* [ ] Logistic Regression (Binary & Multinomial)
* [ ] Naive Bayes
* [] KNN (Brute-force & KD-Tree)
* [ ] SVM (Linear & Kernel)

### Tree-Based Models

* [] Decision Trees (ID3, C4.5, CART)
* [ ] Bagging Ensembles (Random Forest, Extra Trees)
* [ ] Boosting Ensembles (AdaBoost, GBM, XGBoost)

### Unsupervised Learning

* [ ] K-Means & K-Means++
* [ ] Hierarchical Clustering
* [ ] DBSCAN
* [ ] GMM
* [ ] PCA & t-SNE

### Deep Learning & NLP

* [ ] Perceptron & MLP
* [ ] CNN
* [ ] RNN, LSTM, GRU
* [ ] Attention & Transformers
* [ ] TF-IDF & Word2Vec

## Philosophy

For every algorithm:

* Implement from scratch
* Implement using modern libraries
* Compare results
* Document lessons learned

## Running

```bash
uv sync
uv run main.py
```
