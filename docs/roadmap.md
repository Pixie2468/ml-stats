# ML-STATS Roadmap

## Mathematical Foundations

### Regression

* [ ] **Linear Regression**
  * [ ] Ordinary Least Squares (OLS)
  * [ ] Gradient Descent (Batch & SGD)
* [ ] **Regularized Regression**
  * [ ] Ridge Regression (L2)
  * [ ] Lasso Regression (L1 with Coordinate Descent)
  * [ ] Elastic Net

### Classification

* [ ] **Logistic Regression**
  * [ ] Binary Logistic
  * [ ] Multinomial Logistic (Softmax)
* [ ] **Naive Bayes**
  * [ ] Gaussian NB
  * [ ] Multinomial NB
  * [ ] Bernoulli NB
* [] **KNN**
  * [] Brute-force KNN
  * [] KD-Tree implementation
* [ ] **SVM**
  * [ ] Linear SVM (Hinge Loss)
  * [ ] Kernel SVM (SMO algorithm, RBF/Poly kernels)

---

## Tree-Based Models

### Decision Trees

* [] ID3 (Entropy & Information Gain)
* [] C4.5 (Gain Ratio)
* [] CART (Gini Impurity & MSE)

### Bagging Ensembles

* [ ] Random Forest
* [ ] Extra Trees

### Boosting Ensembles

* [ ] AdaBoost (SAMME)
* [ ] Gradient Boosting Machine (GBM)
* [ ] XGBoost (Simplified 2nd-order approximation)

---

## Unsupervised Learning (Clustering & Dim Reduction)

### Clustering

* [ ] **K-Means**
  * [ ] Lloyd's Algorithm
  * [ ] K-Means++ Initialization
* [ ] **Hierarchical Clustering**
  * [ ] Agglomerative (Single, Complete, Average, Ward's)
* [ ] **Density & Mixture**
  * [ ] DBSCAN
  * [ ] Gaussian Mixture Models (EM Algorithm)

### Dimensionality Reduction

* [ ] PCA (Eigen-decomposition / SVD)
* [ ] LDA (Linear Discriminant Analysis)
* [ ] t-SNE
* [ ] UMAP

---

## Deep Learning

### Core Networks

* [ ] Perceptron
* [ ] Multi-Layer Perceptron (MLP with Backprop)
* [ ] CNN (Forward/Backward 2D Convolutions, Pooling)

### Sequence Models

* [ ] RNN (BPTT)
* [ ] LSTM (Forget, Input, Output Gates)
* [ ] GRU

### Attention Mechanisms

* [ ] Scaled Dot-Product Attention
* [ ] Multi-Head Attention
* [ ] Transformer Architecture

---

## NLP

### Basics

* [ ] N-Grams
* [ ] TF-IDF
* [ ] BPE (Byte-Pair Encoding)

### Embeddings

* [ ] Word2Vec (Skip-gram with Negative Sampling)
* [ ] GloVe

---

## For Every Algorithm

### Scratch Version

* [ ] Mathematical derivation understood
* [ ] Implemented from scratch
* [ ] Fully vectorized using NumPy
* [ ] Unit tested
* [ ] Benchmarked

### Practical Version

* [ ] Implemented using ecosystem tools
* [ ] Hyperparameter tuning
* [ ] Evaluation completed

### Documentation

* [ ] Learning notes written
* [ ] Common pitfalls documented
* [ ] References added
