# Key Concept

kNN is an algorithm for supervised learning that simply stores
the labeled training examples,
$\left(\{x^{[i]}, y^{[i]}\} \in \mathcal{D} \quad (|\mathcal{D}| = n)\right)$
during the training phase. For this reason, kNN is also called a lazy learning algorithm.
What it means to be a lazy learning algorithm is that the processing of the training examples
is postponed until making predictions again, the training consists literally of just storing
the training data.
Then, to make a prediction (class label or continuous target), a trained kNN model finds
the k nearest neighbors of a query point and computes the class label (classification) or
continuous target (regression) based on the k nearest (most “similar”) points. However, the overall idea is that instead of approximating the target function f(x) = y globally, during each prediction, kNN approxi-
mates the target function locally. In practice, it is easier to learn to approximate a function locally than globally.
