# k-Nearest Neighbor (kNN) Methods

## Main Ideas

### 1. Introduction and Core Concepts
**Definition:** kNN is a supervised learning algorithm that makes predictions by finding the $k$ most similar (closest) training examples to a given query point. 1-Nearest Neighbor (NN) is a special case where $k=1$.

**Explanation:** Unlike many other algorithms that build a generalized global model during a training phase, kNN is a **lazy learning algorithm**. The "training" phase consists merely of storing the labeled training data. The actual processing and computation are deferred until the prediction (or "inference") step. kNN approximates the target function locally rather than globally.

**Categorization of kNN:**
* **Lazy (vs. Eager):** Defers computation until prediction. Does not build an explicit global model.
* **Instance-based / Memory-based:** Predictions rely on comparing query points directly to stored training instances (as opposed to eager instance-based learners like SVMs).
* **Nonparametric:** Makes no underlying assumptions about the functional form of the data distribution.
* **Discriminative (vs. Generative):** Models the posterior probabilities directly without explicitly modeling the underlying data generating process.

**Common Use Cases:**
* Computer vision, pattern classification, and biometrics (e.g., ear recognition via geometrical features).
* Recommender systems (via collaborative filtering).
* Unsupervised outlier detection.

### 2. Nearest Neighbor Decision Boundary
**Explanation:** In a 1NN model with a Euclidean distance metric, the decision boundary between any two training points is a straight line, representing the set of points equidistant to both. 

When considering the entire training set, the global decision boundary forms a set of connected, convex polyhedra known as a **Voronoi diagram** (or Voronoi tessellation). 
* Every point within a specific polyhedron is closest to the training example located inside it.
* The decision boundary of the overall classifier is the union of the pair-wise decision boundaries between instances of the same class.

### 3. Classification vs. Regression in kNN
kNN can be adapted for both discrete labels and continuous values.

#### Classification
* **Method:** Predicts the class label that is most frequently represented among the $k$ nearest neighbors.
* **Voting Mechanism:** This is a **plurality vote** (the mode), not necessarily a strict majority vote. A strict majority implies $>50\%$, which is guaranteed in binary classification but not in multi-class scenarios. For example, in a 3-class problem, a class might win the plurality vote with just $34\%$ representation.

#### Regression
* **Method:** Predicts a continuous target variable by averaging the values of the $k$ nearest neighbors.
* **Alternative:** The median of the $k$ nearest neighbors' targets can be used instead of the mean to provide robustness against outliers.

### 4. The Curse of Dimensionality
**Explanation:** kNN is highly susceptible to the curse of dimensionality. As the number of dimensions (features) increases, the hypervolume required to capture a fixed number of neighbors grows exponentially. 

**Example:**
* **1D Space:** To find 3 neighbors out of 100 uniformly distributed points, you need $3\%$ of the feature axis.
* **2D Space:** You need $\sqrt{0.03} \approx 17.3\%$ of the area.
* **10D Space:** You need $0.03^{1/10} \approx 70.4\%$ of the hypervolume.

**Conclusion:** In high-dimensional spaces, the "nearest" neighbors are actually quite far away from the query point across many dimensions, making the local approximation unreliable.

### 5. Computational Complexity (Big-O)
**Explanation:** The brute-force kNN algorithm has a prediction time complexity of $O(n \times m)$, where $n$ is the number of training examples and $m$ is the number of dimensions. Assuming $n \gg m$, the complexity is generally considered $O(n)$. Because this computation must be repeated for every single query point, brute-force kNN is very slow on large datasets.

### 6. Improving Computational Performance
To mitigate the $O(n)$ search time, several data structures and algorithmic tricks are used:

* **Priority Queues (Heaps):** Reduces the search tracking overhead from $O(n \times k)$ to $O(k \log n)$ by efficiently maintaining the top $k$ closest distances during the linear scan.
* **Bucketing:** Dividing the space into a grid of equally sized cells to restrict the search space.
* **KD-Trees:** A binary search tree that partitions the search space perpendicular to the feature axes. Average time complexity is $O(\log n)$, but it degrades to $O(n)$ in high dimensions.
* **Ball-Trees:** Partitions space using hyperspheres instead of hypercubes. While more computationally expensive to build than KD-trees, they are far more efficient in higher-dimensional spaces.
* **Dimensionality Reduction:** Techniques like Feature Selection (Sequential Forward Selection) or Feature Extraction (PCA) reduce $m$, speeding up distance calculations.
* **Pruning (Editing/Prototypes):** Removing training points that do not affect the decision boundary (e.g., points deeply embedded within a single class region or distant outliers) to save memory and distance computations.

### 7. Distance Measures
The definition of "nearest" relies entirely on the chosen distance metric.

* **Euclidean ($L^2$):** Standard straight-line distance.
* **Manhattan ($L^1$):** Sum of absolute differences. Less sensitive to extreme outliers than Euclidean.
* **Minkowski:** A generalized metric parameterized by $p$. ($p=1$ is Manhattan, $p=2$ is Euclidean).
* **Mahalanobis:** Considers the variance and covariance of features, but is computationally expensive to implement.
* **Hamming Distance:** Used for discrete/categorical binary features. Also known as the overlap metric, it counts the number of positions where two vectors differ.
* **Cosine Similarity:** Used frequently in text/document analysis (word counts). It measures the angle between vectors, normalizing for vector magnitude. (Note: Cosine similarity is not a proper distance metric as it violates the triangle inequality).

**Feature Scaling and Weighting:**
Because distance metrics like Euclidean treat all dimensions equally, the algorithm is highly sensitive to the scale of features. Features with larger numerical ranges will dominate the distance calculation. Scaling features is crucial. Feature weighting can be mathematically represented by a transformation matrix $W$ applied to the distance calculation.

### 8. Distance-weighted kNN
**Explanation:** In standard kNN, all $k$ neighbors have an equal vote. In Distance-Weighted kNN, neighbors that are physically closer to the query point are given a higher weight in the plurality vote or average.
* A common weighting scheme uses the inverse squared distance: $w^{[i]} = \frac{1}{d(x^{[i]}, x^{[q]})^2}$.
* This approach can technically turn kNN into a global method by considering all points $n$ (instead of just $k$), weighting distant points approaching zero.

### 9. Improving Predictive Performance
Predictive performance and generalization (balancing the bias-variance tradeoff) can be tuned via cross-validation over several hyperparameters:
1.  **Value of $k$:** * Small $k$: High complexity, high variance, low bias (prone to overfitting).
    * Large $k$: Low complexity, low variance, high bias (smoother decision boundaries, prone to underfitting).
    * *Tip:* Use odd values for $k$ in binary classification to prevent voting ties.
2.  **Distance measure choice.**
3.  **Feature scaling.**
4.  **Distance weighting schemes.**

### 10. Bayesian Perspective & Error Bounds
* **Error Bounds:** Cover and Hart proved that the error rate of 1NN is asymptotically no worse than twice the Bayes error (the theoretical minimum possible error).
* **Bayesian Density Estimation:** kNN can be viewed as estimating the posterior probability $P(y=i | x)$. It draws a sphere around $x$ capturing $k$ neighbors, yielding a density estimate that ultimately simplifies to the proportion of class $i$ samples within those $k$ neighbors.

### 11. Other Forms of Instance-Based Learning
* **Locally Weighted Regression:** Fits a simple regression model (like linear regression) only on the local neighborhood of training examples surrounding a query point. This is the basis for interpretability tools like LIME.
* **Kernel Methods (e.g., SVMs):** Uses a kernel (similarity function) to implicitly map data to a higher-dimensional space for linear separation. Note that standard SVMs are eager learners, unlike kNN.

---

## Important Terms
* **Lazy Learning** — An algorithmic approach where processing and model building are deferred until a prediction is explicitly requested.
* **Instance-based Learning** — Methods that predict labels by directly comparing a query instance against a memory of stored training instances.
* **Nonparametric Model** — A model that does not make strict assumptions about the underlying mathematical distribution or functional form of the data.
* **Voronoi Diagram** — A geometrical partitioning of a plane into regions based on distance to a specific set of points.
* **Curse of Dimensionality** — The phenomenon where the volume of the feature space increases so fast with adding dimensions that the available data becomes sparse, rendering distance metrics less meaningful.
* **Plurality Vote** — A voting system where the winning class simply has the most votes, regardless of whether it achieves an absolute majority (>50%).
* **Hamming Distance** — An overlap metric for discrete features that counts the number of differing positions between two arrays.

---

## Process / Steps / Workflow

### 1NN Prediction Algorithm
1.  Initialize `closest_point` as `None` and `closest_distance` as $\infty$.
2.  For every point $i$ from $1$ to $n$ in the training dataset:
    1. Calculate the distance $d$ between query point $x^{[q]}$ and training point $x^{[i]}$.
    2. If $d < \text{closest\_distance}$:
        1. Set `closest_distance` = $d$.
        2. Set `target_value_of_closest_point` = $y^{[i]}$.
3.  Output `target_value_of_closest_point`.

### kNN Classification Algorithm
1.  Calculate distances between the query point $x^{[q]}$ and all training points.
2.  Identify the set $\mathcal{D}_k$ containing the $k$ training points with the smallest distances.
3.  Count the frequency of each class label within $\mathcal{D}_k$.
4.  Output the class label with the highest frequency (Plurality Vote).

---

## Formula & Equation Reference

### Euclidean Distance ($L^2$)
$$
d(x^{[a]}, x^{[b]}) = \sqrt{\sum_{j=1}^{m} (x_j^{[a]} - x_j^{[b]})^2}
$$
* **Variables:** $x^{[a]}$ and $x^{[b]}$ are feature vectors. $m$ is the number of dimensions/features.
* **Interpretation:** The standard straight-line distance between two points in Euclidean space.

### Manhattan Distance ($L^1$)
$$
d(x^{[a]}, x^{[b]}) = \sum_{j=1}^{m} |x_j^{[a]} - x_j^{[b]}|
$$

### Minkowski Distance
$$
d(x^{[a]}, x^{[b]}) = \left[ \sum_{j=1}^{m} (|x_j^{[a]} - x_j^{[b]}|)^p \right]^{\frac{1}{p}}
$$
* **Variables:** $p$ is a parameter. $p=1$ yields Manhattan, $p=2$ yields Euclidean.

### Distance-weighted Euclidean Distance
$$
d_w(x^{[a]}, x^{[b]}) = \sqrt{\sum_{j=1}^{m} w_j (x_j^{[a]} - x_j^{[b]})^2} = \sqrt{(x^{[a]} - x^{[b]})^T W (x^{[a]} - x^{[b]})}
$$
* **Variables:** $w_j$ is the weight for feature $j$. $W$ is a diagonal matrix of feature weights.
* **Interpretation:** Allows scaling of features so that certain dimensions do not disproportionately dominate the distance calculation.

### kNN Classification Hypothesis (Plurality Vote)
$$
h(x^{[q]}) = \text{mode}(\{f(x^{[1]}), ..., f(x^{[k]})\})
$$
Alternatively expressed using the Kronecker delta function $\delta$:
$$
h(x^{[q]}) = \arg\max_{y \in \{1,...,t\}} \sum_{i=1}^{k} \delta(y, f(x^{[i]}))
$$
* **Variables:** $x^{[q]}$ is the query point. $t$ is the total number of classes. $\delta(a,b) = 1$ if $a=b$, else $0$.

### kNN Regression Hypothesis (Mean)
$$
h(x^{[q]}) = \frac{1}{k} \sum_{i=1}^{k} f(x^{[i]})
$$

### Inverse Squared Distance Weighting
$$
w^{[i]} = \frac{1}{d(x^{[i]}, x^{[q]})^2}
$$
* **Usage:** Used in distance-weighted kNN. Points closer to the query point exert a quadratically larger influence on the prediction.

### Cosine Similarity
$$
\cos(\theta) = \frac{a^T b}{||a|| \cdot ||b||}
$$
* **Usage:** Highly useful for comparing document word counts. It normalizes by vector magnitude, ensuring that document length does not skew the similarity score.

### kNN Posterior Probability (Bayesian Perspective)
$$
P(y=i | x) = \frac{p(x|y=i) \times p(y=i)}{p(x)} = \frac{k_i}{k}
$$
* **Variables:** $k_i$ is the number of points belonging to class $i$ among the $k$ nearest neighbors.
* **Interpretation:** The probability that point $x$ belongs to class $i$ is simply the fraction of its neighbors that belong to class $i$.

---

## Key Takeaways
1.  **Lazy but Expensive:** kNN requires zero training time but demands massive computational resources during inference ($O(n)$). Use KD-Trees or Ball-Trees to optimize production systems.
2.  **Distance is Everything:** kNN completely relies on the chosen distance metric. Always ensure features are properly scaled/normalized before running the algorithm to prevent axes with large numeric ranges from dictating the predictions.
3.  **Dimensionality is an Enemy:** The curse of dimensionality drastically reduces kNN's effectiveness. Utilize PCA or feature selection to maintain a dense, meaningful feature space.
4.  **Bias-Variance Tuning:** The hyperparameter $k$ dictates model flexibility. Small $k$ captures local noise (high variance/overfitting), while large $k$ smooths out the space (high bias/underfitting).
5.  **Robustness Tweaks:** Consider using the median instead of the mean for regression, utilizing odd values for $k$ in classification to avoid ties, and applying inverse distance weighting if the local neighborhood radius is very wide.

---

## Notes
* **KD-Tree vs Ball-Tree:** KD-Trees fail in high-dimensional spaces because the hypercubes overlap extensively with the search radius, forcing the algorithm to check many branches (reducing back to $O(n)$). Ball-trees solve this by using nested hyperspheres, but building them costs more upfront. In `scikit-learn`, setting `algorithm='auto'` generally manages this trade-off correctly behind the scenes.
* **Cosine Similarity as a Metric:** Note that while Cosine Similarity is an excellent measure for text retrieval, it is *not* a proper mathematical distance metric because it violates the triangle inequality ($d(a,c) \le d(a,b) + d(b,c)$).
* **Ties in Plurality Voting:** The document notes that an odd $k$ helps prevent ties in binary classification. However, for multi-class classification, ties can still occur even with odd $k$ values (e.g., $k=3$ with 3 distinct classes). Implementations generally break ties randomly or by favoring the class that is closest in distance.