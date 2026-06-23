import numpy as np


def _get_majority_vote(neighbors):
    if not neighbors:
        raise ValueError("Neighbours list cannot be empty.")

    vote_count = {}
    min_distance_per_label = {}

    for neighbour in neighbors:
        label = neighbour["label"]
        distance = neighbour["distance"]

        vote_count[label] = vote_count.get(label, 0) + 1
        min_distance_per_label[label] = min(
            distance, min_distance_per_label.get(label, float("inf"))
        )

    # More efficient than sorting full list
    best_label = None
    best_vote = -1
    best_distance = float("inf")

    for label in vote_count:
        votes = vote_count[label]
        dist = min_distance_per_label[label]

        if (votes > best_vote) or (votes == best_vote and dist < best_distance):
            best_label = label
            best_vote = votes
            best_distance = dist

    return best_label


def knn_predict(training_data, labels, query_point, k):
    if len(training_data) == 0 or len(labels) == 0:
        raise ValueError("Training data and labels must not be empty.")

    if len(training_data) != len(labels):
        raise ValueError("Training data and labels must have the same length.")

    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")

    if k > len(training_data):
        raise ValueError("k cannot be larger than the number of training samples.")

    # Convert to NumPy for efficiency
    X = np.array(training_data)
    q = np.array(query_point)

    # Vectorized distance computation
    distances = np.linalg.norm(X - q, axis=1)

    # Get indices of k smallest distances
    nearest_idx = np.argsort(distances)[:k]

    neighbours = [{"distance": distances[i], "label": labels[i]} for i in nearest_idx]

    return _get_majority_vote(neighbours)


if __name__ == "__main__":
    training_data = [[1, 2], [2, 3], [3, 4], [6, 7]]
    labels = ["A", "A", "B", "B"]
    query_point = [5, 4]

    prediction = knn_predict(training_data, labels, query_point, k=3)
    print("Predicted label:", prediction)
