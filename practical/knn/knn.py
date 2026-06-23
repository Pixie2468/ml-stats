import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def knn(n_neighbours: int, metric: str = "euclidean"):
    model = KNeighborsClassifier(n_neighbors=n_neighbours, metric=metric)
    return model


if __name__ == "__main__":
    training_data = np.array([[1, 2], [2, 3], [3, 4], [6, 7]])
    labels = ["A", "A", "B", "B"]
    query_point = [5, 4]
    model = knn(n_neighbours=3)

    # Train
    model.fit(training_data, labels)

    # Predict
    prediction = model.predict(np.array([query_point]))
    print("Predicted label:", prediction[0])
