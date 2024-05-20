import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader


class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors):
        super().__init__()
        self.k = n_neighbors

    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values
        use the Euclidean distance to find the k nearest neighbors
        Warning: Maybe you need to reduce the size of X to avoid memory errors

        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        self
            Returns self as a classifier
        """
        self.x = x
        self.y = y
        return self

    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        predictions = []
        for i in tqdm(x):
            distances = np.linalg.norm(self.x - i, axis=1)
            k_nearest_labels = self.y[np.argsort(distances)[:min(self.k, len(distances))]]
            prediction = np.argmax(np.bincount(k_nearest_labels))
            predictions.append(prediction)
        return np.array(predictions)

    def prediction_report(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        return classification_report(y, self.predict(x))


# F1 Accuracy : 70%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    loader = ReviewLoader()
    loader.load_data()
    loader.get_embeddings()
    x_train, x_test, y_train, y_test = loader.split_data()

    """label_enc = LabelEncoder()
    y = label_enc.fit_transform(y)
    y_test = label_enc.transform(y_test)"""

    knn_classifier = KnnClassifier(n_neighbors=8)  # Initialize with desired k value
    knn_classifier.fit(x_train, y_train)
    print(knn_classifier.prediction_report(x_test, y_test))
