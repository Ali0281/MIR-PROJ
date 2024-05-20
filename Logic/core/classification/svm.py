import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader


class SVMClassifier(BasicClassifier):
    def __init__(self):
        super().__init__()
        self.model = SVC()

    def fit(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc
        """
        self.model.fit(x, y)

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
        return self.model.predict(x)

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


# F1 accuracy : 78%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    loader = ReviewLoader()
    loader.load_data()
    loader.get_embeddings()
    x_train, x_test, y_train, y_test = loader.split_data()

    '''
    # we do it in loader so we dont need it here
    label_enc = LabelEncoder()
    y_train = label_enc.fit_transform(y_train)
    y_test = label_enc.transform(y_test)
    '''

    svm = SVMClassifier()
    svm.fit(x_train, y_train)
    print(svm.prediction_report(x_test, y_test))
