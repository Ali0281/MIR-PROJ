import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader


class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=0.6):
        super().__init__()
        self.cv = count_vectorizer
        self.num_classes = None
        self.classes = None
        self.number_of_features = None
        self.number_of_samples = None
        self.prior = None
        self.feature_probabilities = None
        self.log_probs = None
        self.alpha = alpha

        if self.cv is None: self.cv = CountVectorizer()
        self.log_prior = None
        self.batch_size = 1000

    def fit(self, x, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

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
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        self.number_of_samples, self.number_of_features = x.shape

        self.prior = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            self.prior[i] = np.sum(y == self.classes[i]) / self.number_of_samples
        self.log_prior = np.log(self.prior)

        self.feature_probabilities = np.zeros((self.num_classes, self.number_of_features))
        for i in range(0, self.number_of_samples, self.batch_size):
            x_ = x[i:i + self.batch_size]
            y_ = y[i:i + self.batch_size]

            for j in range(self.num_classes):
                indice = np.where(y_ == self.classes[j])[0]
                features = x_[indice]
                self.feature_probabilities[j] += (features.sum(axis=0) + self.alpha)

        self.feature_probabilities = self.feature_probabilities / (
                self.alpha * self.number_of_features + np.sum(self.feature_probabilities, axis=1, keepdims=True))
        self.log_probs = np.log(self.feature_probabilities)

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
        # TODO: note : work with logs as they are easier
        return self.classes[np.argmax(self.log_prior + (x @ self.log_probs.T), axis=1)]

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

    def get_percent_of_positive_reviews(self, sentences):
        """
        You have to override this method because we are using a different embedding method in this class.
        """
        prediction = self.predict(self.cv.transform(sentences).toarray())
        print(prediction)
        return np.sum(prediction == "positive" ) / len(prediction) * 100

    # F1 Accuracy : 85%


if __name__ == '__main__':
    """
    First, find the embeddings of the revies using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn's CountVectorizer to find the embeddings.
    """
    loader = ReviewLoader()
    loader.load_data()
    x_train, x_test, y_train, y_test = loader.split_data(enc=False)

    # DF = pd.read_csv("C:/Users/Ali/PycharmProjects/MIR-PROJ/Logic/core/classification/labeled-comments.csv")
    # x_train, x_test, y_train, y_test = train_test_split(DF["review"].array, DF["sentiment"].array, test_size=0.2, random_state=42)

    count_vectorizer = CountVectorizer()
    x_train_vec = count_vectorizer.fit_transform(x_train)
    x_test_vec = count_vectorizer.transform(x_test)

    nb_classifier = NaiveBayes(count_vectorizer)
    nb_classifier.fit(x_train_vec.toarray(), y_train)

    print(nb_classifier.prediction_report(x_test_vec.toarray(), y_test))
    print(nb_classifier.get_percent_of_positive_reviews(
        ["happy happy good very chill and interesting", "excellent actors and staff and entertaining", "very bad acting and horrible taste of art"]))
    print(nb_classifier.get_percent_of_positive_reviews(
        ["bro that movies didnt make any sense", "didnt like it i got bored", "i would recommend it"]))
