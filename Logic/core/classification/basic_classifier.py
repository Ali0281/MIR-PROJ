import numpy as np
from tqdm import tqdm

from ..word_embedding.fasttext_model import FastText


class BasicClassifier:
    def __init__(self):
        self.embedding = FastText()

    def fit(self, x, y):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def prediction_report(self, x, y):
        raise NotImplementedError()

    def get_percent_of_positive_reviews(self, sentences):
        """
        Get the percentage of positive reviews in the given sentences
        Parameters
        ----------
        sentences: list
            The list of sentences to get the percentage of positive reviews
        Returns
        -------
        float
            The percentage of positive reviews
        """
        prediction = self.predict([self.embedding.get_query_embedding(query) for query in sentences])
        return sum([1 if i == 1 else 0 for i in prediction]) / len(prediction) * 100
