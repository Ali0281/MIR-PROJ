import os
import string

import fasttext
import re

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
import numpy as np

from Logic.core.word_embedding.fasttext_data_loader import FastTextDataLoader


def preprocess_text(text, minimum_length=1, stopword_removal=True, stopwords_domain=[], lower_case=True,
                    punctuation_removal=True):
    """
    preprocess text by removing stopwords, punctuations, and converting to lowercase, and also filter based on a min length
    for stopwords use nltk.corpus.stopwords.words('english')
    for punctuations use string.punctuation

    Parameters
    ----------
    text: str
        text to be preprocessed
    minimum_length: int
        minimum length of the token
    stopword_removal: bool
        whether to remove stopwords
    stopwords_domain: list
        list of stopwords to be removed base on domain
    lower_case: bool
        whether to convert to lowercase
    punctuation_removal: bool
        whether to remove punctuations
    """
    if lower_case: text = text.lower()
    tokenized = [i for i in word_tokenize(text) if len(i) >= minimum_length]
    if punctuation_removal: tokenized = [i for i in tokenized if i not in string.punctuation]
    if stopword_removal: tokenized = [i for i in tokenized if
                                      i not in set(stopwords.words('english')).union(set(stopwords_domain))]
    text = " ".join(tokenized)
    return text


class FastText:
    """
    A class used to train a FastText model and generate embeddings for text data.

    Attributes
    ----------
    method : str
        The training method for the FastText model.
    model : fasttext.FastText._FastText
        The trained FastText model.
    """

    def __init__(self, method='skipgram', preprocessor=None):
        """
        Initializes the FastText with a preprocessor and a training method.

        Parameters
        ----------
        method : str, optional
            The training method for the FastText model.
        """
        self.method = method
        self.model = None
        self.preprocessor = preprocessor

    def train(self, texts, E=None):
        """
        Trains the FastText model with the given texts.

        Parameters
        ----------
        texts : list of str
            The texts to train the FastText model.
        """

        # TODO : note : may need to change
        # self.load_model()
        # return

        if E is None:
            with open(path, mode='w', encoding="utf-8") as f:
                for i in texts: f.write(i + os.linesep)
            self.model = fasttext.train_unsupervised(path, self.method)
        else:
            with open(E, mode='w', encoding="utf-8") as f:
                for i in texts: f.write(i + os.linesep)
            self.model = fasttext.train_unsupervised(E, self.method)

        # self.save_model()

        return self.model

    def get_query_embedding(self, query):
        """
        Generates an embedding for the given query.

        Parameters
        ----------
        query : str
            The query to generate an embedding for.
        tf_idf_vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
            The TfidfVectorizer to transform the query.
        do_preprocess : bool, optional
            Whether to preprocess the query.

        Returns
        -------
        np.ndarray
            The embedding for the query.
        """
        # TODO : note : seems like im missing two params?
        if self.preprocessor is None: return self.model.get_sentence_vector(query)
        return self.model.get_sentence_vector(self.preprocessor(query))

    def analogy(self, word1, word2, word3):
        """
        Perform an analogy task: word1 is to word2 as word3 is to __.

        Args:
            word1 (str): The first word in the analogy.
            word2 (str): The second word in the analogy.
            word3 (str): The third word in the analogy.

        Returns:
            str: The word that completes the analogy.
        """
        # TODO: note : ready type
        # return self.model.get_analogies(word1, word2, word3)[0][1]

        # Obtain word embeddings for the words in the analogy
        # TODO
        vectors = {word: self.model.get_word_vector(word) for word in [word1, word2, word3]}

        # Perform vector arithmetic
        # TODO
        result1 = vectors[word3] + vectors[word2] - vectors[word1]
        result2 = vectors[word3] - vectors[word2] + vectors[word1]

        # Create a dictionary mapping each word in the vocabulary to its corresponding vector
        # TODO
        word_to_vec = {}
        for i in self.model.get_words():
            word_to_vec[i] = self.model.get_word_vector(i)

        # Exclude the input words from the possible results
        # TODO
        for i in [word1, word2, word3]:
            if i in word_to_vec: del word_to_vec[i]

        # Find the word whose vector is closest to the result vector
        # TODO
        candidate, min_distance = None, None
        for k, v in word_to_vec.items():
            new_distance = min(np.linalg.norm(result1 - v), np.linalg.norm(result2 - v))
            if candidate is None:
                candidate, min_distance = v, new_distance
            if new_distance < min_distance:
                candidate, min_distance = k, new_distance

        return candidate

    def save_model(self, path='FastText_model.bin'):
        """
        Saves the FastText model to a file.

        Parameters
        ----------
        path : str, optional
            The path to save the FastText model.
        """
        self.model.save_model(path)

    def load_model(self, path="FastText_model.bin"):
        """
        Loads the FastText model from a file.

        Parameters
        ----------
        path : str, optional
            The path to load the FastText model.
        """
        self.model = fasttext.load_model(path)

    def prepare(self, dataset, mode, save=False, path='FastText_model.bin'):
        """
        Prepares the FastText model.

        Parameters
        ----------
        dataset : list of str
            The dataset to train the FastText model.
        mode : str
            The mode to prepare the FastText model.
        """
        if mode == 'train':
            self.train(dataset)
        if mode == 'load':
            self.load_model(path)
        if save:
            self.save_model(path)


if __name__ == "__main__":
    ft_model = FastText(preprocessor=preprocess_text, method='skipgram')

    path = 'training.txt'
    ft_data_loader = FastTextDataLoader(path)

    X, y = ft_data_loader.create_train_data()

    ft_model.prepare(None, mode="load")
    # model = ft_model.train(X)
    # ft_model.prepare(model, mode="save")

    print(10 * "*" + "Similarity" + 10 * "*")
    word = 'queen'
    neighbors = ft_model.model.get_nearest_neighbors(word, k=5)

    for neighbor in neighbors:
        print(f"Word: {neighbor[1]}, Similarity: {neighbor[0]}")

    print(10 * "*" + "Analogy" + 10 * "*")
    word1 = "man"
    word2 = "king"
    word3 = "queen"
    print(
        f"Similarity between {word1} and {word2} is like similarity between {word3} and {ft_model.analogy(word1, word2, word3)}")
