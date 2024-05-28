import json

import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from Logic.core.word_embedding.fasttext_model import FastText, preprocess_text

from sklearn.feature_extraction.text import TfidfVectorizer


class ReviewLoader:
    def __init__(self,
                 file_path="C:/Users/Ali/PycharmProjects/MIR-PROJ/Logic/core/classification/labeled-comments.csv"):
        self.file_path = file_path
        self.fasttext_model = None
        self.review_tokens = []
        self.sentiments = []
        self.embeddings = []
        self.label_encoder = LabelEncoder()

        self.DF = None

    def load_data(self):
        """
        Load the data from the csv file and preprocess the text. Then save the normalized tokens and the sentiment labels.
        Also, load the fasttext model.
        """
        self.fasttext_model = FastText()
        self.label_encoder = LabelEncoder()
        self.DF = pd.read_csv(self.file_path)[:5000]

        # self.DF["review"] = self.DF["review"].apply(lambda x: preprocess_text(x)).tolist()
        # self.DF.to_csv("C:/Users/Ali/PycharmProjects/MIR-PROJ/Logic/core/classification/labeled-comments-preprocessed.csv")

        self.review_tokens = self.DF["review"].apply(lambda x: preprocess_text(x)).tolist()

        self.review_tokens = self.DF["review"].tolist()
        self.sentiments = self.DF["sentiment"].tolist()

        # TODO : note : it gets bugged and all when we do it for naive so it will be implemented else where
        # self.sentiments = self.label_encoder.fit_transform(self.DF["sentiment"].tolist())

        # TODO : note : just for not being null in case of naive bayes cause it doesnt use .get_emb..
        self.embeddings = self.review_tokens

        # self.fasttext_model.train([" ".join(i) for i in self.review_tokens])

    def get_embeddings(self):
        """
        Get the embeddings for the reviews using the fasttext model.
        """
        self.fasttext_model.prepare(None, mode="load")
        # model = self.fasttext_model.train(self.review_tokens, E="training-clustering.txt")
        # self.fasttext_model.prepare(model, mode="save", path="classification")

        # with open('embeddings.json', "r") as f:
        #    self.embeddings = np.array( float(i) for i in  json.load(f)["embeddings"] )
        # return

        self.embeddings = []
        for tokens in tqdm.tqdm(self.review_tokens, desc="embeddings ..."):
            self.embeddings.append(self.fasttext_model.get_query_embedding(tokens))

        self.embeddings = np.array(self.embeddings)
        # with open('embeddings.json', 'w') as f:
        #    arr = json.dumps({"embeddings": self.embeddings.tolist()})
        #    json.dump(arr, f, indent=4)

    def split_data(self, test_data_ratio=0.2, enc=True):
        """
        Split the data into training and testing data.

        Parameters
        ----------
        test_data_ratio: float
            The ratio of the test data
        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            Return the training and testing data for the embeddings and the sentiments.
            in the order of x_train, x_test, y_train, y_test
        """
        x_train, x_test, y_train, y_test = train_test_split(self.embeddings, self.sentiments, test_size=test_data_ratio,
                                                            random_state=42)
        if enc:
            y_train = self.label_encoder.fit_transform(y_train)
            y_test = self.label_encoder.transform(y_test)
        return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

    def get_model(self):
        return self.fasttext_model