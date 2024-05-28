import json
import string

import nltk
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synposis, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """

    def __init__(self, file_path):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.file_path = file_path
        self.DF = None

    def read_data_to_df(self):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synposis, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synposis, summaries, reviews, titles, genres).
        """
        movies_dataset = None
        # TODO : note : imported pre processed data so i have lower over head | could have used path too
        with open("C:/Users/Ali/PycharmProjects/MIR-PROJ/Logic/core/preprocess.json", "r") as f:
            movies_dataset = json.load(f)

        values = []
        for movie in movies_dataset:
            values.append({"id": movie["id"], "synposis": movie["synposis"],
                           "summaries": movie["summaries"], "reviews": movie["reviews"],
                           "titles": movie["title"], "genres": movie["genres"]})

        self.DF = pd.DataFrame.from_dict(values)

    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        self.read_data_to_df()

        # for merging the sub lists into a complete string
        for index, document_summary in enumerate(self.DF["summaries"]):
            text = []
            for summary in document_summary:
                text.append(summary)
            self.DF.loc[index, "summaries"] = " ".join(text)

        for index, document_synposis in enumerate(self.DF["synposis"]):
            text = []
            for synposis in document_synposis:
                text.append(synposis)
            self.DF.loc[index, "synposis"] = " ".join(text)

        for index, document_review in enumerate(self.DF["reviews"]):
            text = []
            for review in document_review:
                text.append(review[0])
            self.DF.loc[index, "reviews"] = " ".join(text)


        # TODO : removed cause im using an preprocessed file
        # tqdm.pandas(desc="Preprocessing text")
        # self.DF["synposis"] = self.DF["synposis"].fillna('').progress_apply(lambda x: self.preprocess_text(x))
        # self.DF["summaries"] = self.DF["summaries"].fillna('').progress_apply(lambda x: self.preprocess_text(x))
        # self.DF["reviews"] = self.DF["reviews"].fillna('').progress_apply(lambda x: self.preprocess_text(x))
        # self.DF["synposis"] = self.DF["synposis"].fillna('').progress_apply(lambda x: self.preprocess_text(x))

        self.DF['text'] = self.DF[['synposis', 'summaries', 'reviews', 'titles']].fillna('').agg(' '.join, axis=1)

        # label_encoder = LabelEncoder()
        # self.DF['genres'] = label_encoder.fit_transform(self.DF['genres'])

        # self.DF["genres"] = self.DF['genres'].explode()
        # self.DF["genres"][:] = self.DF["genres"].factorize()[0]
        # self.DF["genres"] = self.DF["genres"].groupby(level=0).agg(list)

        # enc.fit(all_genres)
        # self.DF['genres'] = self.DF['genres'].apply(enc.transform)

        # TODO : note : i just used the first genre.
        self.DF["genres"] = self.DF["genres"].fillna('').apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        all_genres = self.DF["genres"].apply(pd.Series).stack().values
        enc = LabelEncoder()
        self.DF['genres'] = enc.fit_transform(self.DF['genres'])


        # X = self.DF[["synposis", "summaries", "reviews", "titles"]]
        X = self.DF["text"]
        y = self.DF["genres"].values

        return X, y

    def preprocess_text(self, text, minimum_length=1, stopword_removal=True, stopwords_domain=[], lower_case=True,
                        punctuation_removal=True):
        if lower_case: text = text.lower()
        tokenized = [i for i in word_tokenize(text) if len(i) >= minimum_length]
        if punctuation_removal: tokenized = [i for i in tokenized if i not in string.punctuation]
        if stopword_removal: tokenized = [i for i in tokenized if i not in set(stopwords.words('english')).union(set(stopwords_domain))]
        text = " ".join(tokenized)
        return text
