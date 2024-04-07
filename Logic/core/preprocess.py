import json
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class Preprocessor:
    # TODO : if you had problem using nltk
    #nltk.download('punkt')
    #nltk.download('wordnet')

    def __init__(self, documents: list, path):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed, path to stop words, or other parameters.
        """
        # TODO
        self.documents = documents
        self.stopwords = []
        self.lemmatized_stopwords = []
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = word_tokenize
        try:
            with open(path, 'r') as f:
                for word in f:
                    self.stopwords.append(word.strip().lower())
                self.lemmatized_stopwords = [self.normalize(w) for w in self.stopwords]
        except Exception as e:
            print(f"couldn't get the stopwords file, exception : {e}")

    def preprocess_text(self, text):
        return " ".join(self.remove_stopwords(self.remove_punctuations(self.remove_links(self.normalize(text)))))

    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[str]
            The preprocessed documents.
        """


        if not isinstance(self.documents[0], dict):
            res = []
            for doc in self.documents:
                res.append(self.preprocess_text(doc))

            return res

        # TODO : note : just to save and reuse
        with open("C:/Users/HSM/PycharmProjects/MIR-PROJ-/Logic/core/preprocess.json", "r") as f:
            data = json.load(f)
        self.documents = data
        return data

        for doc in self.documents:
            # TODO : note : only give attention to <first_page_summary>, <summaries>, <synopsis>, <reviews>
            doc["first_page_summary"] = self.preprocess_text(doc["first_page_summary"])

            preprocessed_summaries = []
            for summary in doc["summaries"]:
                preprocessed_summaries.append(self.preprocess_text(summary))
            doc["summaries"] = preprocessed_summaries

            preprocessed_synopsis = []
            for synopsis in doc["synopsis"]:
                preprocessed_synopsis.append(self.preprocess_text(synopsis))
            doc["synopsis"] = preprocessed_synopsis

            preprocessed_reviews = []
            for review in doc["reviews"]:
                preprocessed_reviews.append([self.preprocess_text(review[0]), review[1]])  # TODO : do i need to use the score?
            doc["reviews"] = preprocessed_reviews

            doc["title"] = self.preprocess_text(doc["title"])

            doc["release_year"] = self.preprocess_text(doc["release_year"])

            preprocessed_stars = []
            for star in doc["stars"]:
                preprocessed_stars.append(self.preprocess_text(star))
            doc["stars"] = preprocessed_stars

            preprocessed_writers = []
            for writer in doc["writers"]:
                preprocessed_writers.append(self.preprocess_text(writer))
            doc["writers"] = preprocessed_writers

            preprocessed_directors = []
            for director in doc["directors"]:
                preprocessed_directors.append(self.preprocess_text(director))
            doc["directors"] = preprocessed_directors

            preprocessed_genres = []
            for genre in doc["genres"]:
                preprocessed_genres.append(self.preprocess_text(genre))
            doc["genres"] = preprocessed_genres

            preprocessed_languages = []
            for language in doc["languages"]:
                preprocessed_languages.append(self.preprocess_text(language))
            doc["languages"] = preprocessed_languages

            preprocessed_coo = []
            for c in doc["countries_of_origin"]:
                preprocessed_coo.append(self.preprocess_text(c))
            doc["countries_of_origin"] = preprocessed_coo

        with open('C:/Users/HSM/PycharmProjects/MIR-PROJ-/Logic/core/preprocess.json', 'w') as f:
            json.dump(self.documents, f, indent=4)

        return self.documents

    def normalize(self, text: str):
        """
        Normalize the text by converting it to a lower case, stemming, lemmatization, etc.

        Parameters
        ----------
        text : str
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """
        # TODO
        lemmatizer = WordNetLemmatizer()
        return " ".join([lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(text)])

    def remove_links(self, text: str):
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """
        # TODO
        patterns = [r'\S*http\S*', r'\S*www\S*', r'\S+\.ir\S*', r'\S+\.com\S*', r'\S+\.org\S*', r'\S*@\S*']
        for patt in patterns:
            text = re.sub(patt, "", text)
        return text

    def remove_punctuations(self, text: str):
        """
        Remove punctuations from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        # TODO
        return re.sub(r'[^\w\s]', '', text)

    def tokenize(self, text: str):
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """
        # TODO
        return self.tokenizer(text)

    def remove_stopwords(self, text: str):
        """
        Remove stopwords from the text.

        Parameters
        ----------
        text : str
            The text to remove stopwords from.

        Returns
        ----------
        list
            The list of words with stopwords removed.
        """
        # TODO
        filter_1 = [word for word in self.tokenize(text) if word not in self.stopwords]
        # TODO : note : i added another filter to lemmatize the stopwords and do the filter
        return [word for word in filter_1 if word not in self.lemmatized_stopwords]
        # return  filter_1


def main():
    with open("IMDB_movies.json", "r") as f:
        data = json.load(f)
    print(data[0])
    pre = Preprocessor(data, "C:/Users/HSM/PycharmProjects/MIR-PROJ-/Logic/core/stopwords.txt")
    pre.preprocess()
    #print(pre.documents[:5])
    print(pre.documents[0])

if __name__ == '__main__':
    main()
