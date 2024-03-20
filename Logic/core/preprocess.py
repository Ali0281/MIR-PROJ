import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class Preprocessor:

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
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = word_tokenize
        try:
            with open(path, 'r') as f:
                for word in f:
                    self.stopwords.append(word.strip().lower())
        except Exception as e:
            print(f"couldn't get the stopwords file, exception : {e}")

    def preprocess_text(self, text):
        return self.remove_stopwords(self.remove_punctuations(self.remove_links(self.normalize(text))))

    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[str]
            The preprocessed documents.
        """
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
                preprocessed_reviews.append(self.preprocess_text(review))
            doc["reviews"] = preprocessed_reviews

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
        return [self.lemmatizer.lemmatize(w.lower) for w in self.tokenize(text)]

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
        return " ".join([word for word in self.tokenize(text) if word not in self.stopwords])
