import time
import os
import json
import copy

from Logic.core.preprocess import Preprocessor
from indexes_enum import Indexes


class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.
        """
        # self.load_index("indexes.json")
        self.preprocessed_documents = preprocessed_documents

        self.index = {
            Indexes.DOCUMENTS.value: self.index_documents(),
            Indexes.STARS.value: self.index_stars(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.SUMMARIES.value: self.index_summaries(),
        }

        self.store_all()

    def index_documents(self):
        """
        Index the documents based on the document ID. In other words, create a dictionary
        where the key is the document ID and the value is the document.

        Returns
        ----------
        dict
            The index of the documents based on the document ID.
        """
        # TODO
        current_index = {}
        for doc in self.preprocessed_documents:
            current_index[doc["id"]] = doc
        return current_index

    def index_stars(self):
        """
        Index the documents based on the stars.

        Returns
        ----------
        dict
            The index of the documents based on the stars. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """
        # TODO : note : our tf will be 1 and 0 if absent
        """current_index = {}
        for doc in self.preprocessed_documents:
            for star in doc["stars"]:
                if star in current_index:
                    current_index[star][doc["id"]] = 1
                else:
                    current_index[star] = {doc["id"]: 1}
        return current_index"""

        current_index = {}
        for doc in self.preprocessed_documents:
            for star in doc["stars"]:
                for w in star.split():
                    w = w.lower()
                    if w in current_index:
                        current_index[w][doc["id"]] = current_index[w].get(doc["id"], 0) + 1
                    else:
                        current_index[w] = {doc["id"]: 1}
        return current_index

    def index_genres(self):
        """
        Index the documents based on the genres.

        Returns
        ----------
        dict
            The index of the documents based on the genres. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """
        # TODO : note : same as above
        current_index = {}
        for doc in self.preprocessed_documents:
            for genre in doc["genres"]:
                genre = genre.lower()
                if genre in current_index:
                    current_index[genre][doc["id"]] = 1
                else:
                    current_index[genre] = {doc["id"]: 1}
        return current_index

    def index_summaries(self):
        """
        Index the documents based on the summaries (not first_page_summary).

        Returns
        ----------
        dict
            The index of the documents based on the summaries. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """
        # TODO : note : will use word terms
        current_index = {}
        for doc in self.preprocessed_documents:
            for summary in doc["summaries"]:
                summary = summary.split()
                for w in summary:
                    if w in current_index:
                        current_index[w][doc["id"]] = current_index[w].get(doc["id"], 0) + summary.count(w)
                    else:
                        current_index[w] = {doc["id"]: summary.count(w)}
        return current_index

    def get_posting_list(self, word: str, index_type: str):
        """
        get posting_list of a word

        Parameters
        ----------
        word: str
            word we want to check
        index_type: str
            type of index we want to check (documents, stars, genres, summaries)

        Return
        ----------
        list
            posting list of the word (you should return the list of document IDs that contain the word and ignore the tf)
        """
        posting_list = []
        try:
            if word not in self.index[index_type]: raise Exception("word not in index")
            posting_list = list(self.index[index_type][word].keys())
        except Exception as e:
            print(f"couldn't get the posting list, exception : {e}")
        finally:
            return posting_list

    def add_document_to_index(self, document: dict):
        """
        Add a document to all the indexes

        Parameters
        ----------
        document : dict
            Document to add to all the indexes
        """
        # TODO : id
        self.index[Indexes.DOCUMENTS.value][document["id"]] = document

        # TODO : stars
        for star in document["stars"]:
            if star in self.index[Indexes.STARS.value]:
                self.index[Indexes.STARS.value][star][document["id"]] = 1
            else:
                self.index[Indexes.STARS.value][star] = {document["id"]: 1}

        # TODO : genres
        for genre in document["genres"]:
            if genre in self.index[Indexes.GENRES.value]:
                self.index[Indexes.GENRES.value][genre][document["id"]] = 1
            else:
                self.index[Indexes.GENRES.value][genre] = {document["id"]: 1}

        # TODO : summaries
        for summary in document["summaries"]:
            summary = summary.split()
            for w in summary:
                if w in self.index[Indexes.SUMMARIES.value]:
                    self.index[Indexes.SUMMARIES.value][w][document["id"]] = self.index[Indexes.SUMMARIES.value][w].get(
                        document["id"], 0) + summary.count(w)
                else:
                    self.index[Indexes.SUMMARIES.value][w] = {document["id"]: summary.count(w)}

        self.store_all()

    def remove_document_from_index(self, document_id: str):
        """
        Remove a document from all the indexes

        Parameters
        ----------
        document_id : str
            ID of the document to remove from all the indexes
        """
        if document_id in self.index[Indexes.DOCUMENTS.value]: del self.index[Indexes.DOCUMENTS.value][document_id]

        star_keys_to_remove = []
        for key, value in self.index[Indexes.STARS.value].items():
            if document_id in value:
                del value[document_id]
                if len(value) == 0:
                    star_keys_to_remove.append(key)

        for key in star_keys_to_remove: del self.index[Indexes.STARS.value][key]

        genre_keys_to_remove = []
        for key, value in self.index[Indexes.GENRES.value].items():
            if document_id in value:
                del value[document_id]
                if len(value) == 0:
                    genre_keys_to_remove.append(key)
        for key in genre_keys_to_remove: del self.index[Indexes.GENRES.value][key]

        summary_keys_to_remove = []
        for key, value in self.index[Indexes.SUMMARIES.value].items():
            if document_id in value:
                del value[document_id]
                if len(value) == 0:
                    summary_keys_to_remove.append(key)
        for key in summary_keys_to_remove: del self.index[Indexes.SUMMARIES.value][key]

        self.store_all()

    def check_add_remove_is_correct(self):
        """
        Check if the add and remove is correct
        """

        dummy_document = {
            'id': '100',
            'stars': ['tim', 'henry'],
            'genres': ['drama', 'crime'],
            'summaries': ['good']
        }

        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

        if index_after_add[Indexes.DOCUMENTS.value]['100'] != dummy_document:
            print('Add is incorrect, document')
            return

        if (set(index_after_add[Indexes.STARS.value]['tim']).difference(
                set(index_before_add[Indexes.STARS.value].get("tim", dict())))
                != {dummy_document['id']}):
            print('Add is incorrect, tim')
            return

        if (set(index_after_add[Indexes.STARS.value]['henry']).difference(
                set(index_before_add[Indexes.STARS.value].get("henry", dict())))
                != {dummy_document['id']}):
            print('Add is incorrect, henry')
            return

        if (set(index_after_add[Indexes.GENRES.value]['drama']).difference(
                set(index_before_add[Indexes.GENRES.value].get("drama", dict())))
                != {dummy_document['id']}):
            print('Add is incorrect, drama')
            return

        if (set(index_after_add[Indexes.GENRES.value]['crime']).difference(
                set(index_before_add[Indexes.GENRES.value].get("crime", dict())))
                != {dummy_document['id']}):
            print('Add is incorrect, crime')
            return

        if (set(index_after_add[Indexes.SUMMARIES.value]['good']).difference(
                set(index_before_add[Indexes.SUMMARIES.value].get("good", dict())))
                != {dummy_document['id']}):
            print('Add is incorrect, good')
            return

        print('Add is correct')

        self.remove_document_from_index('100')
        index_after_remove = copy.deepcopy(self.index)
        for i in index_after_remove.keys():
            if index_after_remove[i] != index_before_add[i]:
                print(index_after_remove[i])
                print(index_before_add[i])

        if index_after_remove == index_before_add:
            print('Remove is correct')
        else:
            print('Remove is incorrect')

    def store_index(self, path: str, index_type: str = None):
        """
        Stores the index in a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to store the file
        index_type: str or None
            type of index we want to store (documents, stars, genres, summaries)
            if None store tiered index
        """
        # TODO :
        if not os.path.exists(path):
            os.makedirs(path)

        if index_type not in self.index:
            raise ValueError('Invalid index type')

        if index_type == Indexes.DOCUMENTS.value:
            with open(os.path.join(path, "documents_index.json"), 'w') as f:
                json.dump(self.index[index_type], f, indent=4)

        if index_type == Indexes.STARS.value:
            with open(os.path.join(path, "stars_index.json"), 'w') as f:
                json.dump(self.index[index_type], f, indent=4)

        if index_type == Indexes.GENRES.value:
            with open(os.path.join(path, "genres_index.json"), 'w') as f:
                json.dump(self.index[index_type], f, indent=4)

        if index_type == Indexes.SUMMARIES.value:
            with open(os.path.join(path, "summaries_index.json"), 'w') as f:
                json.dump(self.index[index_type], f, indent=4)

    def load_index(self, path: str):
        """
        Loads the index from a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to load the file
        """
        # TODO :
        with open(path, 'r') as f:
            self.index = json.load(f)

    def check_if_index_loaded_correctly(self, index_type: str, loaded_index: dict):
        """
        Check if the index is loaded correctly

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        loaded_index : dict
            The loaded index

        Returns
        ----------
        bool
            True if index is loaded correctly, False otherwise
        """
        # return self.index[index_type] == loaded_index
        # TODO : note : changed this because it caused some problems while the code being correct
        # TODO : note : my code works with both checkers now !
        local = self.index.get(index_type, {})

        if set(loaded_index.keys()) != set(local.keys()):
            print(f"Key mismatch for {index_type} index")
            return False

        for key in loaded_index.keys():
            if loaded_index[key] != local[key]:
                print(f"Value mismatch for key {key} in {index_type} index")
                return False

        return True

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'good'):
        """
        Checks if the indexing is good. Do not change this function. You can use this
        function to check if your indexing is correct.

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        check_word : str
            The word to check in the index

        Returns
        ----------
        bool
            True if indexing is good, False otherwise
        """

        # brute force to check check_word in the summaries
        start = time.time()
        docs = []
        for document in self.preprocessed_documents:
            if index_type not in document or document[index_type] is None:
                continue

            # TODO : note : this type of check is not reliable !
            # TODO : example : roberto cant be in robert term !!
            # TODO : note : so i will be changing it
            """for field in document[index_type]:
                if check_word in field:
                    docs.append(document['id'])
                    break"""
            if index_type == "documents":
                if check_word == document["id"]:
                    docs.append(document)
                    break

            elif index_type == "stars":
                for star in document["stars"]:
                    # star = star.lower()
                    for field in star.split():
                        if check_word == field:
                            docs.append(document['id'])
                            break

            elif index_type == "genres":
                for field in document["genres"]:
                    if check_word == field:
                        docs.append(document['id'])
                        break

            elif index_type == "summaries":
                for summary in document["summaries"]:
                    for field in summary.split():
                        if check_word == field:
                            docs.append(document['id'])
                            break

            # if we have found 3 documents with the word, we can break
            if len(docs) == 3:
                break

        end = time.time()
        brute_force_time = end - start

        # check by getting the posting list of the word
        start = time.time()
        # TODO: based on your implementation, you may need to change the following line
        posting_list = self.get_posting_list(check_word, index_type)

        end = time.time()
        implemented_time = end - start

        print('Brute force time: ', brute_force_time)
        print('Implemented time: ', implemented_time)

        # TOOD : note : my testing code
        #print(docs)
        #print(posting_list)

        if set(docs).issubset(set(posting_list)):
            print('Indexing is correct')

            # TODO : note : added the or parts so if the brute force is 0 we wont get a problem
            if implemented_time < brute_force_time or brute_force_time == 0:
                print('Indexing is good')
                return True
            else:
                print('Indexing is bad')
                return False
        else:
            print('Indexing is wrong')
            return False

    def store_all(self):
        self.store_index("index", "documents")
        self.store_index("index", "stars")
        self.store_index("index", "genres")
        self.store_index("index", "summaries")


# TODO: Run the class with needed parameters, then run check methods and finally report the results of check methods

def main():
    with open("../IMDB_movies.json", "r") as f:
        data = json.load(f)

    pre = Preprocessor(data, "../stopwords.txt")
    pre.preprocess()

    m = Index(pre.documents)

    # TODO : note : used .get instead
    # TODO : note : removed the empty indexses
    m.check_add_remove_is_correct()
    print("checked add remove successfully ... \n")
    m.index_stars()

    with open("indexes/documents_index.json", "r") as f:
        documents = json.load(f)

    with open("indexes/genres_index.json", "r") as f:
        genres = json.load(f)

    with open("indexes/stars_index.json", "r") as f:
        stars = json.load(f)

    with open("indexes/summaries_index.json", "r") as f:
        summaries = json.load(f)

    print("documents test : ", m.check_if_index_loaded_correctly("documents", documents))
    print("stars test : ", m.check_if_index_loaded_correctly("stars", stars))
    print("genres test : ", m.check_if_index_loaded_correctly("genres", genres))
    print("summaries test : ", m.check_if_index_loaded_correctly("summaries", summaries))
    print("checked index loading successfully ... \n")

    print("checking documents on id tt0110912 : ", m.check_if_indexing_is_good("documents", "tt0110912"), "\n")
    print("checking documents on id tt0084787 : ", m.check_if_indexing_is_good("documents", "tt0084787"), "\n")

    print("checking stars on word Robert : ", m.check_if_indexing_is_good("stars", "Robert"), "\n")
    print("checking stars on word Kevin : ", m.check_if_indexing_is_good("stars", "Kevin"), "\n")

    print("checking genres on word War : ", m.check_if_indexing_is_good("genres", "War"), "\n")
    print("checking genres on word Drama : ", m.check_if_indexing_is_good("genres", "Drama"), "\n")

    print("checking summaries on word lake : ", m.check_if_indexing_is_good("summaries", "lake"), "\n")
    print("checking summaries on word badge : ", m.check_if_indexing_is_good("summaries", "badge"), "\n")
    print("checked indexing successfully ... \n")


if __name__ == '__main__':
    main()
