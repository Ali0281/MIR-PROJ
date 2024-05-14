import json
from typing import List, Dict

from Logic.core.indexer.index_reader import Index_reader
from Logic.core.indexer.indexes_enum import Indexes, Index_types
from Logic.core.utility.preprocess import Preprocessor
from Logic.core.utility.scorer import Scorer
import numpy as np
from Logic.core.utility import Preprocessor, Scorer
from Logic.core.indexer import Indexes, Index_types, Index_reader


class SearchEngine:
    def __init__(self):
        """
        Initializes the search engine.

        """
        path = '/index'
        self.document_indexes = {
            Indexes.STARS: Index_reader(path, Indexes.STARS),
            Indexes.GENRES: Index_reader(path, Indexes.GENRES),
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES)
        }
        self.tiered_index = {
            Indexes.STARS: Index_reader(path, Indexes.STARS, Index_types.TIERED),
            Indexes.GENRES: Index_reader(path, Indexes.GENRES, Index_types.TIERED),
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES, Index_types.TIERED)
        }
        self.document_lengths_index = {
            Indexes.STARS: Index_reader(path, Indexes.STARS, Index_types.DOCUMENT_LENGTH),
            Indexes.GENRES: Index_reader(path, Indexes.GENRES, Index_types.DOCUMENT_LENGTH),
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES, Index_types.DOCUMENT_LENGTH)
        }
        self.metadata_index = Index_reader(path, Indexes.DOCUMENTS, Index_types.METADATA)

    def search(
        self,
        query,
        method,
        weights,
        safe_ranking=True,
        max_results=10,
        smoothing_method=None,
        alpha=0.5,
        lamda=0.5,
    ):
        """
        searches for the query in the indexes.

        Parameters
        ----------
        query : str
            The query to search for.
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25 | Unigram
            The method to use for searching.
        weights: dict
            The weights of the fields.
        safe_ranking : bool
            If True, the search engine will search in whole index and then rank the results.
            If False, the search engine will search in tiered index.
        max_results : int
            The maximum number of results to return. If None, all results are returned.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        list
            A list of tuples containing the document IDs and their scores sorted by their scores.
        """

        preprocessor = Preprocessor([query], "C:/Users/Ali/PycharmProjects/MIR-PROJ/Logic/core/utility/stopwords.txt")

        query = preprocessor.preprocess()[0].split()
        print(query)

        scores = {}
        if method == "unigram":
            self.find_scores_with_unigram_model(
                query, smoothing_method, weights, scores, alpha, lamda
            )
        elif safe_ranking:
            self.find_scores_with_safe_ranking(query, method, weights, scores)
        else:
            self.find_scores_with_unsafe_ranking(query, method, weights, max_results, scores)

        final_scores = {}
        self.aggregate_scores(weights, scores, final_scores)

        result = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        if max_results is not None:
            result = result[:max_results]

        return result

    def aggregate_scores(self, weights, scores, final_scores):
        """
        Aggregates the scores of the fields.

        Parameters
        ----------
        weights : dict
            The weights of the fields.
        scores : dict
            The scores of the fields.
        final_scores : dict
            The final scores of the documents.
        """
        # TODO
        for doc_id, score_dict in scores.items():
            final_score = 0
            for field, score in score_dict.items():
                final_score += score * weights[field]
            final_scores[doc_id] = final_score

    def find_scores_with_unsafe_ranking(self, query, method, weights, max_results, scores):
        """
        Finds the scores of the documents using the unsafe ranking method using the tiered index.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        max_results : int
            The maximum number of results to return.
        scores : dict
            The scores of the documents.
        """
        for tier in ["first_tier", "second_tier", "third_tier"]:
            iter_scores = {}
            for field in weights:
                # TODO
                scorer = Scorer(self.tiered_index[field].get_index()[tier],
                                self.metadata_index.get_index()["document_count"], self.document_indexes[field])
                if method == "OkapiBM25": scores_for_field = scorer.compute_scores_with_okapi_bm25(query,
                                                                                                   self.metadata_index.get_index()[
                                                                                                       "averge_document_length"][
                                                                                                       field.value],
                                                                                                   self.document_lengths_index[
                                                                                                       field].get_index())
                if "." in method: scores_for_field = scorer.compute_scores_with_vector_space_model(query, method)
                for doc_id, score in scores_for_field.items():
                    if doc_id not in iter_scores:
                        iter_scores[doc_id] = {}
                    iter_scores[doc_id][field] = score


            #scores = self.merge_scores(scores, iter_scores, weights)

            for id in set(scores.keys()).union(set(iter_scores.keys())):
                if id not in scores: scores[id] = {}
                for field in weights:
                    scores[id][field] = max(scores.get(id, {}).get(field, 0), iter_scores.get(id, {}).get(field, 0))


            if len(scores) >= max_results: break

    def find_scores_with_safe_ranking(self, query, method, weights, scores):
        """
        Finds the scores of the documents using the safe ranking method.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        scores : dict
            The scores of the documents.
        """

        for field in weights:
            # TODO
            scorer = Scorer(self.document_indexes[field], self.metadata_index.get_index()["document_count"])
            if method == "OkapiBM25": scores_for_field = scorer.compute_scores_with_okapi_bm25(query,
                                                                                               self.metadata_index.get_index()[
                                                                                                   "averge_document_length"][
                                                                                                   field.value],
                                                                                               self.document_lengths_index[
                                                                                                   field].get_index())
            if "." in method: scores_for_field = scorer.compute_scores_with_vector_space_model(query, method)
            for doc_id, score in scores_for_field.items():
                if doc_id not in scores:
                    scores[doc_id] = {}
                scores[doc_id][field] = score
            # TODO
            pass

    def find_scores_with_unigram_model(
        self, query, smoothing_method, weights, scores, alpha=0.5, lamda=0.5
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        weights : dict
            A dictionary mapping each field (e.g., 'stars', 'genres', 'summaries') to its weight in the final score. Fields with a weight of 0 are ignored.
        scores : dict
            The scores of the documents.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.
        """
        # TODO
        pass

    def merge_scores(self, scores1, scores2, weights):
        """
        Merges two dictionaries of scores.

        Parameters
        ----------
        scores1 : dict
            The first dictionary of scores.
        scores2 : dict
            The second dictionary of scores.

        Returns
        -------
        dict
            The merged dictionary of scores.
        """
        # TODO
        merged_scores = {}
        for id in set(scores1.keys()).union(set(scores2.keys())):
            merged_scores[id] = {}
            for field in weights:
                merged_scores[id][field] = max(scores1.get(id, {}).get(field, 0), scores2.get(id, {}).get(field, 0))
        return merged_scores

# just for testing
def get_movie_by_id(id: str, movies_dataset: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Get movie by its id

    Parameters
    ---------------------------------------------------------------------------------------------------
    id: str
        The id of the movie

    movies_dataset: List[Dict[str, str]]
        The dataset of movies

    Returns
    ----------------------------------------------------------------------------------------------------
    dict
        The movie with the given id
    """

    """result = movies_dataset.get(
        id,
        {
            "Title": "This is movie's title",
            "Summary": "This is a summary",
            "URL": "https://www.imdb.com/title/tt0111161/",
            "Cast": ["Morgan Freeman", "Tim Robbins"],
            "Genres": ["Drama", "Crime"],
            "Image_URL": "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg",
        },
    )"""
    result = None
    for document in movies_dataset:
        if document["id"] == id:
            result = document
            break
    if result is None :
        result = movies_dataset[0]

    result["Image_URL"] = (
        "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg"
        # a default picture for selected movies
    )
    result["URL"] = (
        f"https://www.imdb.com/title/{result.get('id', 'NOT FOUND')}"  # The url pattern of IMDb movies
    )
    return result

if __name__ == "__main__":
    search_engine = SearchEngine()
    query = "spiderman wonderland"
    method = "ltc.lnc"
    # method = "ltn.lnn"
    # method = "OkapiBM25"
    weights = {
        Indexes.STARS: 0.1,
        Indexes.GENRES: 0.05,
        Indexes.SUMMARIES: 0.85
    }
    result = search_engine.search(query, method, weights, safe_ranking=False)
    # print(result)

    with open("C:/Users/Ali/PycharmProjects/MIR-PROJ/Logic/core/IMDB_movies.json", "r") as f:
        movies_dataset = json.load(f)

    for r in result:
        print(get_movie_by_id(r[0], movies_dataset)["title"])
