import numpy as np


class Scorer:
    def __init__(self, index, number_of_documents, all = None):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The index to score the documents with.
        number_of_documents : int
            The number of documents in the index.
        """

        if all is None :
            self.index = index.get_index()
            self.all = None
        else:
            self.index = index
            self.all = all.get_index()

        self.idf = {}
        self.N = number_of_documents

    def get_list_of_documents(self,query):
        """
        Returns a list of documents that contain at least one of the terms in the query.

        Parameters
        ----------
        query: List[str]
            The query to be scored

        Returns
        -------
        list
            A list of documents that contain at least one of the terms in the query.
        
        Note
        ---------
            The current approach is not optimal but we use it due to the indexing structure of the dict we're using.
            If we had pairs of (document_id, tf) sorted by document_id, we could improve this.
                We could initialize a list of pointers, each pointing to the first element of each list.
                Then, we could iterate through the lists in parallel.
            
        """
        list_of_documents = []
        for term in query:
            if term in self.index.keys():
                list_of_documents.extend(self.index[term].keys())
        return list(set(list_of_documents))
    
    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.

        Parameters
        ----------
        term : str
            The term to get the inverse document frequency for.

        Returns
        -------
        float
            The inverse document frequency of the term.
        
        Note
        -------
            It was better to store dfs in a separate dict in preprocessing.
        """
        idf_ = self.idf.get(term, None)
        if idf_ is None:
            if self.all is None:
                # TODO
                df = len(self.index.get(term, dict()))
                # TODO : note : can change this
                idf_ = np.log2(self.N / (df + 1))
                self.idf[term] = idf_
            else :
                df = len(self.all.get(term, dict()))
                # TODO : note : can change this
                idf_ = np.log2(self.N / (df + 1))
                self.idf[term] = idf_
        return idf_

    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.

        Parameters
        ----------
        query : List[str]
            The query to get the term frequencies for.

        Returns
        -------
        dict
            A dictionary of the term frequencies of the terms in the query.
        """
        # TODO
        result = {}
        for term in query:
            if term not in result: result[term] = 0
            result[term] += 1
        return result

    def compute_scores_with_vector_space_model(self, query, method):
        """
        compute scores with vector space model

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c))
            The method to use for searching.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """

        # TODO
        scores = {}
        query_tfs = self.get_query_tfs(query)
        for document_id in self.get_list_of_documents(query):
            scores[document_id] = self.get_vector_space_model_score(query, query_tfs, document_id, method.split(".")[0],
                                                                    method.split(".")[1])
        return scores

    def get_vector_space_model_score(
        self, query, query_tfs, document_id, document_method, query_method
    ):
        """
        Returns the Vector Space Model score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        query_tfs : dict
            The term frequencies of the terms in the query.
        document_id : str
            The document to calculate the score for.
        document_method : str (n|l)(n|t)(n|c)
            The method to use for the document.
        query_method : str (n|l)(n|t)(n|c)
            The method to use for the query.

        Returns
        -------
        float
            The Vector Space Model score of the document for the query.
        """

        # TODO
        pass
    def get_vector_space_model_score(self, query, query_tfs, document_id, document_method, query_method):
        # TODO
        d_vec, q_vec = [], []
        for term in query:
            dtf, qtf = self.index.get(term, {}).get(document_id, 0), query_tfs.get(term, 0)

            dtf = dtf if document_method[0] == 'n' else np.log(dtf + 1)
            qtf = qtf if query_method[0] == 'n' else np.log(qtf + 1)

            ddf = 1 if document_method[1] == 'n' else self.get_idf(term)
            qdf = 1 if query_method[1] == 'n' else self.get_idf(term)

            d_vec.append(dtf * ddf)
            q_vec.append(qtf * qdf)

        d_vec = d_vec if document_method[2] == 'n' or np.linalg.norm(d_vec) == 0 else d_vec / np.linalg.norm(d_vec)
        q_vec = q_vec if query_method[2] == 'n' or np.linalg.norm(q_vec) == 0 else q_vec / np.linalg.norm(q_vec)
        return np.dot(np.array(d_vec), np.array(q_vec))

    def compute_scores_with_okapi_bm25(self, query, average_document_field_length, document_lengths):
        """
        compute scores with okapi bm25

        Parameters
        ----------
        query: List[str]
            The query to be scored
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """
        # TODO
        scores = {}
        k1 = 1.5
        b = 1
        for id in self.get_list_of_documents(query):
            # TODO : note : added some parameters to the get_okapi_bm25_score for easier computation
            scores[id] = self.get_okapi_bm25_score(query, id, average_document_field_length, document_lengths, k1, b)
        return scores

    def get_okapi_bm25_score(self, query, document_id, average_document_field_length, document_lengths, k1=1.5, b=0.75):
        """
        Returns the Okapi BM25 score of a document for a query.

        Parameters
        ----------
        b: float
            tuning parameter
        k1: float
            tuning parameter
        query: List[str]
            The query to be scored
        document_id : str
            The document to calculate the score for.
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        float
            The Okapi BM25 score of the document for the query.
        """
        # TODO
        pass

    def compute_scores_with_unigram_model(
        self, query, smoothing_method, document_lengths=None, alpha=0.5, lamda=0.5
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            A dictionary of the document IDs and their scores.
        """

        # TODO
        pass

    def compute_score_with_unigram_model(
        self, query, document_id, smoothing_method, document_lengths, alpha, lamda
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        document_id : str
            The document to calculate the score for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            The Unigram score of the document for the query.
        """

        # TODO
        pass

        # TODO : note : had some help from : https://github.com/yutayamazaki/okapi-bm25/blob/master/okapi_bm25/bm25.py
        score = 0
        for term in query:
            idf = self.get_idf(term)

            if term not in self.index or document_id not in self.index[term]:
                tf = 0
            else:
                tf = self.index[term][document_id]

            if document_id not in document_lengths:
                dl = 0
            else:
                dl = document_lengths[document_id]
            if tf + k1 * (1 - b + (b * dl / average_document_field_length)) == 0 : continue
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + (b * dl / average_document_field_length)))

        return score
