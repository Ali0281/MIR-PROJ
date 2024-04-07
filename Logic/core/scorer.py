import logging

import numpy as np


class Scorer:
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The index to score the documents with.
        number_of_documents : int
            The number of documents in the index.
        """

        self.index = index
        self.idf = {}
        self.N = number_of_documents

    def get_list_of_documents(self, query):
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
            if term in self.index.get_index().keys():
                list_of_documents.extend(self.index.get_index()[term].keys())
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
        idf = self.idf.get(term, None)
        if idf is None:
            # TODO
            df = len(self.index.get_index().get(term, {}))
            # TODO : note : can change this
            if df > 0: idf = np.log((self.N - df + 0.5) / (df + 0.5))

        return 0 if idf is None else idf

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
            if term in result:
                result[term] += 1
            else:
                result[term] = 1
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
        result = {}
        qtf = self.get_query_tfs(query)
        for document in self.get_list_of_documents(query):
            result[document] = self.get_vector_space_model_score(query, qtf, document, method.split(".")[0],
                                                                 method.split(".")[1])
        return result

    def get_vector_space_model_score(self, query, query_tfs, document_id, document_method, query_method):
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
        score = 0

        for term in query:
            if term not in self.index.get_index() or document_id not in self.index.get_index()[term]: continue
            tf, idf, q_tf = self.index.get_index()[term][document_id], self.get_idf(term), query_tfs[term]

            if document_method[0] == 'n':
                dtf = tf
            else:
                dtf = 1 + np.log(tf)

            if query_method[0] == 'n':
                qtf = 1
            else:
                qtf = 1 + np.log(q_tf)

            if document_method[1] == 'n':
                dl = 1
            else:
                dl = tf / (tf + 0.5 + 1.5 * (len(self.index.get_index()[term]) / self.N))

            if query_method[1] == 'n':
                ql = 1
            else:
                ql = q_tf / (q_tf + 0.5 + 1.5 * (len(query) / self.N))

            if document_method[2] == 'n':
                dn = 1
            else:
                dn = np.sqrt(np.sum([v ** 2 for v in self.index.get_index()[term].values()]))

            if query_method[2] == 'n':
                qn = 1
            else:
                qn = np.sqrt(np.sum([v ** 2 for v in query_tfs.values()]))

            score += (dtf * qtf * idf) / (dl * ql * dn * qn)

        return score

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
        result = {}
        for document_id in self.index.values():
            result[document_id] = self.get_okapi_bm25_score(query, document_id, average_document_field_length,
                                                            document_lengths)
        return result

    def get_okapi_bm25_score(self, query, document_id, average_document_field_length, document_lengths):
        """
        Returns the Okapi BM25 score of a document for a query.

        Parameters
        ----------
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
        k1 = 1.5
        b = 0.75
        score = 0
        for term in query:
            if term in self.index and document_id in self.index[term]:
                tf = self.index[term][document_id]
                idf = self.get_idf(term)
                doc_length = document_lengths.get(document_id, 0)
                score += idf * (
                            (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / average_document_field_length))))
        return score
