import json

import numpy as np
import itertools
import random
from string import punctuation


class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """

        # TODO : note : i will use dict["summaries"] or .join of it
        # TODO : note : i used joined summaries for documents and i will use their order as id

        # self.documents_ids = document_ids  # TODO : note : mine

        self.documents = documents
        self.num_hashes = num_hashes
        self.num_documents = len(documents)
        # self.num_shingles = 0  # TODO : need to be updated
        self.hashes = None  # TODO
        # self.presence = None  # TODO : need to be updated
        self.documents_shingled = None  # TODO
        self.shingles = set()  # TODO

    def update_shingles(self, _shingle):
        self.shingles.update(_shingle)
        # self.presence = sorted(self.shingles)
        # self.num_shingles = len(self.shingles)

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        # TODO : note : i will shingle by word as the slides imply
        # TODO : note : using set as df doesnt matter
        shing = set()
        # TODO : note : remove not needed punctuations for pure words
        words = [word.strip(punctuation) for word in document.split()]
        for i in range(0, len(words) + 1 - k):
            shing.add(" ".join(words[i: i + k]))

        self.update_shingles(shing)
        return shing

        '''
        shing = set()
        summary = " ".join([word.strip(punctuation) for word in document.split()])
        #summary = document
        for i in range(len(summary) + 1 - k):
            shing.add(summary[i : i + k])
        self.update_shingles(shing)
        return shing
        '''

    def shingle_all_documents(self, k=2):
        self.documents_shingled = []
        for i in range(self.num_documents):
            self.documents_shingled.append(self.shingle_document(self.documents[i], k))

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        self.shingle_all_documents(2)
        characteristic = np.zeros((self.num_documents, len(self.shingles)))
        for i, item in enumerate(self.documents):
            for j, shingle in enumerate(self.shingles):
                if shingle in item: characteristic[i, j] = 1
                else : characteristic[i, j] = 0
        return characteristic

    def create_hashes(self):
        # TODO : note : use a permutation as the hash itself
        self.hashes = []
        for i in range(self.num_hashes):
            self.hashes.append(np.array(np.random.permutation(len(self.shingles))))
        self.hashes = np.array(self.hashes)

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        characteristic = self.build_characteristic_matrix()
        min_hash_signatures = np.full((self.num_hashes, self.num_documents), np.inf)
        self.create_hashes()
        for i in range(self.num_documents):
            for j in range(len(self.shingles)):
                if characteristic[i, j] == 1:
                    min_hash_signatures[:, i] = np.minimum(min_hash_signatures[:, i], self.hashes[:, j])
        return min_hash_signatures

    def lsh_buckets(self, signature, bands=40, rows_per_band=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        # TODO : need to add the id part!
        buckets = {}
        counter = 0
        for i in range(bands):
            bh = {}
            for j in range(self.num_documents):
                slice_ = tuple(signature[rows_per_band * i: rows_per_band * (i + 1), j])
                if slice_ not in bh:
                    bh[slice_] = counter
                    buckets[counter] = [j]
                    counter += 1
                else:
                    if j not in buckets[bh[slice_]]: buckets[bh[slice_]].append(j)

        res = {}
        checker = set()
        for k in buckets.values():
            if len(k) == 0: continue
            checker.add(tuple(k))
        for index, value in enumerate(checker): res[index] = value
        return res

        return buckets

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        # TODO
        return self.lsh_buckets(self.min_hash_signature())

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        if len(first_set.union(second_set)) == 0: return 0
        return len(first_set.intersection(second_set)) / len(first_set.union(second_set))

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)

                        ##########################
                        # random_doc_id = self.documents_ids[random_doc_id]
                        ############################
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score >= 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)



def main():
    # TODO : note : it works with 1 score. just used first summary for lower time, skiped null summaries and they match together !
    with open('LSHFakeData.json', 'r') as f:
        data = json.load(f)
    with open("C:/Users/Ali/PycharmProjects/MIR-PROJ/Logic/core/preprocess.json", "r") as f:
        data1 = json.load(f)

    data.extend(data1)

    texts = []
    for d in data:
        if len(d["summaries"]) == 0: continue
        joined = d["summaries"][0]
        # TOOD : note : code will give acceptable accuracy, but if it didnt, just make bigger summeries , but it will take longer to evaluate
        # TODO : note : it may need some touch for better speed but the speed wansnt mentioned in requirements

        """if len(d["summaries"]) > 3 :
            joined = " ".join(d["summaries"][:3])
        else :
            joined = " ".join(d["summaries"])"""

        texts.append(joined)

    m = MinHashLSH(texts, 400)
    m.jaccard_similarity_test(m.perform_lsh(), texts)


if __name__ == '__main__':
    main()

