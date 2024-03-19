import numpy as np
import re
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

        self.documents = documents
        self.num_hashes = num_hashes
        self.num_documents = len(documents)
        self.num_shingles = 0  # TODO : need to be updated
        self.hashes = None  # TODO : need to be updated
        self.presence = None  # TODO : need to be updated
        self.documents_shingeled = None  # TODO : need to be done

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
        shingles = set()
        words = [word.strip(punctuation) for word in document.split()]
        for i in range(0, len(words) + 1 - k):
            shingles.add(" ".join(words[i: i + k]))
            # print(" ".join(words[i: i + k]))
        return shingles

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        presence = dict()
        shingled_docs = []
        for doc in self.documents:
            shingled_docs.append(self.shingle_document(document=doc, k=2))
            presence.update(shingled_docs[-1])

        # TODO : note : for a definite ans
        self.num_shingles = len(presence)
        self.presence = sorted(presence)

        characteristic = np.zeros(self.num_documents, self.num_shingles)
        for i, shingle in enumerate(presence):
            for j, shingles in enumerate(shingled_docs):
                if shingle in shingles:
                    characteristic[i, j] = 1

        return characteristic

    def create_hashes(self):
        possible_hashes = list(itertools.permutations(range(self.num_shingles)))
        self.hashes = random.sample(possible_hashes, self.num_hashes)

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        characteristic = self.build_characteristic_matrix()
        min_hash_signatures = np.zeros((self.num_hashes, self.num_documents))
        for index_hash, perm in enumerate(self.hashes):
            for index_doc, doc in enumerate(self.documents_shingeled):
                for index_perm, index in enumerate(perm):
                    if self.presence[index] in doc:
                        characteristic[index_hash, index_doc] = index_perm

        return characteristic

    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
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

        for i in range(bands):
            for j in range(self.num_documents):
                slice = signature[rows_per_band * i: rows_per_band * (i + 1), j:j+1]
                hashed = hash(tuple(slice.flatten()))
                if hashed in buckets: buckets[hashed].append(j)
                else : buckets[hashed] = []

        return buckets

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        # TODO : definitely needs debug
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
        return len(first_set.intersection(second_set)) / len(first_set.union(second_set)) if len(first_set.union(second_set)) != 0 else 0

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
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)


def main():
    test = MinHashLSH(None, None)
    test.shingle_document(
        " One of the most  influential movies of all time, that still holds up extremely well  nearly 50 years later.  Akira Kurosawa's epic tale of heroism and barbarism set the standard in  so many ways it's hard to imagine that any modern film does not show  its influence in some way or other. A great script, great characters,  mostly great acting, splendid cinematography and action sequences that  wrote the book about how these things should be filmed. Even now, after  so many have tried to imitate or beat it, Seven Samurai remains a  totally gripping 3.5 hour experience.  Akira Kurosawa is one of the gods of Cinema - men who seem to have been  born to make films, who have it in their blood.")


if __name__ == '__main__':
    main()
