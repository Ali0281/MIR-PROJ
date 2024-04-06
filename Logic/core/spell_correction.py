import heapq
import json
import re

from Logic.core.preprocess import Preprocessor


class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        # TODO : note : clear the text from : ; and such
        for index, document in enumerate(all_documents):
            all_documents[index] = re.sub(r'\b[^a-zA-Z0-9\']+\b', ' ', document)
        # TODO : note : can get rid of lowers or add or ...
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)
        # TODO : note : will add stopwords to skip them as we had the data preprocessed
        self.stopwords = []
        with open("stopwords.txt", 'r') as f:
            for word in f:
                self.stopwords.append(word.strip().lower())

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        # TODO :
        shingles = set()
        try:
            for i in range(len(word) + 1 - k):
                shingles.add(word[i: i + k])
        except Exception as e:
            shingles.add(word)
            print(f"warning : cant shingle words for spell correction ,error : {e}")
        finally:
            return shingles

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """

        # TODO: Calculate jaccard score here.
        if len(first_set.union(second_set)) == 0: return 0
        return len(first_set.intersection(second_set)) / len(first_set.union(second_set))

    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        # TODO :
        all_shingled_words = dict()
        word_counter = dict()

        for doc in all_documents:
            for w in doc.lower().split():
                # can strip or ... word
                if w not in word_counter:
                    word_counter[w] = 1
                else:
                    word_counter[w] += 1
                if w not in all_shingled_words: all_shingled_words[w] = self.shingle_word(w)
        return all_shingled_words, word_counter

    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : stf
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        # TODO :
        top5_candidates = list()
        # can do more to make word more likely
        word = word.lower()
        self.all_shingled_words[word] = self.all_shingled_words.get(word, self.shingle_word(word))
        shingles = self.all_shingled_words[word]
        for key, value in self.all_shingled_words.items():
            heapq.heappush(top5_candidates, (self.jaccard_score(shingles, value), key, value))
            # TODO : note : giving out 6 candidates one probably being the same word
            if len(top5_candidates) > 6: heapq.heappop(top5_candidates)
        #print(top5_candidates)
        return [word for score, word, shingles in sorted(top5_candidates)][::-1]

    def word_spell_checker(self, word):
        self.all_shingled_words[word] = self.all_shingled_words.get(word, self.shingle_word(word))

        nearest_words = self.find_nearest_words(word)
        nearest_words.append(word)
        scores = []
        tf = [self.word_counter.get(n, 0) for n in nearest_words]
        normalized_tf = [x / max(tf) for x in tf]
        for index, n in enumerate(nearest_words):
            j_score = self.jaccard_score(self.all_shingled_words[word], self.all_shingled_words[n])
            scores.append((n, j_score * normalized_tf[index]))
            #print((n, j_score * normalized_tf[index]))

        if len(scores) > 0:
            return max(scores, key=lambda x: x[1])[0]
        else:
            return word

    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : stf
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        # TODO
        query = query.lower()
        estimated_query = ""
        for w in query.split():
            if w not in self.stopwords:
                estimated_query += self.word_spell_checker(w) + " "
            else:
                estimated_query += w + " "

        return estimated_query.strip()


def main():
    with open("IMDB_movies.json", "r") as f:
        data = json.load(f)

    # TODO : note : in case you needed a pre proccessed input, but it seems to work poorly as the pre process proceeds to remove stopwords and such
    #pre = Preprocessor(data, "stopwords.txt")
    #pre.preprocess()
    #data = pre.documents

    documents_as_string = []
    # TODO : note : so we are dealing with a list of strings as input of the class
    for doc in data:
        # TODO : note : i will just use the summaries / reviews / synopsis as they have the most text available
        documents_as_string.append(
            " ".join(doc["summaries"]) + " ".join([review[0] for review in doc["reviews"]]) + " ".join(doc["synopsis"]))
    s = SpellCorrection(documents_as_string)
    print(s.spell_check("chec thi mis spel"))


if __name__ == '__main__':
    main()
