import json

from nltk.tokenize import word_tokenize

class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """

        # TODO: note : can use the preprocessor code but there are some complications
        # TODO: i should, change this code
        stopwords = set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
            'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
            'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
            'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
            'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
        ])

        return " ".join([word for word in word_tokenize(query) if word not in stopwords])

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        # TODO : note : i will iterate through the text and check every key word to avoid duplicate texts and support the guide rules
        # TODO : note on implementation : if 1 2 3 4 5  is the string and we want 1 and 5 with 2 near words : 3 will appear two times, so for convenience i return ... ***1*** 2 3 4 ***5*** ...

        final_snippet = "..."
        words, document = self.remove_stop_words_from_query(query.lower()).split(), word_tokenize(doc.lower())
        not_exist_words = words.copy()
        penalty = 0
        for i, w in enumerate(document):
            if penalty > 0:
                penalty -= 1
                continue
            if w not in words: continue

            start, end = i - self.number_of_words_on_each_side, i + self.number_of_words_on_each_side + 1
            if start < 0: start = 0
            if end >= len(document): end = len(document)

            iter = start
            while iter < end:
                if document[iter] in words:
                    end = max(end, iter + self.number_of_words_on_each_side * 2 + 1)
                    if end >= len(document): end = len(document)
                iter += 1
            end -= self.number_of_words_on_each_side
            if end < 0: end = 0
            penalty = end - i

            snippet = document[start:end]
            for i in range(len(snippet)):
                if snippet[i] in words:
                    if snippet[i] in not_exist_words: not_exist_words.remove(snippet[i])
                    snippet[i] = "***" + snippet[i] + "***"


            final_snippet += " ".join(snippet) + " ... "

        return final_snippet, not_exist_words


def main():
    with open("IMDB_movies.json", "r") as f:
        data = json.load(f)

    for i in data:
        if i["id"] == "tt0050083":
            _12_angry_men = i
            break
    text = _12_angry_men["summaries"][2]
    query = "alleged juror father jurors are the random"
    print("text : ", text)
    s = Snippet()
    print("query : ", query, "\n")
    snippet, missing = s.find_snippet(text, query)
    print("missing : ", missing)
    print("snippet : ", snippet)

    # TODO ; note : code is running properly , if confused by some details check the note on find_snippet


if __name__ == '__main__':
    main()
