import json

import random

from Logic.core.link_analysis.graph import LinkGraph
from Logic.core.indexer.indexes_enum import Indexes
from Logic.core.indexer.index_reader import Index_reader


class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        # TODO : note : we needed to change some parts, the expand part will be two levels and the hub and auth will be devided !

        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = dict()  # TODO : note : for ease
        self.authorities = dict()
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        for movie in self.root_set:
            # TODO
            movie_title, movie_id, movie_stars = movie["title"], movie["id"], movie["stars"]
            movie_exists = self.graph.exists(movie_id)
            if not movie_exists: self.graph.add_node(movie_id)
            for star in movie_stars:
                star_exists = self.graph.exists(star)
                if not star_exists: self.graph.add_node(star)

                self.graph.add_edge(star, movie_id)

                if not movie_exists: self.authorities[movie_id] = 1
                if not star_exists: self.hubs[star] = 1

    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        all_stars = set(self.graph.get_stars())
        for movie in corpus:
            # TODO

            movie_id, movie_stars = movie["id"], movie["stars"]
            if self.graph.exists(movie_id): continue

            add = False

            if len(set(movie_stars).intersection(all_stars)): add = True

            for star in movie_stars:
                if add: break
                if self.graph.exists(star): add = True

            if add:
                self.graph.add_node(movie_id)
                self.authorities[movie_id] = 1

                for star in movie_stars:
                    if not self.graph.exists(star):
                        self.graph.add_node(star)
                        self.hubs[star] = 1

                    self.graph.add_edge(star, movie_id)

    def hits(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """
        a_s = []
        h_s = []
        # TODO
        for i in range(num_iteration):
            for movie in self.authorities:
                new_authority = 0
                for hub in self.graph.get_predecessors(movie):
                    new_authority += self.hubs[hub]
                self.authorities[movie] = new_authority

            for star in self.hubs:
                new_hub = 0
                for authority in self.graph.get_successors(star):
                    new_hub += self.authorities[authority]
                self.hubs[star] = new_hub

        sorted_authorities = dict(sorted(self.authorities.items(), key=lambda x: x[1], reverse=True))
        sorted_hubs = dict(sorted(self.hubs.items(), key=lambda x: x[1], reverse=True))

        if max_result is None:
            a_s = list(sorted_authorities.keys())
            h_s = list(sorted_hubs.keys())
        else:
            a_max = min(len(sorted_authorities), max_result)
            h_max = min(len(sorted_hubs), max_result)

            a_s_ = list(sorted_authorities.keys())[:a_max]
            h_s = list(sorted_hubs.keys())[:h_max]

        # TODO : note: need to convert the id to name
        #documents_index = Index_reader("", index_name=Indexes.DOCUMENTS).index

        #for i in a_s_:
            #a_s.append(documents_index[i]["title"])

        return h_s, a_s_


if __name__ == "__main__":
    # You can use this section to run and test the results of your link analyzer
    with open("../preprocess.json", "r") as f:
        data = json.load(f)

    corpus = []
    for document in data:
        corpus.append({"id": document["id"], "title": document["title"], "stars": document["stars"]})

    corpus = corpus
    # root_set = []  # TODO: it shoud be a subset of your corpus
    root_set = list(filter(lambda x: random.randrange(20), corpus))

    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=5, num_iteration=500)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')