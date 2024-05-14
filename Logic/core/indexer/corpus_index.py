import json

from Logic.core.indexer.index_reader import Index_reader
from Logic.core.indexer.indexes_enum import Indexes, Index_types


class Corpus_index:
    def __init__(self, path="index/"):
        self.index = {
            Indexes.STARS: Index_reader(path, index_name=Indexes.STARS).index,
            Indexes.GENRES: Index_reader(path, index_name=Indexes.GENRES).index,
            Indexes.SUMMARIES: Index_reader(path, index_name=Indexes.SUMMARIES).index,
        }

        self.corpus_index = {
            Indexes.STARS: self.convert_to_corpus_index(Indexes.STARS),
            Indexes.SUMMARIES: self.convert_to_corpus_index(Indexes.SUMMARIES),
            Indexes.GENRES: self.convert_to_corpus_index(Indexes.GENRES)
        }

        self.store_corpus_index(path, Indexes.STARS)
        self.store_corpus_index(path, Indexes.SUMMARIES)
        self.store_corpus_index(path, Indexes.GENRES)

    def convert_to_corpus_index(self, index_name):

        if index_name not in self.index:
            raise ValueError("Invalid index type")

        result = {}
        current_index = self.index[index_name]
        for k, v in current_index.items():
            for doc_id, df in v.items():
                result[k] = result.get(k, 0) + df
        return [result, sum(result.values())]

    def store_corpus_index(self, path, index_name):
        path = path + index_name.value + "_corpus_index.json"
        with open(path, "w") as file:
            json.dump(self.corpus_index[index_name], file, indent=4)


if __name__ == "__main__":
    corpus = Corpus_index(path="index/")
