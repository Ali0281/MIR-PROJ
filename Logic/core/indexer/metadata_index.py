from Logic.core.indexer.indexes_enum import Indexes, Index_types
import json


class Metadata_index:
    def __init__(self, path='C:/Users/Ali/PycharmProjects/MIR-PROJ/Logic/core/indexer/index/documents_index.json'):
        """
        Initializes the Metadata_index.

        Parameters
        ----------
        path : str
            The path to the indexes.
        """
        # TODO
        self.path = path
        self.documents = self.read_documents()
        self.metadata_index = None

    def read_documents(self):
        """
        Reads the documents.
        
        """
        # TODO
        with open(self.path, "r") as f:
            return json.load(f)

    def create_metadata_index(self):
        """
        Creates the metadata index.
        """
        metadata_index = {}
        metadata_index['averge_document_length'] = {
            'stars': self.get_average_document_field_length('stars'),
            'genres': self.get_average_document_field_length('genres'),
            'summaries': self.get_average_document_field_length('summaries')
        }
        metadata_index['document_count'] = len(self.documents)
        self.metadata_index = metadata_index

    def get_average_document_field_length(self, where):
        """
        Returns the sum of the field lengths of all documents in the index.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.
        """
        # TODO
        if where not in ["stars", "genres", "summaries"]: return -1
        if where == "summaries":
            summary_count, summary_word_count = 0, 0
            for doc_id, doc in self.documents.items():
                for summary in doc["summaries"]:
                    summary_word_count += len(summary.split())
                summary_count += len(doc["summaries"])
            return summary_word_count / summary_count
        else:
            sum_ = 0
            for doc_id, doc in self.documents.items():
                sum_ += len(doc[where])
            return sum_ / len(self.documents)

    def store_metadata_index(self, path):
        """
        Stores the metadata index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        """
        path = path + Indexes.DOCUMENTS.value + '_' + Index_types.METADATA.value + '_index.json'
        with open(path, 'w') as file:
            json.dump(self.metadata_index, file, indent=4)


def main():
    meta_index = Metadata_index()
    meta_index.create_metadata_index()
    meta_index.store_metadata_index("index/")


if __name__ == "__main__":
    main()
