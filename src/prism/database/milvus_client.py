from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
from src.prism.exceptions import (
    MilvusConnectionError,
    VectorInsertionError,
    DatabaseError,
)

"""
Milvus Client for Vector Database Operations

Usage: helps in connecting to Milvus, creating collections, inserting vectors, and searching.
"""


class MilvusClient:
    def __init__(
        self, collection_name="PrismCollection", host="localhost", port="19530"
    ):
        self.collection_name = collection_name
        self.dim = 8400

        try:
            connections.connect(host=host, port=port)
        except Exception as e:
            raise MilvusConnectionError(
                f"Failed to connect to Milvus at {host}:{port}", original_error=e
            )
        # Initialize or create collection
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        """Create the collection if it doesn't exist."""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="timestamp", dtype=DataType.DOUBLE),
            FieldSchema(name="frame_id", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
        ]
        schema = CollectionSchema(fields, description="Video analytics storage")
        self.collection = Collection(self.collection_name, schema)

        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }

        self.collection.create_index(field_name="embedding", index_params=index_params)
        self.collection.load()

    def insert(self, frame_id: int, timestamp: float, vector: list):
        """Insert a vector along with its metadata into the collection."""
        data = [
            [timestamp],  # timestamp field
            [frame_id],  # frame_id field
            [vector],  # embedding field
        ]
        try:
            self.collection.insert(data)
        except Exception as e:
            raise VectorInsertionError(
                f"Failed to insert frame {frame_id}", original_error=e
            )

    def search(self, query_vector, top_k=5):
        """Search for similar vectors in the collection."""
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        try:
            results = self.collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["timestamp", "frame_id"],
            )
            return results
        except Exception as e:
            raise DatabaseError("Search failed", original_error=e)
