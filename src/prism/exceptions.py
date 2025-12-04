class PrismError(Exception):
    """
    Base exception for the Prism project.
    All custom exceptions should inherit from this.
    
    Attributes:
        message (str): Human-readable error description.
        original_error (Exception): The underlying exception (if any) that caused this.
    """
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message)
        self.message = message
        self.original_error = original_error

    def __str__(self):
        if self.original_error:
            return f"{self.message} | Caused by: {type(self.original_error).__name__}: {self.original_error}"
        return self.message


# Ingestion Layer (Kafka / Video Source)
class IngestionError(PrismError):
    """Base class for errors occurring during video ingestion or Kafka production."""
    pass

class VideoSourceError(IngestionError):
    """Raised when the video file or webcam cannot be opened or read."""
    pass

class KafkaConnectionError(IngestionError):
    """Raised when the producer cannot connect to the Kafka broker."""
    pass

class KafkaPublishError(IngestionError):
    """Raised when a message fails to publish to a topic."""
    pass


# Inference Layer (Triton / Model)
class InferenceError(PrismError):
    """Base class for errors occurring during the inference stage."""
    pass

class TritonConnectionError(InferenceError):
    """Raised when the client cannot connect to the Triton Inference Server."""
    pass

class ModelNotFoundError(InferenceError):
    """Raised when the requested model is not loaded in Triton."""
    pass

class InferenceComputationError(InferenceError):
    """Raised when Triton returns a failure status (e.g., shape mismatch, runtime error)."""
    pass


# Database Layer (Milvus)
class DatabaseError(PrismError):
    """Base class for errors occurring in the Vector Database."""
    pass

class MilvusConnectionError(DatabaseError):
    """Raised when the client cannot connect to the Milvus server."""
    pass

class CollectionNotFoundError(DatabaseError):
    """Raised when trying to query/insert into a non-existent collection."""
    pass

class VectorInsertionError(DatabaseError):
    """Raised when vector data fails to insert (e.g., dimension mismatch)."""
    pass