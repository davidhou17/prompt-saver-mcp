"""Storage backends for prompt persistence."""

from .base import StorageManager
from .file_storage import FileStorageManager

__all__ = ["StorageManager", "FileStorageManager"]

# Optional MongoDB import
try:
    from .mongodb_storage import MongoDBStorageManager
    __all__.append("MongoDBStorageManager")
except ImportError:
    # MongoDB dependencies not installed
    MongoDBStorageManager = None

