"""MongoDB storage backend for prompts.

Requires: motor, pymongo
Install with: pip install motor pymongo
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..models import PromptTemplate, PromptSearchResult
from .base import StorageManager

# Try to import MongoDB dependencies
try:
    import motor.motor_asyncio
    from bson import ObjectId
    from pymongo import IndexModel, TEXT
    from pymongo.errors import DuplicateKeyError
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    motor = None
    ObjectId = None
    IndexModel = None
    TEXT = None
    DuplicateKeyError = None


class MongoDBStorageManager(StorageManager):
    """MongoDB-based storage for prompts.
    
    Supports text search and vector similarity search (with MongoDB Atlas).
    """

    def __init__(
        self,
        mongodb_uri: str,
        database_name: str = "prompt_saver",
        collection_name: str = "prompts",
        vector_index_name: str = "vector_index"
    ):
        """Initialize MongoDB storage.
        
        Args:
            mongodb_uri: MongoDB connection URI
            database_name: Name of the database
            collection_name: Name of the collection
            vector_index_name: Name of the vector search index (for Atlas)
        """
        if not MONGODB_AVAILABLE:
            raise ImportError(
                "MongoDB dependencies not installed. "
                "Install with: pip install motor pymongo"
            )
        
        self.mongodb_uri = mongodb_uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.vector_index_name = vector_index_name
        self.client = None
        self.database = None
        self.collection = None

    async def connect(self) -> None:
        """Connect to MongoDB and set up indexes."""
        self.client = motor.motor_asyncio.AsyncIOMotorClient(self.mongodb_uri)
        self.database = self.client[self.database_name]
        self.collection = self.database[self.collection_name]
        await self.client.admin.command('ping')
        await self._setup_indexes()

    async def disconnect(self) -> None:
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()

    async def _setup_indexes(self) -> None:
        """Set up database indexes."""
        indexes = [
            IndexModel([("use_case", 1)]),
            IndexModel([("last_updated", -1)]),
            IndexModel([("summary", TEXT), ("prompt_template", TEXT)]),
        ]
        try:
            await self.collection.create_indexes(indexes)
        except DuplicateKeyError:
            pass  # Indexes already exist

    async def save_prompt(self, prompt: PromptTemplate) -> str:
        """Save a new prompt to MongoDB."""
        if self.collection is None:
            raise RuntimeError("Database not connected")
        
        prompt_dict = prompt.model_dump()
        result = await self.collection.insert_one(prompt_dict)
        return str(result.inserted_id)

    async def get_prompt_by_id(self, prompt_id: str) -> Optional[PromptSearchResult]:
        """Get a specific prompt by its ObjectId."""
        if self.collection is None:
            raise RuntimeError("Database not connected")

        try:
            doc = await self.collection.find_one({"_id": ObjectId(prompt_id)})
            if doc:
                return PromptSearchResult(
                    id=str(doc["_id"]),
                    use_case=doc["use_case"],
                    summary=doc["summary"],
                    prompt_template=doc["prompt_template"],
                    history=doc["history"],
                    last_updated=doc["last_updated"],
                    num_updates=doc["num_updates"]
                )
        except Exception:
            pass

        return None

    async def search_prompts_by_text(self, query: str, limit: int = 3) -> List[PromptSearchResult]:
        """Search prompts using MongoDB text search."""
        if self.collection is None:
            raise RuntimeError("Database not connected")

        cursor = self.collection.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(limit)

        results = []
        async for doc in cursor:
            result = PromptSearchResult(
                id=str(doc["_id"]),
                use_case=doc["use_case"],
                summary=doc["summary"],
                prompt_template=doc["prompt_template"],
                history=doc["history"],
                last_updated=doc["last_updated"],
                num_updates=doc["num_updates"],
                score=doc.get("score")
            )
            results.append(result)

        return results

    async def search_prompts_by_vector(self, embedding: List[float], limit: int = 3) -> List[PromptSearchResult]:
        """Search prompts using MongoDB Atlas vector similarity search."""
        if self.collection is None:
            raise RuntimeError("Database not connected")

        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.vector_index_name,
                        "queryVector": embedding,
                        "path": "embedding",
                        "exact": True,
                        "limit": limit
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "use_case": 1,
                        "summary": 1,
                        "prompt_template": 1,
                        "history": 1,
                        "last_updated": 1,
                        "num_updates": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]

            cursor = self.collection.aggregate(pipeline)
            results = []

            async for doc in cursor:
                result = PromptSearchResult(
                    id=str(doc["_id"]),
                    use_case=doc["use_case"],
                    summary=doc["summary"],
                    prompt_template=doc["prompt_template"],
                    history=doc["history"],
                    last_updated=doc["last_updated"],
                    num_updates=doc["num_updates"],
                    score=doc.get("score")
                )
                results.append(result)

            return results

        except Exception:
            return []

    async def search_prompts_by_use_case(self, use_case: str, limit: int = 5) -> List[PromptSearchResult]:
        """Search prompts by use case category."""
        if self.collection is None:
            raise RuntimeError("Database not connected")

        try:
            cursor = self.collection.find(
                {"use_case": use_case}
            ).sort([("last_updated", -1)]).limit(limit)

            results = []
            async for doc in cursor:
                result = PromptSearchResult(
                    id=str(doc["_id"]),
                    use_case=doc["use_case"],
                    summary=doc["summary"],
                    prompt_template=doc["prompt_template"],
                    history=doc["history"],
                    last_updated=doc["last_updated"],
                    num_updates=doc["num_updates"]
                )
                results.append(result)

            return results

        except Exception:
            return []

    async def update_prompt(self, prompt_id: str, updates: Dict[str, Any], change_description: str) -> bool:
        """Update an existing prompt."""
        if self.collection is None:
            raise RuntimeError("Database not connected")

        try:
            now = datetime.now(timezone.utc)
            updates["last_updated"] = now
            updates["$inc"] = {"num_updates": 1}
            updates["$push"] = {"changelog": f"{now.isoformat()}: {change_description}"}

            result = await self.collection.update_one(
                {"_id": ObjectId(prompt_id)},
                {
                    "$set": {k: v for k, v in updates.items() if not k.startswith("$")},
                    "$inc": updates.get("$inc", {}),
                    "$push": updates.get("$push", {})
                }
            )

            return result.modified_count > 0

        except Exception:
            return False

