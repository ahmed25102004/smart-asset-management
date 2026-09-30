import os
import logging
from typing import List, Dict, Any, Optional
from helpers.config import get_settings
from models.schema import ManualChunkIngest, DocumentMetadata
from controllers.embedding_wrapper import EmbeddingGenerator

logger = logging.getLogger("ai_service.vectordb")

class VectorStoreClient:
    def __init__(self, collection_name: Optional[str] = None):
        self.settings = get_settings()
        self.collection_name = collection_name or self.settings.COLLECTION_NAME
        self.db_type = self.settings.VECTOR_DB_TYPE.lower()
        self.embedding_gen = EmbeddingGenerator(model_name=self.settings.EMBEDDING_MODEL_NAME)
        
        self.chroma_client = None
        self.collection = None
        self.qdrant_client = None
        
        self._init_db()

    def _init_db(self):
        if self.db_type == "chroma":
            try:
                import chromadb
                persist_dir = os.path.abspath(self.settings.CHROMA_PERSIST_DIR)
                os.makedirs(persist_dir, exist_ok=True)
                
                self.chroma_client = chromadb.PersistentClient(path=persist_dir)
                self.collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Initialized ChromaDB persistent client at '{persist_dir}' with collection '{self.collection_name}'")
            except Exception as e:
                logger.warning(f"ChromaDB persistent init failed ({e}). Initializing in-memory ChromaDB fallback client.")
                import chromadb
                self.chroma_client = chromadb.Client()
                self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
        elif self.db_type == "qdrant":
            try:
                from qdrant_client import QdrantClient
                self.qdrant_client = QdrantClient(url=self.settings.VECTOR_DB_URL)
                logger.info(f"Initialized Qdrant client at '{self.settings.VECTOR_DB_URL}'")
            except Exception as e:
                logger.error(f"Failed to initialize Qdrant client ({e}). Falling back to ChromaDB.")
                self.db_type = "chroma"
                self._init_db()

    def upsert_chunks(self, chunks: List[ManualChunkIngest]) -> int:
        """Upserts a list of manual chunks into the vector store."""
        if not chunks:
            return 0
            
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        # Batch generate embeddings if missing
        texts_needing_embeddings = [c.text_content for c in chunks if not c.embedding]
        computed_embeddings = self.embedding_gen.embed_batch(texts_needing_embeddings) if texts_needing_embeddings else []
        
        embed_idx = 0
        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text_content)
            
            # Serialize metadata to dict for ChromaDB storage
            meta_dict = chunk.metadata.model_dump()
            # Convert list tags to comma-separated string for vector DB compatibility
            if isinstance(meta_dict.get("tags"), list):
                meta_dict["tags"] = ",".join(meta_dict["tags"])
            metadatas.append(meta_dict)
            
            if chunk.embedding:
                embeddings.append(chunk.embedding)
            else:
                embeddings.append(computed_embeddings[embed_idx])
                embed_idx += 1

        if self.db_type == "chroma" and self.collection:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            return len(ids)
        
        return 0

    def similarity_search(
        self,
        query_text: str,
        top_k: int = 3,
        asset_id: Optional[str] = None,
        manufacturer: Optional[str] = None,
        diagram_present: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Performs vector similarity search with metadata filtering."""
        query_vector = self.embedding_gen.embed_text(query_text)
        
        # Construct metadata filters
        where_filter = {}
        conditions = []
        if asset_id:
            conditions.append({"asset_id": asset_id})
        if manufacturer:
            conditions.append({"manufacturer": manufacturer})
        if diagram_present is not None:
            conditions.append({"diagram_present": diagram_present})
            
        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        results = []
        if self.db_type == "chroma" and self.collection:
            count = self.collection.count()
            if count == 0:
                return []
                
            query_kwargs = {
                "query_embeddings": [query_vector],
                "n_results": min(top_k, count)
            }
            if where_filter:
                query_kwargs["where"] = where_filter

            try:
                chroma_res = self.collection.query(**query_kwargs)
            except Exception as e:
                logger.warning(f"Filter query failed ({e}), falling back to unfiltered query.")
                query_kwargs.pop("where", None)
                chroma_res = self.collection.query(**query_kwargs)

            if chroma_res and chroma_res.get("ids") and chroma_res["ids"][0]:
                for i in range(len(chroma_res["ids"][0])):
                    chunk_id = chroma_res["ids"][0][i]
                    doc_text = chroma_res["documents"][0][i] if chroma_res.get("documents") else ""
                    raw_meta = chroma_res["metadatas"][0][i] if chroma_res.get("metadatas") else {}
                    distance = chroma_res["distances"][0][i] if chroma_res.get("distances") else 0.0
                    
                    # Convert distance to similarity score (cosine distance in Chroma is 1 - sim)
                    relevance_score = max(0.0, min(1.0, 1.0 - distance))

                    # Parse tags back into list
                    if "tags" in raw_meta and isinstance(raw_meta["tags"], str):
                        raw_meta["tags"] = [t.strip() for t in raw_meta["tags"].split(",") if t.strip()]

                    results.append({
                        "chunk_id": chunk_id,
                        "text_content": doc_text,
                        "metadata": raw_meta,
                        "relevance_score": round(relevance_score, 4)
                    })

        return results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Returns collection item count and details."""
        if self.db_type == "chroma" and self.collection:
            return {
                "db_type": "chroma",
                "collection_name": self.collection_name,
                "count": self.collection.count()
            }
        return {"db_type": self.db_type, "collection_name": self.collection_name, "count": 0}

    def reset_collection(self) -> None:
        """Deletes and recreates the collection."""
        if self.db_type == "chroma" and self.chroma_client:
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
            except Exception:
                pass
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
