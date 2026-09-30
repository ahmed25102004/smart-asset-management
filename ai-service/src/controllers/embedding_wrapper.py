import re
import math
import logging
from typing import List, Optional

logger = logging.getLogger("ai_service.embedding")

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer ({e}). Falling back to deterministic embedding encoder.")
                self._model = False
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single string."""
        model = self._get_model()
        if model and model is not False:
            vec = model.encode(text, convert_to_numpy=True).tolist()
            return vec
        
        # Deterministic lightweight fallback embedding (for fast testing / offline execution)
        return self._fallback_embedding(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a list of strings."""
        if not texts:
            return []
        model = self._get_model()
        if model and model is not False:
            vecs = model.encode(texts, convert_to_numpy=True).tolist()
            return vecs
        
        return [self._fallback_embedding(t) for t in texts]

    def _fallback_embedding(self, text: str) -> List[float]:
        """Generates a normalized deterministic vector based on text hash/tokens."""
        vec = [0.0] * self.dimension
        clean_text = text.lower()
        words = re.findall(r'\w+', clean_text)
        
        if not words:
            vec[0] = 1.0
            return vec
            
        for i, word in enumerate(words):
            hash_val = abs(hash(word))
            idx = hash_val % self.dimension
            vec[idx] += 1.0 / (i + 1)

        # L2 Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        else:
            vec[0] = 1.0
        return vec

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Splits raw text into overlapping paragraphs/chunks for RAG ingestion."""
    if not text or len(text.strip()) == 0:
        return []
    
    clean_text = text.strip()
    if len(clean_text) <= chunk_size:
        return [clean_text]
        
    paragraphs = clean_text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = (current_chunk + "\n\n" + para).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            
            # If paragraph itself exceeds chunk_size, split by sentences/words
            if len(para) > chunk_size:
                start = 0
                while start < len(para):
                    end = start + chunk_size
                    chunks.append(para[start:end].strip())
                    start += (chunk_size - overlap)
                current_chunk = ""
            else:
                current_chunk = para
                
    if current_chunk and current_chunk not in chunks:
        chunks.append(current_chunk)
        
    return chunks
