import uuid
import logging
from typing import List
from fastapi import APIRouter, HTTPException, status
from models.schema import (
    IngestManualRequest,
    IngestManualResponse,
    ManualQARequest,
    ManualQAResponse,
    ManualChunkIngest,
    DocumentMetadata,
    Citation
)
from stores.vectordb.client import VectorStoreClient
from controllers.embedding_wrapper import chunk_text
from controllers.prompt_templates import format_manual_qa_prompt, OUT_OF_SCOPE_DECLINE_MESSAGE

logger = logging.getLogger("ai_service.routes")
router = APIRouter(tags=["AI RAG Engine"])

# Initialize singleton vector store client
vector_store = VectorStoreClient()

@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    stats = vector_store.get_collection_stats()
    return {
        "status": "ok",
        "service": "Smart Asset AI Service",
        "vector_store": stats
    }

@router.get("/stats", status_code=status.HTTP_200_OK)
def get_vector_stats():
    return vector_store.get_collection_stats()

@router.post("/ingest/manual", response_model=IngestManualResponse, status_code=status.HTTP_201_CREATED)
def ingest_manual(payload: IngestManualRequest):
    """
    Ingests technical manual chunks or raw manual text into the Vector Store.
    """
    try:
        chunks_to_ingest: List[ManualChunkIngest] = []
        
        # Scenario A: Explicit pre-chunked items provided
        if payload.chunks:
            chunks_to_ingest = payload.chunks
            
        # Scenario B: Raw text provided -> Auto-chunking
        elif payload.raw_text:
            raw_chunks = chunk_text(payload.raw_text, chunk_size=400, overlap=50)
            for idx, text_seg in enumerate(raw_chunks, start=1):
                chunk_id = f"chk-{payload.asset_id}-{uuid.uuid4().hex[:8]}"
                metadata = DocumentMetadata(
                    asset_id=payload.asset_id,
                    catalog_name=payload.catalog_name,
                    manufacturer=payload.manufacturer,
                    model_number=payload.model_number,
                    page_number=idx,
                    section_title=f"Section {idx}",
                    diagram_present=False,
                    source_file=payload.catalog_name,
                    tags=["manual", "auto_chunked"]
                )
                chunks_to_ingest.append(
                    ManualChunkIngest(
                        chunk_id=chunk_id,
                        text_content=text_seg,
                        metadata=metadata
                    )
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'chunks' or 'raw_text' must be provided in IngestManualRequest."
            )

        ingested_count = vector_store.upsert_chunks(chunks_to_ingest)
        
        return IngestManualResponse(
            status="success",
            ingested_chunks_count=ingested_count,
            collection_name=vector_store.collection_name,
            asset_id=payload.asset_id,
            message=f"Successfully ingested {ingested_count} chunks for asset '{payload.asset_id}' into Vector Store."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting manual chunks: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest manual: {str(e)}"
        )

@router.post("/qa/manual", response_model=ManualQAResponse, status_code=status.HTTP_200_OK)
def manual_qa(payload: ManualQARequest):
    """
    Performs RAG similarity search and generates structured technical answers with citations.
    """
    try:
        # Check basic guardrail / out of scope filters
        lowered_query = payload.query.lower()
        non_technical_keywords = ["weather", "football", "recipe", "song", "movie", "who won"]
        if any(kw in lowered_query for kw in non_technical_keywords):
            return ManualQAResponse(
                query=payload.query,
                answer=OUT_OF_SCOPE_DECLINE_MESSAGE,
                citations=[],
                confidence_score=0.0,
                out_of_scope=True
            )

        # Vector Store Search
        retrieved_chunks = vector_store.similarity_search(
            query_text=payload.query,
            top_k=payload.top_k,
            asset_id=payload.asset_id,
            manufacturer=payload.manufacturer,
            diagram_present=True if payload.include_diagrams else None
        )

        if not retrieved_chunks:
            return ManualQAResponse(
                query=payload.query,
                answer="No relevant technical manual context found for your query in the Vector Store.",
                citations=[],
                confidence_score=0.0,
                out_of_scope=False
            )

        # Build Citations
        citations: List[Citation] = []
        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            citations.append(
                Citation(
                    chunk_id=chunk.get("chunk_id", ""),
                    catalog_name=meta.get("catalog_name", "Technical Manual"),
                    page_number=meta.get("page_number", 1),
                    section_title=meta.get("section_title", "General"),
                    snippet=chunk.get("text_content", "")[:150] + "...",
                    relevance_score=chunk.get("relevance_score", 0.0)
                )
            )

        # Compute aggregate confidence score
        avg_confidence = sum(c.relevance_score for c in citations) / len(citations) if citations else 0.0

        # Build Prompt Baseline for LLM
        prompt = format_manual_qa_prompt(payload.query, retrieved_chunks)

        # Generate baseline answer using retrieved top chunk
        top_snippet = retrieved_chunks[0].get("text_content", "")
        top_meta = retrieved_chunks[0].get("metadata", {})
        catalog = top_meta.get("catalog_name", "Manual")
        page = top_meta.get("page_number", 1)
        
        constructed_answer = (
            f"Based on the technical manual '{catalog}' (Page {page}):\n\n"
            f"{top_snippet}\n\n"
            f"[Citation: {catalog}, Page {page}]"
        )

        return ManualQAResponse(
            query=payload.query,
            answer=constructed_answer,
            citations=citations,
            confidence_score=round(avg_confidence, 4),
            out_of_scope=False
        )

    except Exception as e:
        logger.error(f"Error processing manual QA: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process manual QA query: {str(e)}"
        )
