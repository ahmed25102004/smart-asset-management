import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add src to sys.path for test imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from main import app
from models.schema import DocumentMetadata, ManualChunkIngest, IngestManualRequest, ManualQARequest
from controllers.embedding_wrapper import EmbeddingGenerator, chunk_text
from stores.vectordb.client import VectorStoreClient
from controllers.prompt_templates import format_manual_qa_prompt

client = TestClient(app)

def test_pydantic_schemas():
    """Verify document metadata and chunk Pydantic models."""
    meta = DocumentMetadata(
        asset_id="MCH-101",
        catalog_name="CNC Hydraulic Lathe Manual",
        manufacturer="Haas Automation",
        model_number="ST-20",
        page_number=14,
        section_title="Hydraulic Pump Troubleshooting",
        diagram_present=True,
        tags=["hydraulic", "pump"]
    )
    assert meta.asset_id == "MCH-101"
    assert meta.diagram_present is True
    assert len(meta.tags) == 2

    chunk = ManualChunkIngest(
        chunk_id="chk-101",
        text_content="Check hydraulic oil level in reservoir before starting motor.",
        metadata=meta
    )
    assert chunk.chunk_id == "chk-101"
    assert "hydraulic" in chunk.text_content

def test_embedding_generator_and_chunking():
    """Verify embedding generator output format and text chunking utility."""
    generator = EmbeddingGenerator(dimension=384)
    vec = generator.embed_text("Test hydraulic maintenance procedure")
    assert isinstance(vec, list)
    assert len(vec) == 384
    assert all(isinstance(v, float) for v in vec)

    batch_vecs = generator.embed_batch(["Text 1", "Text 2"])
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 384

    # Test text chunking
    sample_text = "Paragraph 1: Check power supply.\n\nParagraph 2: Calibrate temperature sensor.\n\nParagraph 3: Inspect belt tension."
    chunks = chunk_text(sample_text, chunk_size=100, overlap=10)
    assert len(chunks) >= 1
    assert "Check power supply" in chunks[0]

def test_vector_store_client_operations():
    """Verify VectorStoreClient upsert and similarity search."""
    vec_client = VectorStoreClient(collection_name="test_manuals_col")
    vec_client.reset_collection()

    meta = DocumentMetadata(
        asset_id="MCH-TEST",
        catalog_name="Test Industrial Manual",
        manufacturer="TestCorp",
        page_number=5,
        section_title="Safety Rules"
    )
    chunk = ManualChunkIngest(
        chunk_id="test-chk-001",
        text_content="Always wear protective eye goggles when operating pneumatic press.",
        metadata=meta
    )

    upserted_count = vec_client.upsert_chunks([chunk])
    assert upserted_count == 1

    stats = vec_client.get_collection_stats()
    assert stats["count"] >= 1

    results = vec_client.similarity_search(query_text="eye goggles safety", top_k=1, asset_id="MCH-TEST")
    assert len(results) == 1
    assert results[0]["chunk_id"] == "test-chk-001"
    assert "eye goggles" in results[0]["text_content"]

def test_health_and_stats_endpoints():
    """Verify GET /health and GET /stats API endpoints."""
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"

    res_stats = client.get("/stats")
    assert res_stats.status_code == 200
    assert "collection_name" in res_stats.json()

def test_ingest_and_qa_endpoints():
    """Verify POST /ingest/manual and POST /qa/manual RAG API endpoints."""
    # 1. Ingest via raw_text
    ingest_payload = {
        "asset_id": "MCH-202",
        "catalog_name": "Centrifugal Pump Operating Guide",
        "manufacturer": "Grundfos",
        "model_number": "CR-32",
        "raw_text": "Step 1: Verify suction valve is fully open before starting Grundfos pump motor.\n\nStep 2: Monitor discharge pressure gauge for stable 5.5 bar reading.\n\nStep 3: If pressure fluctuates wildly, bleed air from casing screw."
    }
    ingest_res = client.post("/ingest/manual", json=ingest_payload)
    assert ingest_res.status_code == 201
    assert ingest_res.json()["status"] == "success"
    assert ingest_res.json()["ingested_chunks_count"] >= 1

    # 2. Q&A query
    qa_payload = {
        "query": "How to handle pressure fluctuation in Grundfos pump?",
        "asset_id": "MCH-202",
        "top_k": 2
    }
    qa_res = client.post("/qa/manual", json=qa_payload)
    assert qa_res.status_code == 200
    data = qa_res.json()
    assert data["out_of_scope"] is False
    assert len(data["citations"]) >= 1
    assert data["citations"][0]["catalog_name"] == "Centrifugal Pump Operating Guide"
    assert "bleed air" in data["answer"] or "pressure" in data["answer"]

def test_out_of_scope_query():
    """Verify non-technical query guardrail."""
    qa_payload = {
        "query": "Who won the football world cup match?",
        "top_k": 2
    }
    res = client.post("/qa/manual", json=qa_payload)
    assert res.status_code == 200
    assert res.json()["out_of_scope"] is True
    assert "outside the scope" in res.json()["answer"]
