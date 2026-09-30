from typing import List, Optional
from pydantic import BaseModel, Field

class DocumentMetadata(BaseModel):
    asset_id: str = Field(..., description="Unique ID of the target machine/asset (e.g. MCH-101)")
    catalog_name: str = Field(..., description="Title of the manual or catalog")
    manufacturer: str = Field(default="Generic", description="Manufacturer name")
    model_number: str = Field(default="", description="Model number of the asset")
    page_number: int = Field(default=1, ge=1, description="Page number in the manual")
    section_title: str = Field(default="", description="Section header or title")
    diagram_present: bool = Field(default=False, description="Whether this page contains a visual diagram/figure")
    source_file: str = Field(default="", description="Filename or URI of source document")
    tags: List[str] = Field(default_factory=list, description="Categorization tags (e.g. hydraulic, electrical)")

class ManualChunkIngest(BaseModel):
    chunk_id: str = Field(..., description="Unique identifier for this text chunk")
    text_content: str = Field(..., min_length=5, description="Text content extracted from manual")
    metadata: DocumentMetadata = Field(..., description="Associated document metadata")
    embedding: Optional[List[float]] = Field(default=None, description="Pre-computed vector embedding")

class IngestManualRequest(BaseModel):
    asset_id: str = Field(..., description="Asset identifier")
    catalog_name: str = Field(..., description="Manual title")
    manufacturer: str = Field(default="Generic")
    model_number: str = Field(default="")
    chunks: List[ManualChunkIngest] = Field(default_factory=list, description="Pre-chunked manual content")
    raw_text: Optional[str] = Field(default=None, description="Raw manual text to be auto-chunked")

class IngestManualResponse(BaseModel):
    status: str = "success"
    ingested_chunks_count: int
    collection_name: str
    asset_id: str
    message: str = "Manual chunks ingested into vector store successfully"

class Citation(BaseModel):
    chunk_id: str
    catalog_name: str
    page_number: int
    section_title: str
    snippet: str
    relevance_score: float = 0.0

class ManualQARequest(BaseModel):
    query: str = Field(..., min_length=2, description="User or technician question")
    asset_id: Optional[str] = Field(default=None, description="Filter search by specific asset ID")
    manufacturer: Optional[str] = Field(default=None, description="Filter search by manufacturer")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of relevant chunks to retrieve")
    include_diagrams: bool = Field(default=False, description="Filter for chunks with visual diagrams")

class ManualQAResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    out_of_scope: bool = False
