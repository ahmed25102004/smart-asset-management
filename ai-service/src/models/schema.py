from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator

class DocumentMetadata(BaseModel):
    asset_id: str = Field(..., description="Unique ID of the target machine/asset (e.g. MCH-101)")
    catalog_name: str = Field(..., description="Title of the manual or catalog")
    manufacturer: str = Field(default="Generic", description="Manufacturer name")
    model_number: str = Field(default="", description="Model number of the asset")
    page_number: int = Field(default=1, ge=1, description="Page number in the manual")
    section_title: str = Field(default="", description="Section header or title")
    diagram_present: bool = Field(default=False, description="Whether this page contains a visual diagram/figure")
    diagram_caption: Optional[str] = Field(default=None, description="Extracted caption or description of visual diagram")
    image_url: Optional[str] = Field(default=None, description="URI or path to extracted technical diagram image")
    source_file: str = Field(default="", description="Filename or URI of source document")
    tags: List[str] = Field(default_factory=list, description="Categorization tags (e.g. hydraulic, electrical)")

    @field_validator("asset_id")
    @classmethod
    def normalize_asset_id(cls, v: str) -> str:
        return v.strip().upper() if v else v

    @field_validator("catalog_name", "manufacturer")
    @classmethod
    def sanitize_strings(cls, v: str) -> str:
        return v.strip() if v else v

    model_config = {
        "json_schema_extra": {
            "example": {
                "asset_id": "MCH-LATHE-500",
                "catalog_name": "Haas CNC Lathe Maintenance Manual",
                "manufacturer": "Haas Automation",
                "model_number": "ST-20",
                "page_number": 14,
                "section_title": "Hydraulic Pump Troubleshooting",
                "diagram_present": True,
                "diagram_caption": "Figure 4.2: Hydraulic Assembly Diagram",
                "source_file": "haas_st20_manual.pdf",
                "tags": ["hydraulic", "pump", "troubleshooting"]
            }
        }
    }

class ManualChunkIngest(BaseModel):
    chunk_id: str = Field(..., description="Unique identifier for this text chunk")
    text_content: str = Field(..., min_length=5, description="Text content extracted from manual")
    metadata: DocumentMetadata = Field(..., description="Associated document metadata")
    embedding: Optional[List[float]] = Field(default=None, description="Pre-computed vector embedding")

    @field_validator("text_content")
    @classmethod
    def clean_text_content(cls, v: str) -> str:
        return v.strip()

class IngestManualRequest(BaseModel):
    asset_id: str = Field(..., description="Asset identifier")
    catalog_name: str = Field(..., description="Manual title")
    manufacturer: str = Field(default="Generic")
    model_number: str = Field(default="")
    chunks: List[ManualChunkIngest] = Field(default_factory=list, description="Pre-chunked manual content")
    raw_text: Optional[str] = Field(default=None, description="Raw manual text to be auto-chunked")

    @field_validator("asset_id")
    @classmethod
    def normalize_request_asset_id(cls, v: str) -> str:
        return v.strip().upper()

class IngestManualResponse(BaseModel):
    status: str = "success"
    ingested_chunks_count: int
    collection_name: str
    asset_id: str
    message: str = "Manual chunks ingested into vector store successfully"

class BatchIngestManualRequest(BaseModel):
    manuals: List[IngestManualRequest] = Field(..., min_length=1, description="List of manual ingestion requests for bulk processing")

class Citation(BaseModel):
    chunk_id: str
    catalog_name: str
    page_number: int
    section_title: str
    snippet: str
    diagram_present: bool = False
    image_url: Optional[str] = None
    relevance_score: float = 0.0

class ManualQARequest(BaseModel):
    query: str = Field(..., min_length=2, description="User or technician question")
    asset_id: Optional[str] = Field(default=None, description="Filter search by specific asset ID")
    manufacturer: Optional[str] = Field(default=None, description="Filter search by manufacturer")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of relevant chunks to retrieve")
    include_diagrams: bool = Field(default=False, description="Filter for chunks with visual diagrams")

    @field_validator("query")
    @classmethod
    def clean_query(cls, v: str) -> str:
        return v.strip()

class ManualQAResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    out_of_scope: bool = False
    retrieval_time_ms: float = Field(default=0.0, description="Query execution & vector retrieval time in milliseconds")
