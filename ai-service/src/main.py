import sys
import os

# Ensure src directory is in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from helpers.config import get_settings
from routes import base

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="Microservice for Multimodal Machine Catalog RAG, Vector Search, and Asset Maintenance Q&A.",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Router
app.include_router(base.router, prefix="/api/v1")
app.include_router(base.router)  # Also expose at root level for legacy/health endpoints

@app.get("/")
def root():
    return {
        "message": "Welcome to Smart Asset AI & RAG Service",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "ingest_manual": "POST /ingest/manual",
            "manual_qa": "POST /qa/manual",
            "stats": "GET /stats"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
