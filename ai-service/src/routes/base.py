from fastapi import APIRouter

router = APIRouter(tags=["Base"])

@router.get("/health")
def health_check():
    return {"status": "ok", "service": "AI Service (mini-rag architecture)"}
