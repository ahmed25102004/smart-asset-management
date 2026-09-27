from fastapi import FastAPI
from routes import base, rag, data
from helpers.config import get_settings

app = FastAPI(title="Smart Asset AI Service (mini-rag architecture)")

app.include_router(base.router)
app.include_router(rag.router)
app.include_router(data.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
