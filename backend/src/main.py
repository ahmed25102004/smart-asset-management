from fastapi import FastAPI
from routes import base, auth, assets, work_orders

app = FastAPI(title="Smart Asset Core Backend")

app.include_router(base.router)
app.include_router(auth.router)
app.include_router(assets.router)
app.include_router(work_orders.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
