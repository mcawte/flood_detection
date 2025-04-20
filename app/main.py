from fastapi import FastAPI
from .api import endpoints

app = FastAPI(title="Flood Detection API")

app.include_router(endpoints.router, prefix="/api/v1")


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Flood Detection API"}

# Optional: Add startup/shutdown events if needed, e.g., to load models


@app.on_event("startup")
async def startup_event():
    print("Starting up...")


@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down...")
