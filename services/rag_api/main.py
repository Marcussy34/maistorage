"""
FastAPI main application for MAI Storage RAG API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
import uvicorn


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    app_name: str = "MAI Storage RAG API"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    
    class Config:
        env_file = ".env"


settings = Settings()

app = FastAPI(
    title=settings.app_name,
    description="Agentic RAG API with Next.js frontend",
    version="0.1.0",
)

# CORS configuration for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    """Health check endpoint."""
    return {
        "message": "MAI Storage RAG API is running",
        "app_name": settings.app_name,
        "model": settings.openai_model,
        "embedding_model": settings.embedding_model
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "service": "rag_api",
        "version": "0.1.0"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
