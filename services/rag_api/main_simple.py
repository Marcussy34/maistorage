#!/usr/bin/env python3
"""
Simplified main.py for debugging Render deployment
"""
import os
import sys
from fastapi import FastAPI
import uvicorn

print("🚀 STARTING SIMPLIFIED MAISTORAGE API")
print(f"🐍 Python: {sys.version}")
print(f"📁 Working dir: {os.getcwd()}")
print(f"🔧 PORT: {os.environ.get('PORT', 'NOT_SET')}")

# Create minimal app
app = FastAPI(title="MAI Storage RAG API (Simple)")

@app.get("/")
def read_root():
    return {"message": "MAI Storage RAG API is running (simplified version)"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "Simplified API is working",
        "port": os.environ.get("PORT", "unknown")
    }

@app.get("/test")
def test_endpoint():
    return {
        "message": "Test endpoint working",
        "environment": os.environ.get("ENVIRONMENT", "unknown"),
        "qdrant_url": os.environ.get("QDRANT_URL", "not_set")[:30] + "..." if os.environ.get("QDRANT_URL") else "not_set"
    }

if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 8000))
        print(f"🚀 Starting simplified server on port {port}")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
    except Exception as e:
        print(f"💥 Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
