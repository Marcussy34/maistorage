#!/usr/bin/env python3
"""
Simplified FastAPI main for Google Cloud Run deployment.
This version starts quickly and initializes services lazily.
"""
import os
import sys
import traceback
import time
from datetime import datetime
from typing import Optional
from pathlib import Path

# Load environment variables - but don't fail if .env is missing
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)
except:
    print("Note: .env file not loaded (this is normal in Cloud Run)")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Simple health check model
class HealthCheck(BaseModel):
    status: str
    timestamp: str
    version: str = "1.0.0-cloudrun"
    message: str = "Cloud Run deployment successful"

# Create FastAPI app immediately (no complex initialization)
app = FastAPI(
    title="MAI Storage RAG API (Cloud Run)",
    description="Fast-starting Cloud Run deployment",
    version="1.0.0-cloudrun"
)

# Simple CORS setup for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track startup time
app_start_time = time.time()

# Global state for lazy initialization
_services_initialized = False
_initialization_error = None

def initialize_services():
    """Initialize services lazily when first needed."""
    global _services_initialized, _initialization_error
    
    if _services_initialized:
        return True
    
    try:
        print("🚀 Starting lazy service initialization...")
        
        # TODO: Add lazy initialization of heavy services here
        # For now, just verify environment variables
        required_vars = ["OPENAI_API_KEY", "QDRANT_URL"]
        for var in required_vars:
            if not os.environ.get(var):
                raise ValueError(f"Missing required environment variable: {var}")
        
        print("✅ Services initialized successfully")
        _services_initialized = True
        return True
        
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        _initialization_error = str(e)
        return False

@app.get("/", response_model=dict)
def read_root():
    """Root endpoint - starts immediately."""
    return {
        "message": "MAI Storage RAG API is running (Cloud Run)",
        "status": "healthy",
        "version": "1.0.0-cloudrun",
        "uptime_seconds": time.time() - app_start_time,
        "timestamp": datetime.utcnow().isoformat(),
        "environment": {
            "port": os.environ.get("PORT", "unknown"),
            "environment": os.environ.get("ENVIRONMENT", "production"),
            "has_openai_key": bool(os.environ.get("OPENAI_API_KEY")),
            "has_qdrant_url": bool(os.environ.get("QDRANT_URL"))
        }
    }

@app.get("/health", response_model=HealthCheck)
def health_check():
    """Health check endpoint - required for Cloud Run."""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        message=f"Server running for {time.time() - app_start_time:.1f} seconds"
    )

@app.get("/ready")
def readiness_check():
    """Readiness check - initializes services if needed."""
    if initialize_services():
        return {
            "status": "ready",
            "services_initialized": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "services_initialized": False,
                "error": _initialization_error,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/test")
def test_endpoint():
    """Test endpoint to verify environment variables."""
    return {
        "message": "Test endpoint working",
        "environment": os.environ.get("ENVIRONMENT", "unknown"),
        "port": os.environ.get("PORT", "unknown"),
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY")),
        "qdrant_configured": bool(os.environ.get("QDRANT_URL")),
        "qdrant_url_preview": os.environ.get("QDRANT_URL", "not_set")[:30] + "..." if os.environ.get("QDRANT_URL") else "not_set",
        "timestamp": datetime.utcnow().isoformat()
    }

# Future endpoints will be added here with lazy service initialization

if __name__ == "__main__":
    try:
        print("=" * 50)
        print("🚀 STARTING MAISTORAGE RAG API (CLOUD RUN)")
        print("=" * 50)
        
        # Get port from environment variable (Cloud Run sets this automatically)
        port = int(os.environ.get("PORT", 8080))
        print(f"🚀 Starting FastAPI app on port {port}")
        print(f"📍 Host: 0.0.0.0:{port}")
        print(f"🔧 Environment PORT variable: {os.environ.get('PORT', 'NOT SET')}")
        print(f"🐍 Python version: {sys.version}")
        print(f"📁 Working directory: {os.getcwd()}")
        
        # Print key environment variables (masked)
        print("🔧 Key environment variables:")
        env_vars = {
            "ENVIRONMENT": os.environ.get("ENVIRONMENT", "NOT SET"),
            "DEBUG": os.environ.get("DEBUG", "NOT SET"),
            "LOG_LEVEL": os.environ.get("LOG_LEVEL", "NOT SET"),
            "OPENAI_API_KEY": "SET" if os.environ.get("OPENAI_API_KEY") else "NOT SET",
            "QDRANT_URL": "SET" if os.environ.get("QDRANT_URL") else "NOT SET",
            "QDRANT_API_KEY": "SET" if os.environ.get("QDRANT_API_KEY") else "NOT SET"
        }
        
        for key, value in env_vars.items():
            print(f"   {key}: {value}")
        
        print("🚀 Starting uvicorn server...")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        print("=" * 50)
        print("💥 FATAL ERROR DURING STARTUP")
        print("=" * 50)
        print(f"Error: {str(e)}")
        print(f"Type: {type(e).__name__}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("=" * 50)
        sys.exit(1)
