#!/bin/bash
# Startup script for Render deployment
echo "Starting FastAPI app on port ${PORT:-8000}"
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
