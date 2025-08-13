#!/bin/bash
# Startup script for Render deployment
set -e  # Exit on any error

echo "🚀 MAISTORAGE RAG API STARTUP SCRIPT"
echo "=================================="
echo "Current directory: $(pwd)"
echo "Files in directory: $(ls -la)"
echo "PORT environment variable: ${PORT:-NOT_SET}"
echo "Python version: $(python --version)"

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ ERROR: main.py not found!"
    exit 1
fi

# Try to import and run with explicit error handling
echo "🚀 Attempting to start FastAPI app..."
python -c "
import sys
import os
print('Python executable:', sys.executable)
print('Python version:', sys.version)
print('Working directory:', os.getcwd())
print('PORT variable:', os.environ.get('PORT', 'NOT_SET'))
try:
    import main
    print('✅ main.py imported successfully')
except Exception as e:
    print(f'❌ Failed to import main.py: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

# Start the server
echo "🚀 Starting uvicorn server..."
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --log-level info
