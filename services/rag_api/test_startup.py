#!/usr/bin/env python3
"""
Minimal test script to debug Render startup issues
"""
import os
import sys
import traceback

def test_basic_imports():
    """Test if we can import basic modules"""
    try:
        print("=" * 50)
        print("🧪 TESTING BASIC IMPORTS")
        print("=" * 50)
        
        print("✅ Testing Python basics...")
        import json
        import time
        print("✅ json, time - OK")
        
        print("✅ Testing environment...")
        port = os.environ.get("PORT", "8000")
        print(f"✅ PORT environment variable: {port}")
        
        print("✅ Testing FastAPI...")
        from fastapi import FastAPI
        print("✅ FastAPI - OK")
        
        print("✅ Testing uvicorn...")
        import uvicorn
        print("✅ uvicorn - OK")
        
        print("✅ Testing pydantic...")
        from pydantic import BaseModel
        print("✅ pydantic - OK")
        
        print("✅ Testing dotenv...")
        from dotenv import load_dotenv
        print("✅ dotenv - OK")
        
        print("✅ All basic imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_environment_variables():
    """Test environment variables"""
    try:
        print("=" * 50)
        print("🔧 TESTING ENVIRONMENT VARIABLES")
        print("=" * 50)
        
        required_vars = [
            "OPENAI_API_KEY",
            "QDRANT_URL", 
            "QDRANT_API_KEY"
        ]
        
        for var in required_vars:
            value = os.environ.get(var)
            if value:
                # Mask sensitive data
                if "api" in var.lower() or "key" in var.lower():
                    masked = value[:10] + "..." if len(value) > 10 else "***"
                    print(f"✅ {var}: {masked}")
                else:
                    print(f"✅ {var}: {value}")
            else:
                print(f"❌ {var}: NOT SET")
                
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        traceback.print_exc()
        return False

def test_minimal_server():
    """Test starting a minimal FastAPI server"""
    try:
        print("=" * 50)
        print("🚀 TESTING MINIMAL SERVER")
        print("=" * 50)
        
        from fastapi import FastAPI
        import uvicorn
        
        # Create minimal app
        app = FastAPI(title="Test App")
        
        @app.get("/")
        def read_root():
            return {"message": "Hello from minimal test app!"}
        
        @app.get("/health")
        def health():
            return {"status": "ok"}
        
        port = int(os.environ.get("PORT", 8000))
        print(f"🚀 Starting minimal server on port {port}")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
        
    except Exception as e:
        print(f"❌ Server test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 RENDER STARTUP DIAGNOSTIC")
    print(f"🐍 Python: {sys.version}")
    print(f"📁 Working dir: {os.getcwd()}")
    print(f"📄 Files in current dir: {os.listdir('.')}")
    
    # Test 1: Basic imports
    if not test_basic_imports():
        print("💥 Basic imports failed - stopping")
        sys.exit(1)
    
    # Test 2: Environment variables
    test_environment_variables()
    
    # Test 3: Try minimal server
    print("🚀 Attempting to start minimal server...")
    test_minimal_server()
