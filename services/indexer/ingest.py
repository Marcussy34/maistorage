"""
Document ingestion script for MAI Storage.
Phase 1 implementation: Document loading, chunking, embedding, and storage.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import uuid

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_environment():
    """Load environment variables from .env file."""
    # Try to load from various .env locations
    env_paths = [
        Path(".env"),
        Path("../.env"),
        Path("../../.env"),
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            logging.info(f"Loaded environment from: {env_path}")
            break
    else:
        logging.warning("No .env file found. Using environment variables.")


def get_document_loader(file_path: Path):
    """
    Get appropriate document loader based on file extension.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Document loader instance
    """
    suffix = file_path.suffix.lower()
    
    if suffix == '.pdf':
        return PyPDFLoader(str(file_path))
    elif suffix in ['.md', '.markdown']:
        return UnstructuredMarkdownLoader(str(file_path))
    elif suffix in ['.html', '.htm']:
        return UnstructuredHTMLLoader(str(file_path))
    elif suffix == '.txt':
        return TextLoader(str(file_path))
    else:
        # Fallback to text loader for unknown types
        logging.warning(f"Unknown file type {suffix}, using TextLoader")
        return TextLoader(str(file_path))


def create_text_splitter():
    """
    Create a text splitter with semantic chunking parameters.
    200-500 tokens with 15-20% overlap as specified in PLAN.md
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=500,  # Target ~500 tokens (roughly 375 chars per token)
        chunk_overlap=100,  # ~20% overlap
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )


def setup_qdrant_client():
    """Setup Qdrant client and create collection if needed."""
    logger = logging.getLogger(__name__)
    
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
    
    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        # Test connection
        collections = client.get_collections()
        logger.info(f"Connected to Qdrant at {qdrant_url}")
        
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        raise


def create_collection_if_not_exists(client: QdrantClient, collection_name: str = "maistorage_documents"):
    """Create Qdrant collection if it doesn't exist."""
    logger = logging.getLogger(__name__)
    
    try:
        collections = client.get_collections()
        existing_names = [col.name for col in collections.collections]
        
        if collection_name not in existing_names:
            logger.info(f"Creating collection: {collection_name}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI text-embedding-3-small dimensions
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {collection_name}")
        else:
            logger.info(f"Collection already exists: {collection_name}")
            
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        raise


def ingest_documents(data_path: Path):
    """
    Ingest documents from the specified path.
    
    Args:
        data_path: Path to the directory containing documents to ingest
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting document ingestion from: {data_path}")
    
    if not data_path.exists():
        logger.error(f"Data path does not exist: {data_path}")
        return
    
    # Load environment variables
    load_environment()
    
    # Setup components
    try:
        # Initialize embeddings model
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        embeddings = OpenAIEmbeddings(model=embedding_model)
        logger.info(f"Initialized OpenAI embeddings with model: {embedding_model}")
        
        # Setup Qdrant
        client = setup_qdrant_client()
        collection_name = "maistorage_documents"
        create_collection_if_not_exists(client, collection_name)
        
        # Setup text splitter
        text_splitter = create_text_splitter()
        
    except Exception as e:
        logger.error(f"Failed to setup components: {e}")
        return
    
    # Find all documents in the data path
    supported_extensions = {'.pdf', '.md', '.markdown', '.html', '.htm', '.txt'}
    document_files = []
    
    if data_path.is_file():
        document_files = [data_path]
    else:
        for ext in supported_extensions:
            document_files.extend(data_path.glob(f"**/*{ext}"))
    
    if not document_files:
        logger.warning(f"No supported documents found in {data_path}")
        return
    
    logger.info(f"Found {len(document_files)} documents to process")
    
    # Process each document
    total_chunks = 0
    
    for doc_path in document_files:
        try:
            logger.info(f"Processing: {doc_path}")
            
            # Load document
            loader = get_document_loader(doc_path)
            documents = loader.load()
            
            # Split into chunks
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Split {doc_path.name} into {len(chunks)} chunks")
            
            # Generate embeddings and prepare points
            texts = [chunk.page_content for chunk in chunks]
            embeddings_vectors = embeddings.embed_documents(texts)
            
            # Prepare points for Qdrant
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_vectors)):
                point_id = str(uuid.uuid4())
                
                # Create comprehensive metadata
                metadata = {
                    "doc_id": str(doc_path),
                    "doc_name": doc_path.name,
                    "doc_type": doc_path.suffix.lower(),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "text": chunk.page_content,
                    "timestamp": datetime.now().isoformat(),
                    "char_count": len(chunk.page_content),
                    "start_index": chunk.metadata.get("start_index", 0),
                }
                
                # Add any additional metadata from the document
                metadata.update(chunk.metadata)
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=metadata
                )
                points.append(point)
            
            # Store in Qdrant
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            total_chunks += len(chunks)
            logger.info(f"Stored {len(chunks)} chunks for {doc_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to process {doc_path}: {e}")
            continue
    
    # Final summary
    logger.info(f"Ingestion complete. Total chunks stored: {total_chunks}")
    
    # Verify collection
    try:
        collection_info = client.get_collection(collection_name)
        logger.info(f"Collection info: {collection_info.points_count} points stored")
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")


def main():
    """Main entry point for the ingestion script."""
    parser = argparse.ArgumentParser(description="Ingest documents into MAI Storage")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("./data"),
        help="Path to directory containing documents to ingest"
    )
    
    args = parser.parse_args()
    setup_logging()
    ingest_documents(args.path)


if __name__ == "__main__":
    main()
