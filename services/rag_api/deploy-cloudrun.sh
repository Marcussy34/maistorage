#!/bin/bash
# Google Cloud Run Deployment Script for MAI Storage RAG API

set -e  # Exit on any error

echo "🚀 DEPLOYING TO GOOGLE CLOUD RUN"
echo "================================"

# Configuration
PROJECT_ID="maistoragebackend"  # Replace with your actual project ID
SERVICE_NAME="maistorage-backend"
REGION="asia-southeast1"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

# Environment variables (replace with your actual values)
ENV_VARS="Use yours"

echo "📋 Configuration:"
echo "  Project ID: $PROJECT_ID"
echo "  Service Name: $SERVICE_NAME"
echo "  Region: $REGION"
echo "  Image: $IMAGE_NAME"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install it first:"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "🔐 Please authenticate with gcloud:"
    gcloud auth login
fi

# Set the project
echo "🔧 Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Build and deploy
echo "🏗️ Building and deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --source . \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --set-env-vars="$ENV_VARS" \
    --memory 4Gi \
    --cpu 2 \
    --max-instances 5 \
    --timeout 900 \
    --concurrency 20

echo "✅ Deployment completed!"
echo "🌐 Your service URL:"
gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)"

echo ""
echo "🧪 Test your deployment:"
echo "  Health check: $(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")/health"
echo "  Root endpoint: $(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")/"
