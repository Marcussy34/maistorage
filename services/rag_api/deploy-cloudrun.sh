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

# Check if .env.cloudrun exists
if [ ! -f ".env.cloudrun" ]; then
    echo "❌ .env.cloudrun file not found!"
    echo "📝 Please create .env.cloudrun with your environment variables."
    echo "   You can copy from .env.example and modify for production."
    exit 1
fi

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
echo "⏱️  This may take several minutes..."

# Deploy with improved settings for better startup
gcloud run deploy $SERVICE_NAME \
    --source . \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --env-vars-file .env.cloudrun \
    --memory 2Gi \
    --cpu 1 \
    --max-instances 10 \
    --timeout 3600 \
    --concurrency 10 \
    --cpu-boost \
    --execution-environment gen2 \
    --no-traffic

# Check deployment status
if [ $? -eq 0 ]; then
    echo "🎉 Build successful! Now routing traffic..."
    gcloud run services update-traffic $SERVICE_NAME --region=$REGION --to-latest
else
    echo "❌ Deployment failed!"
    echo "🔍 Check logs with:"
    echo "   gcloud run services logs read $SERVICE_NAME --region=$REGION"
    exit 1
fi

echo "✅ Deployment completed!"
echo "🌐 Your service URL:"
gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)"

echo ""
echo ""
echo "🧪 Test your deployment:"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
echo "  Health check: $SERVICE_URL/health"
echo "  Root endpoint: $SERVICE_URL/"
echo ""
echo "🔍 Debugging commands (if needed):"
echo "  View logs: gcloud run services logs read $SERVICE_NAME --region=$REGION"
echo "  Service details: gcloud run services describe $SERVICE_NAME --region=$REGION"
echo "  List revisions: gcloud run revisions list --service=$SERVICE_NAME --region=$REGION"
echo ""
echo "🚨 If deployment fails:"
echo "  1. Check the Cloud Build logs in the console"
echo "  2. Verify all environment variables in .env.cloudrun"
echo "  3. Ensure Docker builds locally: docker build -t test-image ."
echo "  4. Check service logs for startup errors"
