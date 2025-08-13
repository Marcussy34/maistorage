/**
 * Document Deletion API Endpoint
 * 
 * Handles deletion of specific documents or complete collection reset.
 * This endpoint removes documents from the Qdrant vector store.
 */

// Configuration
const QDRANT_URL = process.env.QDRANT_URL || 'http://localhost:6333';
const QDRANT_API_KEY = process.env.QDRANT_API_KEY || '';
const COLLECTION_NAME = 'maistorage_documents';

// Helper function to get Qdrant headers
function getQdrantHeaders() {
  const headers = {
    'Content-Type': 'application/json',
  };
  
  if (QDRANT_API_KEY) {
    headers['api-key'] = QDRANT_API_KEY;
  }
  
  return headers;
}

/**
 * Get collection info from Qdrant
 */
async function getCollectionInfo() {
  try {
    const response = await fetch(`${QDRANT_URL}/collections/${COLLECTION_NAME}`, {
      headers: getQdrantHeaders()
    });
    
    if (response.ok) {
      const data = await response.json();
      return {
        exists: true,
        points_count: data.result?.points_count || 0,
        vectors_count: data.result?.vectors_count || 0
      };
    } else if (response.status === 404) {
      return { exists: false, points_count: 0, vectors_count: 0 };
    }
  } catch (error) {
    console.error('Failed to get collection info:', error);
    throw new Error(`Failed to check collection: ${error.message}`);
  }
  return { exists: false, points_count: 0, vectors_count: 0 };
}

/**
 * List all documents in the collection with their metadata
 */
async function listDocuments() {
  try {
    const response = await fetch(`${QDRANT_URL}/collections/${COLLECTION_NAME}/points/scroll`, {
      method: 'POST',
      headers: getQdrantHeaders(),
      body: JSON.stringify({
        limit: 1000, // Get first 1000 points to extract unique documents
        with_payload: true,
        with_vector: false
      })
    });

    if (!response.ok) {
      throw new Error(`Failed to list documents: ${response.statusText}`);
    }

    const data = await response.json();
    const points = data.result?.points || [];
    
    // Group by document name to get unique documents
    const documentsMap = new Map();
    
    points.forEach(point => {
      const payload = point.payload || {};
      const docName = payload.doc_name;
      const docId = payload.doc_id;
      
      if (docName && !documentsMap.has(docName)) {
        documentsMap.set(docName, {
          doc_name: docName,
          doc_id: docId,
          doc_type: payload.doc_type,
          total_chunks: payload.total_chunks || 0,
          timestamp: payload.timestamp,
          char_count: payload.char_count
        });
      }
    });
    
    return Array.from(documentsMap.values());
  } catch (error) {
    console.error('Failed to list documents:', error);
    throw new Error(`Failed to list documents: ${error.message}`);
  }
}

/**
 * Delete specific document by name
 */
async function deleteDocument(docName) {
  try {
    // First, find all points belonging to this document
    const scrollResponse = await fetch(`${QDRANT_URL}/collections/${COLLECTION_NAME}/points/scroll`, {
      method: 'POST',
      headers: getQdrantHeaders(),
      body: JSON.stringify({
        filter: {
          must: [
            {
              key: "doc_name",
              match: { value: docName }
            }
          ]
        },
        limit: 10000, // Large limit to get all chunks of the document
        with_payload: false,
        with_vector: false
      })
    });

    if (!scrollResponse.ok) {
      throw new Error(`Failed to find document points: ${scrollResponse.statusText}`);
    }

    const scrollData = await scrollResponse.json();
    const points = scrollData.result?.points || [];
    
    if (points.length === 0) {
      return { success: false, message: 'Document not found', deleted_chunks: 0 };
    }

    // Extract point IDs
    const pointIds = points.map(point => point.id);
    
    // Delete the points
    const deleteResponse = await fetch(`${QDRANT_URL}/collections/${COLLECTION_NAME}/points/delete`, {
      method: 'POST',
      headers: getQdrantHeaders(),
      body: JSON.stringify({
        points: pointIds
      })
    });

    if (!deleteResponse.ok) {
      throw new Error(`Failed to delete document points: ${deleteResponse.statusText}`);
    }

    console.log(`Successfully deleted document "${docName}" with ${pointIds.length} chunks`);
    
    return {
      success: true,
      message: `Successfully deleted document "${docName}"`,
      deleted_chunks: pointIds.length
    };
    
  } catch (error) {
    console.error(`Failed to delete document "${docName}":`, error);
    throw new Error(`Failed to delete document: ${error.message}`);
  }
}

/**
 * Reset the entire collection (delete all documents)
 */
async function resetCollection() {
  try {
    // Check if collection exists
    const collectionInfo = await getCollectionInfo();
    
    if (!collectionInfo.exists) {
      return {
        success: true,
        message: 'Collection does not exist, nothing to reset',
        deleted_points: 0
      };
    }
    
    const pointsCount = collectionInfo.points_count;
    
    // Delete the entire collection
    const deleteResponse = await fetch(`${QDRANT_URL}/collections/${COLLECTION_NAME}`, {
      method: 'DELETE',
      headers: getQdrantHeaders()
    });

    if (!deleteResponse.ok) {
      throw new Error(`Failed to delete collection: ${deleteResponse.statusText}`);
    }
    
    console.log(`Successfully reset collection, deleted ${pointsCount} points`);
    
    return {
      success: true,
      message: `Successfully reset collection, deleted ${pointsCount} points`,
      deleted_points: pointsCount
    };
    
  } catch (error) {
    console.error('Failed to reset collection:', error);
    throw new Error(`Failed to reset collection: ${error.message}`);
  }
}

export default async function handler(req, res) {
  const startTime = Date.now();
  
  try {
    if (req.method === 'GET') {
      // List all documents
      const documents = await listDocuments();
      const collectionInfo = await getCollectionInfo();
      
      return res.status(200).json({
        success: true,
        documents,
        collection_info: collectionInfo,
        total_documents: documents.length
      });
      
    } else if (req.method === 'DELETE') {
      const { action, document_name } = req.body || {};
      
      if (!action) {
        return res.status(400).json({
          error: 'Missing action parameter',
          details: 'Specify "delete_document" or "reset_collection"'
        });
      }
      
      let result;
      
      if (action === 'delete_document') {
        if (!document_name) {
          return res.status(400).json({
            error: 'Missing document_name parameter',
            details: 'document_name is required when action is "delete_document"'
          });
        }
        
        result = await deleteDocument(document_name);
        
      } else if (action === 'reset_collection') {
        result = await resetCollection();
        
      } else {
        return res.status(400).json({
          error: 'Invalid action parameter',
          details: 'action must be "delete_document" or "reset_collection"'
        });
      }
      
      // Get updated collection info
      const collectionInfo = await getCollectionInfo();
      const processingTime = Date.now() - startTime;
      
      return res.status(200).json({
        ...result,
        collection_info: collectionInfo,
        processing_time_ms: processingTime
      });
      
    } else {
      return res.status(405).json({ error: 'Method not allowed' });
    }
    
  } catch (error) {
    console.error('Document deletion API error:', error);
    
    return res.status(500).json({
      error: 'Operation failed',
      details: error.message,
      code: 'DELETE_ERROR'
    });
  }
}
