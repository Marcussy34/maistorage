/**
 * Document Upload API Endpoint
 * 
 * Handles file uploads and triggers ingestion into the RAG system.
 * This endpoint saves uploaded files and calls the ingestion service.
 */

import formidable from 'formidable';
import fs from 'fs/promises';
import path from 'path';

// Configuration - Use environment variables for production
const RAG_API_URL = process.env.RAG_API_URL || 'http://localhost:8000';
const UPLOAD_DIR = '/tmp/uploads'; // Use /tmp in serverless environment
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
const ALLOWED_EXTENSIONS = ['.pdf', '.md', '.markdown', '.html', '.htm', '.txt'];

// Disable bodyParser to handle multipart form data
export const config = {
  api: {
    bodyParser: false,
  },
};

/**
 * Ensure upload directory exists
 */
async function ensureUploadDir() {
  try {
    await fs.access(UPLOAD_DIR);
    console.log('Upload directory already exists:', UPLOAD_DIR);
  } catch (error) {
    console.log('Creating upload directory:', UPLOAD_DIR);
    try {
      await fs.mkdir(UPLOAD_DIR, { recursive: true });
      console.log('Upload directory created successfully');
    } catch (mkdirError) {
      console.error('Failed to create upload directory:', mkdirError);
      throw new Error(`Failed to create upload directory: ${mkdirError.message}`);
    }
  }
}

/**
 * Validate file type
 */
function isValidFileType(filename) {
  const ext = path.extname(filename).toLowerCase();
  return ALLOWED_EXTENSIONS.includes(ext);
}

/**
 * Parse uploaded files using formidable
 */
function parseFiles(req) {
  return new Promise((resolve, reject) => {
    const form = formidable({
      maxFileSize: MAX_FILE_SIZE,
      maxFiles: 10, // Allow up to 10 files at once
      allowEmptyFiles: false,
      filter: ({ name, originalFilename, mimetype }) => {
        // Filter by file extension since MIME types can be unreliable
        return isValidFileType(originalFilename);
      }
    });

    form.parse(req, (err, fields, files) => {
      if (err) {
        reject(err);
        return;
      }
      resolve({ fields, files });
    });
  });
}

/**
 * Simple text chunking function for document processing
 * Matches the indexer service chunking strategy
 */
function chunkText(text, chunkSize = 500, overlap = 100) {
  const chunks = [];
  let start = 0;
  
  // Use similar separators as the indexer service
  const separators = ['\n\n', '\n', '. ', ' ', ''];
  
  while (start < text.length) {
    let end = Math.min(start + chunkSize, text.length);
    
    // Try to find a good breaking point using separators
    if (end < text.length) {
      for (const separator of separators) {
        const separatorPos = text.lastIndexOf(separator, end);
        if (separatorPos > start + chunkSize * 0.5) {
          end = separatorPos + separator.length;
          break;
        }
      }
    }
    
    const chunk = text.slice(start, end).trim();
    
    if (chunk) {
      chunks.push({
        text: chunk,
        start_index: start,
        char_count: chunk.length
      });
    }
    
    start = end - overlap;
    if (start >= text.length) break;
  }
  
  return chunks;
}

/**
 * Generate embeddings using OpenAI API
 */
async function generateEmbeddings(texts) {
  const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
  const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || 'text-embedding-3-small';
  
  if (!OPENAI_API_KEY) {
    throw new Error('OpenAI API key not configured');
  }
  
  console.log(`Generating embeddings for ${texts.length} chunks using ${EMBEDDING_MODEL}`);
  
  try {
    const response = await fetch('https://api.openai.com/v1/embeddings', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${OPENAI_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        input: texts,
        model: EMBEDDING_MODEL
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`OpenAI API error (${response.status}): ${errorText}`);
    }

    const data = await response.json();
    return data.data.map(item => item.embedding);
  } catch (error) {
    console.error('Failed to generate embeddings:', error);
    throw error;
  }
}

/**
 * Store document chunks in Qdrant with embeddings
 */
async function storeInQdrant(fileInfo, chunks, embeddings) {
  const QDRANT_URL = process.env.QDRANT_URL;
  const QDRANT_API_KEY = process.env.QDRANT_API_KEY;
  const COLLECTION_NAME = 'maistorage_documents';
  
  if (!QDRANT_URL) {
    throw new Error('Qdrant URL not configured');
  }
  
  console.log(`Storing ${chunks.length} chunks in Qdrant for ${fileInfo.name}`);
  
  // Prepare points for Qdrant (matching indexer service structure)
  const points = chunks.map((chunk, index) => ({
    id: generateUUID(),
    vector: embeddings[index],
    payload: {
      doc_id: fileInfo.path,
      doc_name: fileInfo.name,
      doc_type: path.extname(fileInfo.name).toLowerCase(),
      chunk_index: index,
      total_chunks: chunks.length,
      text: chunk.text,
      timestamp: new Date().toISOString(),
      char_count: chunk.char_count,
      start_index: chunk.start_index
    }
  }));
  
  // Store in Qdrant
  const headers = {
    'Content-Type': 'application/json'
  };
  
  if (QDRANT_API_KEY) {
    headers['api-key'] = QDRANT_API_KEY;
  }
  
  const response = await fetch(`${QDRANT_URL}/collections/${COLLECTION_NAME}/points`, {
    method: 'PUT',
    headers,
    body: JSON.stringify({
      points
    })
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Qdrant storage failed (${response.status}): ${errorText}`);
  }
  
  const result = await response.json();
  console.log(`Successfully stored ${chunks.length} chunks in Qdrant`);
  
  return {
    stored_chunks: chunks.length,
    qdrant_result: result
  };
}

/**
 * Generate UUID for document chunks
 */
function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

/**
 * Process document for storage with full pipeline
 */
async function processDocumentForStorage(fileInfo) {
  try {
    console.log(`Processing document: ${fileInfo.name}`);
    
    // Read and chunk the document
    const content = await fs.readFile(fileInfo.path, 'utf-8');
    const chunks = chunkText(content);
    
    if (chunks.length === 0) {
      throw new Error('No chunks generated from document');
    }
    
    // Generate embeddings for all chunks
    const chunkTexts = chunks.map(chunk => chunk.text);
    const embeddings = await generateEmbeddings(chunkTexts);
    
    // Store in Qdrant
    const storageResult = await storeInQdrant(fileInfo, chunks, embeddings);
    
    return {
      filename: fileInfo.name,
      chunks: chunks,
      totalChunks: chunks.length,
      stored: true,
      storageResult
    };
  } catch (error) {
    console.error(`Failed to process document ${fileInfo.name}:`, error);
    throw error;
  }
}

/**
 * Run document ingestion with full pipeline: chunking, embedding, and Qdrant storage
 */
async function runIngestion(uploadedFiles) {
  console.log(`Processing and storing ${uploadedFiles.length} files in Qdrant`);
  
  try {
    const processedDocs = [];
    let totalChunks = 0;
    let totalStoredChunks = 0;
    
    for (const fileInfo of uploadedFiles) {
      try {
        const processed = await processDocumentForStorage(fileInfo);
        processedDocs.push(processed);
        totalChunks += processed.totalChunks;
        if (processed.stored) {
          totalStoredChunks += processed.storageResult.stored_chunks;
        }
      } catch (fileError) {
        console.error(`Failed to process file ${fileInfo.name}:`, fileError);
        // Add failed file to results
        processedDocs.push({
          filename: fileInfo.name,
          error: fileError.message,
          stored: false,
          totalChunks: 0
        });
      }
    }

    const successfulFiles = processedDocs.filter(doc => doc.stored);
    const failedFiles = processedDocs.filter(doc => !doc.stored);

    console.log(`Successfully processed ${successfulFiles.length}/${uploadedFiles.length} files`);
    console.log(`Total chunks stored in Qdrant: ${totalStoredChunks}`);

    return {
      success: successfulFiles.length > 0,
      totalChunks: totalStoredChunks,
      processedFiles: successfulFiles.length,
      failedFiles: failedFiles.length,
      logs: [
        `Successfully processed ${successfulFiles.length} out of ${uploadedFiles.length} files`,
        `Generated and stored ${totalStoredChunks} text chunks in Qdrant`,
        `Documents are now available for search and retrieval`,
        ...(failedFiles.length > 0 ? [`${failedFiles.length} files failed to process`] : [])
      ],
      processedDocs: processedDocs,
      details: {
        successful: successfulFiles.map(doc => ({
          filename: doc.filename,
          chunks: doc.totalChunks
        })),
        failed: failedFiles.map(doc => ({
          filename: doc.filename,
          error: doc.error
        }))
      }
    };

  } catch (error) {
    console.error('Document ingestion pipeline failed:', error);
    throw error;
  }
}

/**
 * Get collection info from Qdrant
 */
async function getCollectionInfo() {
  try {
    const QDRANT_URL = process.env.QDRANT_URL || 'http://localhost:6333';
    const QDRANT_API_KEY = process.env.QDRANT_API_KEY || '';
    
    const headers = {};
    if (QDRANT_API_KEY) {
      headers['api-key'] = QDRANT_API_KEY;
    }
    
    const response = await fetch(`${QDRANT_URL}/collections/maistorage_documents`, {
      headers
    });
    
    if (response.ok) {
      const data = await response.json();
      return {
        points_count: data.result?.points_count || 0,
        vectors_count: data.result?.vectors_count || 0
      };
    }
  } catch (error) {
    console.error('Failed to get collection info:', error);
  }
  return null;
}

async function handler(req, res) {
  // Basic CORS headers (safe for same-origin too)
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS, HEAD');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  console.log('Upload handler called with method:', req.method);
  console.log('Request headers:', JSON.stringify(req.headers, null, 2));
  console.log('Request URL:', req.url);

  // Handle preflight
  if (req.method === 'OPTIONS' || req.method === 'HEAD') {
    return res.status(204).end();
  }

  if (req.method !== 'POST') {
    console.log('Method not allowed, received:', req.method);
    res.setHeader('Allow', 'POST, OPTIONS, HEAD');
    return res.status(405).json({ 
      error: 'Method not allowed',
      received_method: req.method,
      allowed_methods: ['POST']
    });
  }

  const startTime = Date.now();
  console.log('Upload API called at:', new Date().toISOString());
  console.log('Upload directory path:', UPLOAD_DIR);
  console.log('RAG API URL:', RAG_API_URL);
  
  try {
    // Ensure upload directory exists
    await ensureUploadDir();
    console.log('Upload directory verified/created successfully');

    const contentType = (req.headers['content-type'] || '').toLowerCase();
    const processedFiles = [];
    const uploadedPaths = [];
    const ingestionFileInfos = [];

    // Branch: multipart form-data (formidable)
    if (contentType.includes('multipart/form-data')) {
      console.log('Parsing uploaded files via multipart/form-data');
      const { files } = await parseFiles(req);

      if (!files || !files.files) {
        return res.status(400).json({ 
          error: 'No files uploaded',
          details: 'Please select at least one file to upload'
        });
      }

      const uploadedFiles = Array.isArray(files.files) ? files.files : [files.files];
      console.log(`Processing ${uploadedFiles.length} uploaded files (multipart)`);

      for (const file of uploadedFiles) {
        try {
          if (!isValidFileType(file.originalFilename)) {
            processedFiles.push({
              name: file.originalFilename,
              success: false,
              error: 'Invalid file type. Only PDF, Markdown, HTML, and TXT files are supported.'
            });
            continue;
          }

          const timestamp = Date.now();
          const ext = path.extname(file.originalFilename);
          const baseName = path.basename(file.originalFilename, ext);
          const uniqueName = `${baseName}_${timestamp}${ext}`;
          const targetPath = path.join(UPLOAD_DIR, uniqueName);

          await fs.rename(file.filepath, targetPath);
          uploadedPaths.push(targetPath);
          ingestionFileInfos.push({ path: targetPath, name: file.originalFilename });

          console.log(`Saved file (multipart): ${uniqueName}`);
          processedFiles.push({
            name: file.originalFilename,
            uniqueName,
            path: targetPath,
            size: file.size,
            success: true
          });
        } catch (error) {
          console.error(`Failed to process file ${file.originalFilename}:`, error);
          processedFiles.push({
            name: file.originalFilename,
            success: false,
            error: error.message
          });
        }
      }
    } else {
      // Branch: JSON body with base64 content (avoids multipart on Vercel)
      console.log('Parsing uploaded files via JSON');

      // Read raw body because bodyParser is disabled
      const rawChunks = [];
      for await (const chunk of req) rawChunks.push(chunk);
      const rawBody = Buffer.concat(rawChunks).toString('utf-8');

      if (!rawBody) {
        return res.status(400).json({ error: 'No request body received' });
      }

      let payload;
      try {
        payload = JSON.parse(rawBody);
      } catch (e) {
        return res.status(400).json({ error: 'Invalid JSON body' });
      }

      const jsonFiles = Array.isArray(payload.files)
        ? payload.files
        : Array.isArray(payload.documents)
          ? payload.documents
          : Array.isArray(payload.items)
            ? payload.items
            : [];
      if (jsonFiles.length === 0) {
        return res.status(400).json({ 
          error: 'No files provided',
          details: 'Send { files: [{ name, content }] } where content is base64 or data URL'
        });
      }

      console.log(`Processing ${jsonFiles.length} uploaded files (JSON)`);

      for (const jf of jsonFiles) {
        try {
          const originalName = jf.name || 'upload.txt';
          if (!isValidFileType(originalName)) {
            processedFiles.push({
              name: originalName,
              success: false,
              error: 'Invalid file type. Only PDF, Markdown, HTML, and TXT files are supported.'
            });
            continue;
          }

          const contentField = jf.content_base64 || jf.base64 || jf.content;
          if (!contentField) {
            processedFiles.push({
              name: originalName,
              success: false,
              error: 'Missing file content'
            });
            continue;
          }

          // Support data URLs and raw base64
          const base64 = typeof contentField === 'string' && contentField.includes('base64,')
            ? contentField.split('base64,').pop()
            : contentField;
          const buffer = Buffer.from(base64, 'base64');

          const timestamp = Date.now();
          const ext = path.extname(originalName) || '.txt';
          const baseName = path.basename(originalName, ext);
          const uniqueName = `${baseName}_${timestamp}${ext}`;
          const targetPath = path.join(UPLOAD_DIR, uniqueName);

          await fs.writeFile(targetPath, buffer);
          uploadedPaths.push(targetPath);
          ingestionFileInfos.push({ path: targetPath, name: originalName });

          console.log(`Saved file (JSON): ${uniqueName}`);
          processedFiles.push({
            name: originalName,
            uniqueName,
            path: targetPath,
            size: buffer.length,
            success: true
          });
        } catch (error) {
          console.error('Failed to process JSON file:', error);
          processedFiles.push({
            name: jf?.name || 'unknown',
            success: false,
            error: error.message
          });
        }
      }
    }

    // Check if any files were successfully uploaded
    const successfulUploads = processedFiles.filter(f => f.success);
    if (successfulUploads.length === 0) {
      return res.status(400).json({
        error: 'No files were successfully uploaded',
        files: processedFiles
      });
    }

    // Run ingestion on the uploaded files
    console.log('Starting document ingestion via RAG API...');
    let ingestionResults = null;
    try {
      // Only pass successfully uploaded files to ingestion
      const filesToIngest = ingestionFileInfos.length
        ? ingestionFileInfos
        : successfulUploads.map(file => ({ path: file.path, name: file.name }));

      ingestionResults = await runIngestion(filesToIngest);
      console.log('Ingestion completed successfully:', ingestionResults);
    } catch (ingestionError) {
      console.error('Ingestion failed:', ingestionError);
      
      // Mark all files as having ingestion errors
      processedFiles.forEach(file => {
        if (file.success) {
          file.ingestionError = ingestionError.message;
        }
      });
    }

    // Get updated collection info
    const collectionInfo = await getCollectionInfo();

    // Prepare response
    const processingTime = Date.now() - startTime;
    const response = {
      success: true,
      message: `Successfully processed ${successfulUploads.length} out of ${uploadedFiles.length} files`,
      files: processedFiles,
      summary: {
        total_files: uploadedFiles.length,
        successful_uploads: successfulUploads.length,
        failed_uploads: processedFiles.length - successfulUploads.length,
        total_chunks: ingestionResults?.totalChunks || 0,
        processed_files: ingestionResults?.processedFiles || 0,
        processing_time_ms: processingTime,
        collection_info: collectionInfo
      },
      ingestion: {
        success: ingestionResults?.success || false,
        details: ingestionResults || null
      }
    };

    // Clean up uploaded files after processing (important in serverless environment)
    try {
      await Promise.all(uploadedPaths.map(async (filePath) => {
        try {
          await fs.unlink(filePath);
          console.log(`Cleaned up uploaded file: ${filePath}`);
        } catch (error) {
          console.warn(`Failed to clean up file ${filePath}:`, error.message);
        }
      }));
    } catch (error) {
      console.warn('Some uploaded files could not be cleaned up:', error.message);
    }

    res.status(200).json(response);

  } catch (error) {
    console.error('Upload API error:', error);
    
    res.status(500).json({
      error: 'Upload failed',
      details: error.message,
      code: 'UPLOAD_ERROR'
    });
  }
}

export default handler;
