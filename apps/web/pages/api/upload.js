/**
 * Document Upload API Endpoint
 * 
 * Handles file uploads and triggers ingestion into the RAG system.
 * This endpoint saves uploaded files and calls the ingestion service.
 */

import formidable from 'formidable';
import fs from 'fs/promises';
import path from 'path';
import { spawn } from 'child_process';

// Configuration
const UPLOAD_DIR = path.join(process.cwd(), '../../data/uploaded'); // Store in project data directory
const INDEXER_DIR = path.join(process.cwd(), '../../services/indexer');
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
  } catch {
    await fs.mkdir(UPLOAD_DIR, { recursive: true });
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
 * Run document ingestion using the indexer service
 */
function runIngestion(uploadDir) {
  return new Promise((resolve, reject) => {
    console.log(`Starting ingestion from directory: ${uploadDir}`);
    
    // Run the ingestion script using the virtual environment
    const pythonPath = path.join(INDEXER_DIR, 'venv', 'bin', 'python');
    const pythonProcess = spawn(pythonPath, ['ingest.py', '--path', uploadDir], {
      cwd: INDEXER_DIR,
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
      console.log(`Ingestion stdout: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
      console.error(`Ingestion stderr: ${data}`);
    });

    pythonProcess.on('close', (code) => {
      console.log(`Ingestion process exited with code ${code}`);
      
      if (code === 0) {
        // Parse ingestion results from stdout
        const lines = stdout.split('\n');
        const results = {
          success: true,
          totalChunks: 0,
          processedFiles: 0,
          logs: lines.filter(line => line.trim())
        };

        // Extract metrics from logs
        lines.forEach(line => {
          if (line.includes('Total chunks stored:')) {
            const match = line.match(/Total chunks stored: (\d+)/);
            if (match) results.totalChunks = parseInt(match[1]);
          }
          if (line.includes('Stored') && line.includes('chunks for')) {
            results.processedFiles++;
          }
        });

        resolve(results);
      } else {
        reject(new Error(`Ingestion failed with code ${code}: ${stderr}`));
      }
    });

    pythonProcess.on('error', (error) => {
      console.error(`Failed to start ingestion process: ${error}`);
      reject(error);
    });
  });
}

/**
 * Get collection info from Qdrant
 */
async function getCollectionInfo() {
  try {
    const QDRANT_URL = process.env.QDRANT_URL || 'http://localhost:6333';
    const response = await fetch(`${QDRANT_URL}/collections/maistorage_documents`);
    
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

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const startTime = Date.now();
  
  try {
    // Ensure upload directory exists
    await ensureUploadDir();

    // Parse uploaded files
    console.log('Parsing uploaded files...');
    const { files } = await parseFiles(req);

    if (!files || !files.files) {
      return res.status(400).json({ 
        error: 'No files uploaded',
        details: 'Please select at least one file to upload'
      });
    }

    // Handle both single file and multiple files
    const uploadedFiles = Array.isArray(files.files) ? files.files : [files.files];
    console.log(`Processing ${uploadedFiles.length} uploaded files`);

    const processedFiles = [];
    const uploadedPaths = [];

    // Process each uploaded file
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

        // Generate unique filename to avoid conflicts
        const timestamp = Date.now();
        const ext = path.extname(file.originalFilename);
        const baseName = path.basename(file.originalFilename, ext);
        const uniqueName = `${baseName}_${timestamp}${ext}`;
        const targetPath = path.join(UPLOAD_DIR, uniqueName);

        // Move file to upload directory
        await fs.rename(file.filepath, targetPath);
        uploadedPaths.push(targetPath);
        
        console.log(`Successfully saved file: ${uniqueName}`);
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

    // Check if any files were successfully uploaded
    const successfulUploads = processedFiles.filter(f => f.success);
    if (successfulUploads.length === 0) {
      return res.status(400).json({
        error: 'No files were successfully uploaded',
        files: processedFiles
      });
    }

    // Run ingestion on the uploaded files
    console.log('Starting document ingestion...');
    let ingestionResults = null;
    try {
      ingestionResults = await runIngestion(UPLOAD_DIR);
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

    // Clean up uploaded files after successful ingestion
    if (ingestionResults?.success) {
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
