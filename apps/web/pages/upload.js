import React, { useEffect, useState, useRef, useCallback } from 'react';
import Head from "next/head";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "../src/components/ui/button";

import { Badge } from "../src/components/ui/badge";
import { 
  Upload, 
  FileText, 
  Database, 
  Moon, 
  Sun, 
  ArrowLeft,
  CheckCircle,
  XCircle,
  Loader2,
  FileIcon,
  Trash2,
  RefreshCw,
  AlertCircle,
  FolderOpen,
  Clock,
  HardDriveIcon
} from "lucide-react";

/**
 * Document Upload Page
 * 
 * Allows users to upload documents for ingestion into the RAG system.
 * Features drag-and-drop upload, progress tracking, and file management.
 */
export default function UploadPage() {
  const [darkMode, setDarkMode] = useState(false);
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [uploadResults, setUploadResults] = useState(null);
  const [existingDocuments, setExistingDocuments] = useState([]);
  const [loadingDocuments, setLoadingDocuments] = useState(false);
  const [deletingDocument, setDeletingDocument] = useState(null);
  const [resettingCollection, setResettingCollection] = useState(false);
  const [showDocumentManager, setShowDocumentManager] = useState(false);
  const fileInputRef = useRef(null);
  const mountedRef = useRef(false);

  // Initialize dark mode from localStorage or system preference
  useEffect(() => {
    if (mountedRef.current) return;
    mountedRef.current = true;
    
    const savedDarkMode = localStorage.getItem('darkMode');
    const systemDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    const shouldUseDarkMode = savedDarkMode 
      ? JSON.parse(savedDarkMode) 
      : systemDarkMode;

    setDarkMode(shouldUseDarkMode);
    updateDarkModeClass(shouldUseDarkMode);
  }, []);

  // Update dark mode class on document
  const updateDarkModeClass = (isDark) => {
    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  };

  // Toggle dark mode
  const toggleDarkMode = () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);
    localStorage.setItem('darkMode', JSON.stringify(newDarkMode));
    updateDarkModeClass(newDarkMode);
  };

  // Handle drag events
  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  // Handle file drop
  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  }, []);

  // Handle file selection
  const handleFiles = (fileList) => {
    const newFiles = Array.from(fileList).map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      file,
      status: 'ready', // ready, uploading, success, error
      progress: 0,
      error: null
    }));

    // Filter for supported file types
    const supportedTypes = ['.pdf', '.md', '.markdown', '.html', '.htm', '.txt'];
    const validFiles = newFiles.filter(fileObj => {
      const extension = '.' + fileObj.file.name.split('.').pop().toLowerCase();
      return supportedTypes.includes(extension);
    });

    if (validFiles.length !== newFiles.length) {
      alert('Some files were filtered out. Only PDF, Markdown, HTML, and TXT files are supported.');
    }

    setFiles(prevFiles => [...prevFiles, ...validFiles]);
  };

  // Handle file input change
  const handleFileInputChange = (e) => {
    if (e.target.files) {
      handleFiles(e.target.files);
    }
  };

  // Remove file from list
  const removeFile = (fileId) => {
    setFiles(prevFiles => prevFiles.filter(f => f.id !== fileId));
  };

  // Clear all files
  const clearAllFiles = () => {
    setFiles([]);
    setUploadResults(null);
  };

  // Get file icon based on extension
  const getFileIcon = (filename) => {
    const extension = filename.split('.').pop().toLowerCase();
    switch (extension) {
      case 'pdf':
        return <FileText className="h-5 w-5 text-red-500" />;
      case 'md':
      case 'markdown':
        return <FileText className="h-5 w-5 text-blue-500" />;
      case 'html':
      case 'htm':
        return <FileText className="h-5 w-5 text-orange-500" />;
      case 'txt':
        return <FileText className="h-5 w-5 text-gray-500" />;
      default:
        return <FileIcon className="h-5 w-5 text-gray-400" />;
    }
  };

  // Format file size
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Load existing documents from the collection
  const loadExistingDocuments = async () => {
    setLoadingDocuments(true);
    try {
      const response = await fetch('/api/documents/delete', {
        method: 'GET',
      });

      if (!response.ok) {
        throw new Error(`Failed to load documents: ${response.statusText}`);
      }

      const result = await response.json();
      setExistingDocuments(result.documents || []);
    } catch (error) {
      console.error('Failed to load existing documents:', error);
      alert('Failed to load existing documents: ' + error.message);
    } finally {
      setLoadingDocuments(false);
    }
  };

  // Delete a specific document
  const deleteDocument = async (docName) => {
    if (!confirm(`Are you sure you want to delete "${docName}"? This action cannot be undone.`)) {
      return;
    }

    setDeletingDocument(docName);
    try {
      const response = await fetch('/api/documents/delete', {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          action: 'delete_document',
          document_name: docName
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to delete document: ${response.statusText}`);
      }

      const result = await response.json();
      
      if (result.success) {
        // Remove the document from the list
        setExistingDocuments(prev => prev.filter(doc => doc.doc_name !== docName));
        alert(result.message);
      } else {
        throw new Error(result.message || 'Unknown error');
      }
    } catch (error) {
      console.error('Failed to delete document:', error);
      alert('Failed to delete document: ' + error.message);
    } finally {
      setDeletingDocument(null);
    }
  };

  // Reset the entire collection
  const resetCollection = async () => {
    if (!confirm('Are you sure you want to delete ALL documents? This will permanently remove all data from your knowledge base and cannot be undone.')) {
      return;
    }

    if (!confirm('This is your final warning. All documents will be permanently deleted. Are you absolutely sure?')) {
      return;
    }

    setResettingCollection(true);
    try {
      const response = await fetch('/api/documents/delete', {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          action: 'reset_collection'
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to reset collection: ${response.statusText}`);
      }

      const result = await response.json();
      
      if (result.success) {
        setExistingDocuments([]);
        alert(result.message);
      } else {
        throw new Error(result.message || 'Unknown error');
      }
    } catch (error) {
      console.error('Failed to reset collection:', error);
      alert('Failed to reset collection: ' + error.message);
    } finally {
      setResettingCollection(false);
    }
  };

  // Upload files
  const uploadFiles = async () => {
    if (files.length === 0) return;

    setUploading(true);
    setUploadResults(null);

    try {
      // Update all files to uploading status
      setFiles(prevFiles => prevFiles.map(f => ({ ...f, status: 'uploading', progress: 0 })));

      // Prefer JSON uploads to avoid multipart issues on serverless (Vercel)
      const toDataURL = (file) =>
        new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.onerror = reject;
          reader.readAsDataURL(file);
        });

      const jsonFiles = await Promise.all(
        files.map(async (fileObj) => ({
          name: fileObj.file.name,
          content: await toDataURL(fileObj.file), // data:...;base64,...
        }))
      );

      // Send directly to Cloud Run via Vercel rewrite
      const response = await fetch('/api/rag/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          // Prefer base64 payload key to avoid any platform filtering
          files: jsonFiles.map(f => ({
            name: f.name,
            content_base64: (typeof f.content === 'string' && f.content.includes('base64,'))
              ? f.content.split('base64,').pop()
              : f.content
          }))
        }),
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const ragResp = await response.json();

      // Map Cloud Run response to UI structure
      const mappedResult = {
        success: !!ragResp?.success,
        files: files.map(f => ({
          name: f.file.name,
          success: !!ragResp?.success,
          error: ragResp?.success ? null : (ragResp?.error || 'Ingestion failed')
        })),
        summary: {
          total_files: files.length,
          successful_uploads: ragResp?.success ? files.length : 0,
          failed_uploads: ragResp?.success ? 0 : files.length,
          total_chunks: ragResp?.stored_chunks ?? 0,
          processed_files: ragResp?.success ? files.length : 0,
          processing_time_ms: 0,
          collection_info: null
        },
        ingestion: {
          success: !!ragResp?.success,
          details: ragResp
        }
      };

      setUploadResults(mappedResult);

      // Update file statuses based on results
      setFiles(prevFiles => prevFiles.map(fileObj => {
        const uploadedFile = mappedResult.files?.find(f => f.name === fileObj.file.name);
        if (uploadedFile) {
          return {
            ...fileObj,
            status: uploadedFile.success ? 'success' : 'error',
            progress: 100,
            error: uploadedFile.error || null
          };
        }
        return { ...fileObj, status: 'error', error: 'Upload result not found' };
      }));

      // Refresh existing documents if the manager is open
      if (showDocumentManager) {
        await loadExistingDocuments();
      }

    } catch (error) {
      console.error('Upload failed:', error);
      setFiles(prevFiles => prevFiles.map(f => ({ 
        ...f, 
        status: 'error', 
        error: error.message 
      })));
    } finally {
      setUploading(false);
    }
  };

  const hasFiles = files.length > 0;
  const hasSuccessfulUploads = files.some(f => f.status === 'success');
  const hasErrors = files.some(f => f.status === 'error');

  return (
    <>
      <Head>
        <title>Upload Documents - MaiStorage</title>
        <meta name="description" content="Upload documents to your MaiStorage knowledge base for intelligent search and Q&A" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      
      <div className="min-h-screen bg-background text-foreground">
        {/* Header */}
        <motion.header 
          className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="container mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <motion.div 
                className="flex items-center gap-3"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
              >
                <Upload className="h-6 w-6 text-primary" />
                <div>
                  <h1 className="text-lg font-semibold">MaiStorage</h1>
                  <p className="text-sm text-muted-foreground">Document Upload & Ingestion</p>
                </div>
              </motion.div>
              
              <motion.div 
                className="flex items-center gap-2"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
              >
                {/* Dark mode toggle */}
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={toggleDarkMode}
                  >
                    {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
                  </Button>
                </motion.div>
                
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Link href="/">
                    <Button variant="outline" size="sm" className="gap-2">
                      <ArrowLeft className="h-4 w-4" />
                      Back to Home
                    </Button>
                  </Link>
                </motion.div>
              </motion.div>
            </div>
          </div>
        </motion.header>

        {/* Main Content */}
        <motion.main 
          className="container mx-auto px-4 py-8"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          <div className="max-w-4xl mx-auto">
            {/* Page Title */}
            <div className="text-center mb-12">
              <motion.div 
                className="flex justify-center mb-6"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.6, delay: 0.3 }}
              >
                <div className="p-3 bg-primary/10 rounded-full">
                  <Upload className="h-12 w-12 text-primary" />
                </div>
              </motion.div>
              
              <motion.h1 
                className="text-4xl font-bold tracking-tight mb-4"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.4 }}
              >
                Upload Documents
              </motion.h1>
              <motion.p 
                className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.5 }}
              >
                Add documents to your knowledge base for intelligent search and Q&A
              </motion.p>
              <motion.div 
                className="flex gap-4 justify-center mb-12"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.6 }}
              >
                <Badge variant="success" className="gap-1">
                  <CheckCircle className="h-3 w-3" />
                  PDF
                </Badge>
                <Badge variant="info">Markdown</Badge>
                <Badge variant="outline">HTML</Badge>
                <Badge variant="outline">TXT</Badge>
              </motion.div>
            </div>

            {/* Upload Area */}
            <motion.div
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.7 }}
            >
              <div className="rounded-lg border bg-card p-6">
                <div className="mb-6">
                  <h3 className="text-lg font-semibold flex items-center gap-2 mb-2">
                    <Upload className="h-5 w-5" />
                    File Upload
                  </h3>
                  <p className="text-muted-foreground">
                    Drag and drop files here or click to browse. Supported formats: PDF, Markdown, HTML, and TXT.
                  </p>
                </div>
                <div>
                  {/* Drop Zone */}
                  <motion.div
                    className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                      dragActive
                        ? 'border-primary bg-primary/5'
                        : 'border-muted-foreground/25 hover:border-muted-foreground/50'
                    }`}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.99 }}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      multiple
                      accept=".pdf,.md,.markdown,.html,.htm,.txt"
                      onChange={handleFileInputChange}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    />
                    
                    <div className="space-y-4">
                      <div className="flex justify-center">
                        <Upload className={`h-12 w-12 ${dragActive ? 'text-primary' : 'text-muted-foreground'}`} />
                      </div>
                      <div>
                        <p className="text-lg font-medium">
                          {dragActive ? 'Drop files here' : 'Drag & drop files here'}
                        </p>
                        <p className="text-muted-foreground">
                          or <span className="text-primary font-medium">click to browse</span>
                        </p>
                      </div>
                    </div>
                  </motion.div>

                  {/* File List */}
                  <AnimatePresence>
                    {hasFiles && (
                      <motion.div
                        className="mt-6 space-y-3"
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.3 }}
                      >
                        <div className="flex items-center justify-between">
                          <h3 className="font-medium">Selected Files ({files.length})</h3>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={clearAllFiles}
                            disabled={uploading}
                          >
                            <Trash2 className="h-4 w-4 mr-2" />
                            Clear All
                          </Button>
                        </div>

                        <div className="space-y-2">
                          {files.map((fileObj) => (
                            <motion.div
                              key={fileObj.id}
                              className="flex items-center justify-between p-3 bg-muted/30 rounded-lg"
                              initial={{ opacity: 0, x: -20 }}
                              animate={{ opacity: 1, x: 0 }}
                              exit={{ opacity: 0, x: 20 }}
                              transition={{ duration: 0.2 }}
                            >
                              <div className="flex items-center gap-3 flex-1 min-w-0">
                                {getFileIcon(fileObj.file.name)}
                                <div className="flex-1 min-w-0">
                                  <p className="font-medium truncate">{fileObj.file.name}</p>
                                  <p className="text-sm text-muted-foreground">
                                    {formatFileSize(fileObj.file.size)}
                                  </p>
                                </div>
                              </div>

                              <div className="flex items-center gap-2">
                                {/* Status Icon */}
                                {fileObj.status === 'ready' && (
                                  <CheckCircle className="h-5 w-5 text-muted-foreground" />
                                )}
                                {fileObj.status === 'uploading' && (
                                  <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />
                                )}
                                {fileObj.status === 'success' && (
                                  <CheckCircle className="h-5 w-5 text-green-500" />
                                )}
                                {fileObj.status === 'error' && (
                                  <XCircle className="h-5 w-5 text-red-500" />
                                )}

                                {/* Remove Button */}
                                {!uploading && (
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => removeFile(fileObj.id)}
                                  >
                                    <XCircle className="h-4 w-4" />
                                  </Button>
                                )}
                              </div>
                            </motion.div>
                          ))}
                        </div>

                        {/* Upload Button */}
                        <div className="flex justify-center pt-4">
                          <Button
                            onClick={uploadFiles}
                            disabled={uploading || files.length === 0}
                            size="lg"
                            className="gap-2"
                          >
                            {uploading ? (
                              <>
                                <Loader2 className="h-5 w-5 animate-spin" />
                                Uploading...
                              </>
                            ) : (
                              <>
                                <Database className="h-5 w-5" />
                                Upload & Ingest ({files.length} files)
                              </>
                            )}
                          </Button>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </div>
            </motion.div>

            {/* Upload Results */}
            <AnimatePresence>
              {uploadResults && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.5 }}
                >
                  <div className="rounded-lg border bg-card p-6">
                    <div className="mb-6">
                      <h3 className="text-lg font-semibold flex items-center gap-2 mb-2">
                        {hasErrors ? (
                          <AlertCircle className="h-5 w-5 text-orange-500" />
                        ) : (
                          <CheckCircle className="h-5 w-5 text-green-500" />
                        )}
                        Upload Results
                      </h3>
                      <p className="text-muted-foreground">
                        {hasSuccessfulUploads && (
                          <span className="text-green-600">
                            {files.filter(f => f.status === 'success').length} files uploaded successfully
                          </span>
                        )}
                        {hasSuccessfulUploads && hasErrors && ' • '}
                        {hasErrors && (
                          <span className="text-red-600">
                            {files.filter(f => f.status === 'error').length} files failed
                          </span>
                        )}
                      </p>
                    </div>
                    <div>
                      {uploadResults.summary && (
                        <div className="space-y-2 text-sm">
                          <p><strong>Total Chunks Created:</strong> {uploadResults.summary.total_chunks}</p>
                          <p><strong>Processing Time:</strong> {uploadResults.summary.processing_time_ms}ms</p>
                          {uploadResults.summary.collection_info && (
                            <p><strong>Total Documents in Collection:</strong> {uploadResults.summary.collection_info.points_count}</p>
                          )}
                        </div>
                      )}

                      {hasSuccessfulUploads && (
                        <div className="mt-4 pt-4 border-t">
                          <p className="text-sm text-muted-foreground mb-3">
                            Your documents have been successfully processed and are now available for search.
                          </p>
                          <div className="flex gap-2">
                            <Link href="/chat">
                              <Button size="sm">
                                Try Chat Interface
                              </Button>
                            </Link>
                            <Link href="/eval">
                              <Button variant="outline" size="sm">
                                Run Evaluation
                              </Button>
                            </Link>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Document Management Section */}
            <motion.div
              className="mt-12"
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.8 }}
            >
              <div className="rounded-lg border bg-card p-6">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h3 className="text-lg font-semibold flex items-center gap-2 mb-2">
                      <HardDriveIcon className="h-5 w-5" />
                      Document Management
                    </h3>
                    <p className="text-muted-foreground">
                      View and manage documents in your knowledge base
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        setShowDocumentManager(!showDocumentManager);
                        if (!showDocumentManager) {
                          loadExistingDocuments();
                        }
                      }}
                      disabled={loadingDocuments}
                    >
                      {loadingDocuments ? (
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <FolderOpen className="h-4 w-4 mr-2" />
                      )}
                      {showDocumentManager ? 'Hide Documents' : 'View Documents'}
                    </Button>
                  </div>
                </div>

                <AnimatePresence>
                  {showDocumentManager && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      {/* Document List Header */}
                      <div className="flex items-center justify-between mb-4 pb-2 border-b">
                        <div className="flex items-center gap-2">
                          <h4 className="font-medium">
                            Stored Documents ({existingDocuments.length})
                          </h4>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={loadExistingDocuments}
                            disabled={loadingDocuments}
                          >
                            <RefreshCw className={`h-4 w-4 ${loadingDocuments ? 'animate-spin' : ''}`} />
                          </Button>
                        </div>
                        
                        {existingDocuments.length > 0 && (
                          <Button
                            variant="destructive"
                            size="sm"
                            onClick={resetCollection}
                            disabled={resettingCollection || loadingDocuments}
                          >
                            {resettingCollection ? (
                              <>
                                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                Resetting...
                              </>
                            ) : (
                              <>
                                <Trash2 className="h-4 w-4 mr-2" />
                                Reset All
                              </>
                            )}
                          </Button>
                        )}
                      </div>

                      {/* Document List */}
                      {loadingDocuments ? (
                        <div className="flex items-center justify-center py-8">
                          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                          <span className="ml-2 text-muted-foreground">Loading documents...</span>
                        </div>
                      ) : existingDocuments.length === 0 ? (
                        <div className="text-center py-8">
                          <Database className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                          <p className="text-muted-foreground">No documents found in the knowledge base</p>
                          <p className="text-sm text-muted-foreground mt-1">Upload some documents to get started</p>
                        </div>
                      ) : (
                        <div className="space-y-2">
                          {existingDocuments.map((doc) => (
                            <motion.div
                              key={doc.doc_name}
                              className="flex items-center justify-between p-3 bg-muted/30 rounded-lg"
                              initial={{ opacity: 0, x: -20 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ duration: 0.2 }}
                            >
                              <div className="flex items-center gap-3 flex-1 min-w-0">
                                {getFileIcon(doc.doc_name)}
                                <div className="flex-1 min-w-0">
                                  <p className="font-medium truncate">{doc.doc_name}</p>
                                  <div className="flex items-center gap-4 text-sm text-muted-foreground">
                                    {doc.doc_type && (
                                      <span>{doc.doc_type.toUpperCase()}</span>
                                    )}
                                    {doc.total_chunks && (
                                      <span>{doc.total_chunks} chunks</span>
                                    )}
                                    {doc.timestamp && (
                                      <span className="flex items-center gap-1">
                                        <Clock className="h-3 w-3" />
                                        {new Date(doc.timestamp).toLocaleDateString()}
                                      </span>
                                    )}
                                  </div>
                                </div>
                              </div>

                              <div className="flex items-center gap-2">
                                <Button
                                  variant="destructive"
                                  size="sm"
                                  onClick={() => deleteDocument(doc.doc_name)}
                                  disabled={deletingDocument === doc.doc_name || loadingDocuments}
                                >
                                  {deletingDocument === doc.doc_name ? (
                                    <>
                                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                      Deleting...
                                    </>
                                  ) : (
                                    <>
                                      <Trash2 className="h-4 w-4 mr-2" />
                                      Delete
                                    </>
                                  )}
                                </Button>
                              </div>
                            </motion.div>
                          ))}
                        </div>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>

            {/* Info Cards */}
            <motion.div 
              className="grid md:grid-cols-2 gap-6 mt-12"
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.8 }}
            >
              <div className="rounded-lg border bg-card p-6">
                <div className="mb-6">
                  <h3 className="text-lg font-semibold mb-2">Processing Pipeline</h3>
                </div>
                <div className="space-y-3 text-sm">
                  <div className="flex items-start gap-3">
                    <div className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                      <span className="text-xs font-medium text-primary">1</span>
                    </div>
                    <div>
                      <p className="font-medium">Document Loading</p>
                      <p className="text-muted-foreground">Files are loaded and parsed based on format</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                      <span className="text-xs font-medium text-primary">2</span>
                    </div>
                    <div>
                      <p className="font-medium">Semantic Chunking</p>
                      <p className="text-muted-foreground">Text is split into 200-500 token chunks with overlap</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                      <span className="text-xs font-medium text-primary">3</span>
                    </div>
                    <div>
                      <p className="font-medium">Vector Embedding</p>
                      <p className="text-muted-foreground">Chunks are embedded using OpenAI text-embedding-3-small</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                      <span className="text-xs font-medium text-primary">4</span>
                    </div>
                    <div>
                      <p className="font-medium">Storage</p>
                      <p className="text-muted-foreground">Vectors and metadata stored in Qdrant collection</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="rounded-lg border bg-card p-6">
                <div className="mb-6">
                  <h3 className="text-lg font-semibold mb-2">Best Practices</h3>
                </div>
                <div className="space-y-3 text-sm">
                  <div className="space-y-2">
                    <p className="font-medium">📄 Document Quality</p>
                    <p className="text-muted-foreground">Use well-structured documents with clear headings and content</p>
                  </div>
                  <div className="space-y-2">
                    <p className="font-medium">📚 Content Diversity</p>
                    <p className="text-muted-foreground">Upload diverse content to improve retrieval coverage</p>
                  </div>
                  <div className="space-y-2">
                    <p className="font-medium">🔍 File Names</p>
                    <p className="text-muted-foreground">Use descriptive file names for better source attribution</p>
                  </div>
                  <div className="space-y-2">
                    <p className="font-medium">⚡ Performance</p>
                    <p className="text-muted-foreground">Upload in batches for better processing efficiency</p>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </motion.main>


      </div>
    </>
  );
}
