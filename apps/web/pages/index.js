import React, { useEffect, useState, useRef } from 'react';
import Head from "next/head";
import Link from "next/link";
import { motion } from "framer-motion";
import { Button } from "../src/components/ui/button";
import { MessageSquare, Database, Zap, Shield, Moon, Sun } from "lucide-react";

export default function Home() {
  const [darkMode, setDarkMode] = useState(false)
  const mountedRef = useRef(false)

  // Initialize dark mode from localStorage or system preference
  useEffect(() => {
    if (mountedRef.current) return // Prevent double execution in StrictMode
    mountedRef.current = true
    
    const savedDarkMode = localStorage.getItem('darkMode')
    const systemDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches
    
    const shouldUseDarkMode = savedDarkMode 
      ? JSON.parse(savedDarkMode) 
      : systemDarkMode

    setDarkMode(shouldUseDarkMode)
    updateDarkModeClass(shouldUseDarkMode)
  }, [])

  // Update dark mode class on document
  const updateDarkModeClass = (isDark) => {
    if (isDark) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }

  // Toggle dark mode
  const toggleDarkMode = () => {
    const newDarkMode = !darkMode
    setDarkMode(newDarkMode)
    localStorage.setItem('darkMode', JSON.stringify(newDarkMode))
    updateDarkModeClass(newDarkMode)
  }

  return (
    <>
      <Head>
        <title>MaiStorage - Agentic RAG System</title>
        <meta name="description" content="Advanced document search and Q&A powered by agentic RAG technology" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      
      <div className="h-screen bg-background text-foreground overflow-hidden flex flex-col">
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
                <Database className="h-6 w-6 text-primary" />
                <div>
                  <h1 className="text-lg font-semibold">MaiStorage</h1>
                  <p className="text-sm text-muted-foreground">Agentic RAG System - Built with Next.js, FastAPI, and shadcn/ui</p>
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
              </motion.div>
            </div>
          </div>
        </motion.header>

        {/* Hero Section */}
        <motion.main 
          className="container mx-auto px-4 py-8 flex-1 flex flex-col justify-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          <div className="text-center mb-8">
            <motion.div 
              className="flex justify-center mb-4"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.6, delay: 0.3 }}
            >
              <div className="p-3 bg-primary/10 rounded-full">
                <Database className="h-12 w-12 text-primary" />
              </div>
            </motion.div>
            
            <motion.h1 
              className="text-4xl font-bold tracking-tight mb-3"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              MaiStorage - Question 1 Agentic RAG
            </motion.h1>
            <motion.p 
              className="text-xl text-muted-foreground mb-4 max-w-2xl mx-auto"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.5 }}
            >
              Advanced Agentic RAG System
            </motion.p>
            <motion.p 
              className="text-lg text-muted-foreground mb-8 max-w-3xl mx-auto"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.6 }}
            >
              Experience intelligent document search with streaming responses, 
              hybrid retrieval, and citation support powered by Next.js and FastAPI.
            </motion.p>

            <motion.div 
              className="flex flex-wrap gap-4 justify-center mb-8"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.7 }}
            >
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Link href="/chat">
                  <Button size="lg" className="gap-2">
                    <MessageSquare className="h-5 w-5" />
                    Try Chat Interface
                  </Button>
                </Link>
              </motion.div>
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Link href="/upload">
                  <Button variant="outline" size="lg" className="gap-2">
                    <Database className="h-5 w-5" />
                    Upload Documents
                  </Button>
                </Link>
              </motion.div>
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Link href="/api-docs">
                  <Button variant="outline" size="lg">
                    API Documentation
                  </Button>
                </Link>
              </motion.div>
            </motion.div>
          </div>

          {/* Features Grid */}
          <motion.div 
            className="grid md:grid-cols-3 gap-8 mb-16"
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.8 }}
          >
            <motion.div 
              className="text-center p-6 rounded-lg border bg-card"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.9 }}
              whileHover={{ y: -5, transition: { duration: 0.2 } }}
            >
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Zap className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Streaming Responses</h3>
              <p className="text-muted-foreground">
                Real-time token streaming with NDJSON parsing for immediate feedback
              </p>
            </motion.div>

            <motion.div 
              className="text-center p-6 rounded-lg border bg-card"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 1.0 }}
              whileHover={{ y: -5, transition: { duration: 0.2 } }}
            >
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Database className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Hybrid Retrieval</h3>
              <p className="text-muted-foreground">
                Dense + BM25 search with cross-encoder reranking and MMR diversity
              </p>
            </motion.div>

            <motion.div 
              className="text-center p-6 rounded-lg border bg-card"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 1.1 }}
              whileHover={{ y: -5, transition: { duration: 0.2 } }}
            >
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Shield className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Source Citations</h3>
              <p className="text-muted-foreground">
                Automatic chunk-level citations with relevance scores and metadata
              </p>
            </motion.div>
          </motion.div>
        </motion.main>
      </div>
    </>
  );
}