import Head from "next/head";
import Link from "next/link";
import { Button } from "../src/components/ui/button";
import { MessageSquare, Database, Zap, Shield } from "lucide-react";

export default function Home() {
  return (
    <>
      <Head>
        <title>MaiStorage - Agentic RAG System</title>
        <meta name="description" content="Advanced document search and Q&A powered by agentic RAG technology" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      
      <div className="min-h-screen bg-background text-foreground">
        {/* Hero Section */}
        <main className="container mx-auto px-4 py-12">
          <div className="text-center mb-12">
            <div className="flex justify-center mb-6">
              <div className="p-3 bg-primary/10 rounded-full">
                <Database className="h-12 w-12 text-primary" />
              </div>
            </div>
            
            <h1 className="text-4xl font-bold tracking-tight mb-4">
              MaiStorage - Question 1 Agentic RAG
            </h1>
            <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
              Advanced Agentic RAG System
            </p>
            <p className="text-lg text-muted-foreground mb-12 max-w-3xl mx-auto">
              Experience intelligent document search with streaming responses, 
              hybrid retrieval, and citation support powered by Next.js and FastAPI.
            </p>

            <div className="flex gap-4 justify-center mb-16">
              <Link href="/chat">
                <Button size="lg" className="gap-2">
                  <MessageSquare className="h-5 w-5" />
                  Try Chat Interface
                </Button>
              </Link>
              <Button variant="outline" size="lg" asChild>
                <a href="http://localhost:8000" target="_blank" rel="noopener noreferrer">
                  API Documentation
                </a>
              </Button>
            </div>
          </div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-3 gap-8 mb-16">
            <div className="text-center p-6 rounded-lg border bg-card">
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Zap className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Streaming Responses</h3>
              <p className="text-muted-foreground">
                Real-time token streaming with NDJSON parsing for immediate feedback
              </p>
            </div>

            <div className="text-center p-6 rounded-lg border bg-card">
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Database className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Hybrid Retrieval</h3>
              <p className="text-muted-foreground">
                Dense + BM25 search with cross-encoder reranking and MMR diversity
              </p>
            </div>

            <div className="text-center p-6 rounded-lg border bg-card">
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Shield className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Source Citations</h3>
              <p className="text-muted-foreground">
                Automatic chunk-level citations with relevance scores and metadata
              </p>
            </div>
          </div>

          {/* Status Section */}
          <div className="bg-muted rounded-lg p-6 mb-8">
            <h2 className="text-lg font-semibold mb-4">System Status</h2>
            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-sm">Phase 4 Complete</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-sm">Streaming UI</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-sm">API Proxy</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-sm">Dark Mode</span>
              </div>
            </div>
          </div>

          {/* Next Steps */}
          <div className="text-center">
            <h2 className="text-2xl font-semibold mb-4">What's Next?</h2>
            <p className="text-muted-foreground mb-6">
              Upcoming phases will add agentic loops with LangGraph, 
              sentence-level citations, and comprehensive evaluation metrics.
            </p>
            <p className="text-sm text-muted-foreground">
              Current implementation provides baseline RAG with chunk-level citations.
              <br />
              Phase 5 will introduce multi-step agentic reasoning with planner and verifier.
            </p>
          </div>
        </main>

        {/* Footer */}
        <footer className="border-t py-6 text-center text-sm text-muted-foreground">
          <p>
            MAI Storage Agentic RAG System - Built with Next.js, FastAPI, and shadcn/ui
          </p>
        </footer>
      </div>
    </>
  );
}