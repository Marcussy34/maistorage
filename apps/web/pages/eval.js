import React, { useEffect, useState, useRef } from 'react'
import Head from 'next/head'
import { motion } from 'framer-motion'
import { Button } from '../src/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../src/components/ui/tabs'
import { Badge } from '../src/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../src/components/ui/card'
import { 
  BarChart3, 
  TrendingUp, 
  TrendingDown, 
  Clock, 
  Zap, 
  CheckCircle, 
  XCircle, 
  PlayCircle,
  RefreshCw,
  Download,
  ArrowLeft,
  Upload,
  Moon,
  Sun
} from 'lucide-react'
import Link from 'next/link'

/**
 * Phase 8: Evaluation Dashboard
 * 
 * Displays RAGAS metrics, retrieval metrics, and performance comparisons
 * between Traditional and Agentic RAG systems.
 */
export default function EvaluationPage() {
  const [evaluationResults, setEvaluationResults] = useState(null)
  const [comparisonData, setComparisonData] = useState(null)
  const [isRunningEvaluation, setIsRunningEvaluation] = useState(false)
  const [lastEvaluationMode, setLastEvaluationMode] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
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

  // Fetch evaluation results on component mount
  useEffect(() => {
    fetchEvaluationResults()
    fetchComparisonData()
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

  const fetchEvaluationResults = async () => {
    try {
      setLoading(true)
      const response = await fetch('/api/eval/results')
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      const data = await response.json()
      setEvaluationResults(data)
      setError(null)
    } catch (err) {
      console.error('Failed to fetch evaluation results:', err)
      setError(`Failed to load evaluation results: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const fetchComparisonData = async () => {
    try {
      const response = await fetch('/api/eval/compare')
      if (!response.ok) {
        if (response.status === 404) {
          // No comparison data available yet
          return
        }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      const data = await response.json()
      setComparisonData(data)
    } catch (err) {
      console.error('Failed to fetch comparison data:', err)
      // Don't set error for comparison data, it's optional
    }
  }

  const runEvaluation = async (mode) => {
    try {
      setIsRunningEvaluation(true)
      setError(null)
      setLastEvaluationMode(mode)

      const response = await fetch('/api/eval/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          mode,
          top_k: 5,
          save_results: true
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      // Refresh results after evaluation completes
      await fetchEvaluationResults()
      await fetchComparisonData()
      
    } catch (err) {
      console.error('Failed to run evaluation:', err)
      setError(`Evaluation failed: ${err.message}`)
    } finally {
      setIsRunningEvaluation(false)
    }
  }

  const formatScore = (score) => {
    if (score === null || score === undefined) return 'N/A'
    return (score * 100).toFixed(1) + '%'
  }

  const formatTime = (timeMs) => {
    if (timeMs === null || timeMs === undefined) return 'N/A'
    return timeMs.toFixed(0) + 'ms'
  }

  const formatNumber = (num) => {
    if (num === null || num === undefined) return 'N/A'
    return num.toFixed(2)
  }

  const getImprovementIcon = (improvement) => {
    if (improvement > 0) return <TrendingUp className="h-4 w-4 text-green-500" />
    if (improvement < 0) return <TrendingDown className="h-4 w-4 text-red-500" />
    return <div className="h-4 w-4" />
  }

  const getImprovementColor = (improvement) => {
    if (improvement > 0) return 'text-green-500'
    if (improvement < 0) return 'text-red-500'
    return 'text-muted-foreground'
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-4 text-primary" />
          <p className="text-muted-foreground">Loading evaluation results...</p>
        </div>
      </div>
    )
  }

  return (
    <>
      <Head>
        <title>Evaluation Dashboard - MaiStorage</title>
        <meta name="description" content="RAG system evaluation metrics and comparison dashboard" />
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
                <BarChart3 className="h-6 w-6 text-primary" />
                <div>
                  <h1 className="text-lg font-semibold">MaiStorage</h1>
                  <p className="text-sm text-muted-foreground">Evaluation Dashboard</p>
                </div>
              </motion.div>
              
              <motion.div 
                className="flex items-center gap-2"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
              >
                {/* Back to Chat button */}
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Link href="/chat">
                    <Button variant="outline" size="sm" className="gap-2">
                      <ArrowLeft className="h-4 w-4" />
                      Back to Chat
                    </Button>
                  </Link>
                </motion.div>

                {/* Upload button */}
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Link href="/upload">
                    <Button variant="outline" size="sm" className="hidden sm:flex items-center gap-2">
                      <Upload className="h-4 w-4" />
                      Upload Docs
                    </Button>
                  </Link>
                </motion.div>

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

        {/* Main Content */}
        <motion.main 
          className="container mx-auto px-4 py-8"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          <div className="max-w-7xl mx-auto">
            {/* Page Title and Action Buttons */}
            <motion.div 
              className="text-center mb-12"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
            >
              <div className="flex justify-center mb-6">
                <div className="p-3 bg-primary/10 rounded-full">
                  <BarChart3 className="h-12 w-12 text-primary" />
                </div>
              </div>
              
              <h1 className="text-4xl font-bold tracking-tight mb-4">
                Evaluation Dashboard
              </h1>
              <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
                RAGAS metrics and performance comparison between Traditional and Agentic RAG
              </p>
              
              {/* Action Buttons */}
              <div className="flex flex-wrap justify-center gap-3 mb-8">
                <Button
                  onClick={() => runEvaluation('traditional')}
                  disabled={isRunningEvaluation}
                  variant="outline"
                  className="flex items-center gap-2"
                >
                  {isRunningEvaluation && lastEvaluationMode === 'traditional' ? (
                    <RefreshCw className="h-4 w-4 animate-spin" />
                  ) : (
                    <PlayCircle className="h-4 w-4" />
                  )}
                  Run Traditional
                </Button>
                <Button
                  onClick={() => runEvaluation('agentic')}
                  disabled={isRunningEvaluation}
                  variant="outline" 
                  className="flex items-center gap-2"
                >
                  {isRunningEvaluation && lastEvaluationMode === 'agentic' ? (
                    <RefreshCw className="h-4 w-4 animate-spin" />
                  ) : (
                    <PlayCircle className="h-4 w-4" />
                  )}
                  Run Agentic
                </Button>
                <Button
                  onClick={() => runEvaluation('both')}
                  disabled={isRunningEvaluation}
                  className="flex items-center gap-2"
                >
                  {isRunningEvaluation && lastEvaluationMode === 'both' ? (
                    <RefreshCw className="h-4 w-4 animate-spin" />
                  ) : (
                    <BarChart3 className="h-4 w-4" />
                  )}
                  Run Both
                </Button>
              </div>
            </motion.div>

            {/* Error Display */}
            {error && (
              <motion.div 
                className="mb-6 p-4 bg-destructive/10 border border-destructive/20 rounded-lg"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <div className="flex items-center">
                  <XCircle className="h-5 w-5 text-destructive mr-2" />
                  <p className="text-destructive">{error}</p>
                </div>
              </motion.div>
            )}

            {/* Running Evaluation Status */}
            {isRunningEvaluation && (
              <motion.div 
                className="mb-6 p-4 bg-primary/10 border border-primary/20 rounded-lg"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <div className="flex items-center">
                  <RefreshCw className="h-5 w-5 text-primary mr-2 animate-spin" />
                  <p className="text-foreground">
                    Running {lastEvaluationMode} evaluation... This may take a few minutes.
                  </p>
                </div>
              </motion.div>
            )}

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              <Tabs defaultValue="overview" className="space-y-6">
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="ragas">RAGAS Metrics</TabsTrigger>
                  <TabsTrigger value="retrieval">Retrieval Metrics</TabsTrigger>
                  <TabsTrigger value="performance">Performance</TabsTrigger>
                </TabsList>

            {/* Overview Tab */}
            <TabsContent value="overview" className="space-y-6">
              {/* Quick Stats */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">
                          Total Evaluations
                        </p>
                        <p className="text-2xl font-bold text-foreground">
                          {evaluationResults?.total_files || 0}
                        </p>
                      </div>
                      <BarChart3 className="h-8 w-8 text-primary" />
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">
                          Questions Tested
                        </p>
                        <p className="text-2xl font-bold text-foreground">
                          18
                        </p>
                      </div>
                      <CheckCircle className="h-8 w-8 text-green-500" />
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">
                          Comparison Available
                        </p>
                        <p className="text-2xl font-bold text-foreground">
                          {comparisonData ? 'Yes' : 'No'}
                        </p>
                      </div>
                      {comparisonData ? (
                        <CheckCircle className="h-8 w-8 text-green-500" />
                      ) : (
                        <XCircle className="h-8 w-8 text-muted-foreground" />
                      )}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">
                          Last Updated
                        </p>
                        <p className="text-sm font-medium text-foreground">
                          {evaluationResults?.results?.[0]?.metadata?.evaluation_timestamp 
                            ? new Date(evaluationResults.results[0].metadata.evaluation_timestamp).toLocaleDateString()
                            : 'Never'
                          }
                        </p>
                      </div>
                      <Clock className="h-8 w-8 text-primary" />
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Comparison Summary */}
              {comparisonData && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.5 }}
                >
                  <Card>
                    <CardHeader>
                      <CardTitle>Traditional vs Agentic RAG Comparison</CardTitle>
                      <CardDescription>
                        Key metric differences between the two approaches
                      </CardDescription>
                    </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {/* RAGAS Summary */}
                      <div>
                        <h4 className="font-medium mb-2">RAGAS Metrics</h4>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          {Object.entries(comparisonData.ragas_comparison || {}).map(([metric, data]) => (
                            <div key={metric} className="text-center">
                              <p className="text-sm text-muted-foreground capitalize">
                                {metric.replace('_score', '').replace('_', ' ')}
                              </p>
                              <div className="flex items-center justify-center mt-1">
                                {getImprovementIcon(data.improvement)}
                                <span className={`ml-1 font-medium ${getImprovementColor(data.improvement)}`}>
                                  {data.improvement > 0 ? '+' : ''}{(data.improvement * 100).toFixed(1)}%
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Performance Summary */}
                      <div>
                        <h4 className="font-medium mb-2">Performance Impact</h4>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                          {Object.entries(comparisonData.performance_comparison || {}).map(([metric, data]) => (
                            <div key={metric} className="text-center">
                              <p className="text-sm text-muted-foreground capitalize">
                                {metric.replace('avg_', '').replace('_', ' ')}
                              </p>
                              <div className="flex items-center justify-center mt-1">
                                {getImprovementIcon(metric === 'avg_response_time_ms' ? -data.improvement : data.improvement)}
                                <span className={`ml-1 font-medium ${
                                  metric === 'avg_response_time_ms' 
                                    ? getImprovementColor(-data.improvement)
                                    : getImprovementColor(data.improvement)
                                }`}>
                                  {metric === 'avg_response_time_ms' && data.improvement > 0 ? '+' : ''}
                                  {metric === 'avg_response_time_ms' 
                                    ? data.improvement.toFixed(0) + 'ms'
                                    : (data.improvement_pct > 0 ? '+' : '') + data.improvement_pct.toFixed(1) + '%'
                                  }
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                  </Card>
                </motion.div>
              )}
            </TabsContent>

            {/* RAGAS Metrics Tab */}
            <TabsContent value="ragas" className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {comparisonData && Object.entries(comparisonData.ragas_comparison || {}).map(([metric, data]) => (
                  <Card key={metric}>
                    <CardHeader>
                      <CardTitle className="capitalize">
                        {metric.replace('_score', '').replace('_', ' ')}
                      </CardTitle>
                      <CardDescription>
                        Comparison between Traditional and Agentic RAG
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div className="flex justify-between items-center">
                          <span className="text-sm font-medium">Traditional RAG</span>
                          <Badge variant="outline">{formatScore(data.traditional)}</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm font-medium">Agentic RAG</span>
                          <Badge variant="outline">{formatScore(data.agentic)}</Badge>
                        </div>
                        <div className="flex justify-between items-center pt-2 border-t">
                          <span className="text-sm font-medium">Improvement</span>
                          <div className="flex items-center">
                            {getImprovementIcon(data.improvement)}
                            <span className={`ml-2 font-medium ${getImprovementColor(data.improvement)}`}>
                              {data.improvement > 0 ? '+' : ''}{(data.improvement * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            {/* Retrieval Metrics Tab */}
            <TabsContent value="retrieval" className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {comparisonData && Object.entries(comparisonData.retrieval_comparison || {}).map(([metric, data]) => (
                  <Card key={metric}>
                    <CardHeader>
                      <CardTitle className="capitalize">
                        {metric.replace('_', ' ')}
                      </CardTitle>
                      <CardDescription>
                        Retrieval quality comparison
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div className="flex justify-between items-center">
                          <span className="text-sm font-medium">Traditional RAG</span>
                          <Badge variant="outline">{formatScore(data.traditional)}</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm font-medium">Agentic RAG</span>
                          <Badge variant="outline">{formatScore(data.agentic)}</Badge>
                        </div>
                        <div className="flex justify-between items-center pt-2 border-t">
                          <span className="text-sm font-medium">Improvement</span>
                          <div className="flex items-center">
                            {getImprovementIcon(data.improvement)}
                            <span className={`ml-2 font-medium ${getImprovementColor(data.improvement)}`}>
                              {data.improvement > 0 ? '+' : ''}{(data.improvement * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            {/* Performance Tab */}
            <TabsContent value="performance" className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {comparisonData && Object.entries(comparisonData.performance_comparison || {}).map(([metric, data]) => (
                  <Card key={metric}>
                    <CardHeader>
                      <CardTitle className="capitalize">
                        {metric.replace('avg_', '').replace('_', ' ')}
                      </CardTitle>
                      <CardDescription>
                        Performance comparison
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div className="flex justify-between items-center">
                          <span className="text-sm font-medium">Traditional RAG</span>
                          <Badge variant="outline">
                            {metric.includes('time') ? formatTime(data.traditional) : formatNumber(data.traditional)}
                          </Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm font-medium">Agentic RAG</span>
                          <Badge variant="outline">
                            {metric.includes('time') ? formatTime(data.agentic) : formatNumber(data.agentic)}
                          </Badge>
                        </div>
                        <div className="flex justify-between items-center pt-2 border-t">
                          <span className="text-sm font-medium">
                            {metric === 'avg_response_time_ms' ? 'Latency Change' : 'Improvement'}
                          </span>
                          <div className="flex items-center">
                            {metric === 'avg_response_time_ms' 
                              ? getImprovementIcon(-data.improvement)
                              : metric.includes('token') 
                                ? getImprovementIcon(-data.improvement) // For token usage, negative improvement is positive
                                : getImprovementIcon(data.improvement)
                            }
                            <span className={`ml-2 font-medium ${
                              metric === 'avg_response_time_ms' 
                                ? getImprovementColor(-data.improvement)
                                : metric.includes('token')
                                  ? getImprovementColor(-data.improvement) // For token usage, negative improvement is positive
                                  : getImprovementColor(data.improvement)
                            }`}>
                              {metric.includes('time') 
                                ? (data.improvement > 0 ? '+' : '') + data.improvement.toFixed(0) + 'ms'
                                : metric.includes('token')
                                  ? (data.improvement_pct < 0 ? '+' : '') + Math.abs(data.improvement_pct).toFixed(1) + '%' // Show absolute value for token usage
                                  : (data.improvement_pct > 0 ? '+' : '') + data.improvement_pct.toFixed(1) + '%'
                              }
                            </span>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>
              </Tabs>

              {/* No Data State */}
              {!evaluationResults?.results?.length && !loading && (
                <Card className="text-center py-12">
                  <CardContent>
                    <BarChart3 className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-foreground mb-2">
                      No Evaluation Results
                    </h3>
                    <p className="text-muted-foreground mb-6">
                      Run your first evaluation to see metrics and comparisons.
                    </p>
                    <div className="flex justify-center space-x-3">
                      <Button onClick={() => runEvaluation('traditional')} variant="outline">
                        <PlayCircle className="h-4 w-4 mr-2" />
                        Run Traditional RAG
                      </Button>
                      <Button onClick={() => runEvaluation('agentic')}>
                        <PlayCircle className="h-4 w-4 mr-2" />
                        Run Agentic RAG
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )}
            </motion.div>
          </div>
        </motion.main>
      </div>
    </>
  )
}
