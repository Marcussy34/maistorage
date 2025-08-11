import React, { useEffect, useState } from 'react'
import Head from 'next/head'
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
  Upload
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

  // Fetch evaluation results on component mount
  useEffect(() => {
    fetchEvaluationResults()
    fetchComparisonData()
  }, [])

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
    if (improvement > 0) return 'text-green-600'
    if (improvement < 0) return 'text-red-600'
    return 'text-gray-600'
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-4 text-blue-500" />
          <p className="text-gray-600 dark:text-gray-400">Loading evaluation results...</p>
        </div>
      </div>
    )
  }

  return (
    <>
      <Head>
        <title>Evaluation Dashboard - MAI Storage</title>
        <meta name="description" content="RAG system evaluation metrics and comparison dashboard" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center justify-between">
              <div>
                <Link href="/chat" className="inline-flex items-center text-blue-600 hover:text-blue-800 mb-4">
                  <ArrowLeft className="h-4 w-4 mr-2" />
                  Back to Chat
                </Link>
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                  Evaluation Dashboard
                </h1>
                <p className="text-gray-600 dark:text-gray-400 mt-2">
                  RAGAS metrics and performance comparison between Traditional and Agentic RAG
                </p>
              </div>
              <div className="flex flex-wrap gap-3">
                <Link href="/upload">
                  <Button variant="outline" className="flex items-center">
                    <Upload className="h-4 w-4 mr-2" />
                    Upload Docs
                  </Button>
                </Link>
                <Button
                  onClick={() => runEvaluation('traditional')}
                  disabled={isRunningEvaluation}
                  variant="outline"
                  className="flex items-center"
                >
                  {isRunningEvaluation && lastEvaluationMode === 'traditional' ? (
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <PlayCircle className="h-4 w-4 mr-2" />
                  )}
                  Run Traditional
                </Button>
                <Button
                  onClick={() => runEvaluation('agentic')}
                  disabled={isRunningEvaluation}
                  variant="outline" 
                  className="flex items-center"
                >
                  {isRunningEvaluation && lastEvaluationMode === 'agentic' ? (
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <PlayCircle className="h-4 w-4 mr-2" />
                  )}
                  Run Agentic
                </Button>
                <Button
                  onClick={() => runEvaluation('both')}
                  disabled={isRunningEvaluation}
                  className="flex items-center"
                >
                  {isRunningEvaluation && lastEvaluationMode === 'both' ? (
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <BarChart3 className="h-4 w-4 mr-2" />
                  )}
                  Run Both
                </Button>
              </div>
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
              <div className="flex items-center">
                <XCircle className="h-5 w-5 text-red-500 mr-2" />
                <p className="text-red-700">{error}</p>
              </div>
            </div>
          )}

          {/* Running Evaluation Status */}
          {isRunningEvaluation && (
            <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="flex items-center">
                <RefreshCw className="h-5 w-5 text-blue-500 mr-2 animate-spin" />
                <p className="text-blue-700">
                  Running {lastEvaluationMode} evaluation... This may take a few minutes.
                </p>
              </div>
            </div>
          )}

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
                        <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                          Total Evaluations
                        </p>
                        <p className="text-2xl font-bold text-gray-900 dark:text-white">
                          {evaluationResults?.total_files || 0}
                        </p>
                      </div>
                      <BarChart3 className="h-8 w-8 text-blue-500" />
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                          Questions Tested
                        </p>
                        <p className="text-2xl font-bold text-gray-900 dark:text-white">
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
                        <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                          Comparison Available
                        </p>
                        <p className="text-2xl font-bold text-gray-900 dark:text-white">
                          {comparisonData ? 'Yes' : 'No'}
                        </p>
                      </div>
                      {comparisonData ? (
                        <CheckCircle className="h-8 w-8 text-green-500" />
                      ) : (
                        <XCircle className="h-8 w-8 text-gray-400" />
                      )}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                          Last Updated
                        </p>
                        <p className="text-sm font-medium text-gray-900 dark:text-white">
                          {evaluationResults?.results?.[0]?.metadata?.evaluation_timestamp 
                            ? new Date(evaluationResults.results[0].metadata.evaluation_timestamp).toLocaleDateString()
                            : 'Never'
                          }
                        </p>
                      </div>
                      <Clock className="h-8 w-8 text-purple-500" />
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Comparison Summary */}
              {comparisonData && (
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
                              <p className="text-sm text-gray-600 dark:text-gray-400 capitalize">
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
                              <p className="text-sm text-gray-600 dark:text-gray-400 capitalize">
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
                              : getImprovementIcon(data.improvement)
                            }
                            <span className={`ml-2 font-medium ${
                              metric === 'avg_response_time_ms' 
                                ? getImprovementColor(-data.improvement)
                                : getImprovementColor(data.improvement)
                            }`}>
                              {metric.includes('time') 
                                ? (data.improvement > 0 ? '+' : '') + data.improvement.toFixed(0) + 'ms'
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
                <BarChart3 className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                  No Evaluation Results
                </h3>
                <p className="text-gray-600 dark:text-gray-400 mb-6">
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
        </div>
      </div>
    </>
  )
}
