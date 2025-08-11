import React from 'react'
import { Badge } from './ui/badge'
import { 
  Brain, 
  Search, 
  PenTool, 
  CheckCircle, 
  Clock, 
  AlertCircle, 
  Zap,
  Target
} from 'lucide-react'

/**
 * Get icon for agent step
 */
function getStepIcon(step) {
  const iconMap = {
    planner: Brain,
    retriever: Search,
    synthesizer: PenTool,
    verifier: CheckCircle,
    baseline_rag: Target
  }
  
  const Icon = iconMap[step] || Zap
  return <Icon className="h-4 w-4" />
}

/**
 * Get color variant for step status
 */
function getStepVariant(status, hasError = false) {
  if (hasError) return 'destructive'
  
  switch (status) {
    case 'completed':
    case 'step_complete':
      return 'success'
    case 'in_progress':
    case 'step_start':
      return 'info'
    case 'pending':
      return 'secondary'
    default:
      return 'secondary'
  }
}

/**
 * Format timing information
 */
function formatTiming(timeMs) {
  if (!timeMs) return null
  
  if (timeMs < 1000) {
    return `${Math.round(timeMs)}ms`
  } else {
    return `${(timeMs / 1000).toFixed(1)}s`
  }
}

/**
 * Individual trace step component
 */
export function TraceStep({ event, isLast = false }) {
  const { type, step, timestamp, data = {} } = event
  const isComplete = type === 'step_complete'
  const isStart = type === 'step_start'
  const hasError = type === 'error'
  
  const stepName = step || data.step || 'Unknown Step'
  const variant = getStepVariant(type, hasError)
  
  return (
    <div className="flex items-start gap-3 pb-3">
      {/* Timeline line */}
      <div className="flex flex-col items-center">
        <div className={`
          rounded-full p-2 border-2 
          ${variant === 'success' ? 'bg-green-100 border-green-500 text-green-600' : ''}
          ${variant === 'info' ? 'bg-blue-100 border-blue-500 text-blue-600' : ''}
          ${variant === 'destructive' ? 'bg-red-100 border-red-500 text-red-600' : ''}
          ${variant === 'secondary' ? 'bg-gray-100 border-gray-300 text-gray-600' : ''}
          dark:bg-opacity-20
        `}>
          {getStepIcon(stepName)}
        </div>
        
        {!isLast && (
          <div className="w-0.5 h-8 bg-border mt-1" />
        )}
      </div>

      {/* Step content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="font-medium text-sm capitalize">
            {stepName.replace('_', ' ')}
          </span>
          
          <Badge variant={variant} className="text-xs">
            {isStart ? 'Started' : isComplete ? 'Completed' : hasError ? 'Error' : 'In Progress'}
          </Badge>
          
          {data.time_ms && (
            <Badge variant="outline" className="text-xs">
              <Clock className="h-3 w-3 mr-1" />
              {formatTiming(data.time_ms)}
            </Badge>
          )}
        </div>

        {/* Step details */}
        {data.query && (
          <div className="text-xs text-muted-foreground mb-1">
            Query: {data.query.substring(0, 80)}...
          </div>
        )}
        
        {data.plan && (
          <div className="text-xs text-muted-foreground mb-1">
            Plan: {data.plan.substring(0, 100)}...
          </div>
        )}
        
        {data.sources && (
          <div className="text-xs text-muted-foreground mb-1">
            Found {data.sources.length} sources
          </div>
        )}
        
        {data.verification_passed !== undefined && (
          <div className="text-xs flex items-center gap-1">
            {data.verification_passed ? (
              <>
                <CheckCircle className="h-3 w-3 text-green-500" />
                <span className="text-green-600">Verification passed</span>
              </>
            ) : (
              <>
                <AlertCircle className="h-3 w-3 text-amber-500" />
                <span className="text-amber-600">Needs refinement</span>
              </>
            )}
          </div>
        )}
        
        {hasError && data.error && (
          <div className="text-xs text-red-600 mt-1">
            Error: {data.error}
          </div>
        )}

        {/* Timestamp */}
        <div className="text-xs text-muted-foreground mt-1">
          {new Date(timestamp).toLocaleTimeString()}
        </div>
      </div>
    </div>
  )
}

/**
 * Complete agent trace panel
 */
export function AgentTrace({ 
  traceEvents = [], 
  totalTime,
  refinementCount = 0,
  isVisible = true,
  className = ""
}) {
  if (!isVisible || traceEvents.length === 0) {
    return null
  }

  // Filter and organize trace events
  const stepEvents = traceEvents.filter(event => 
    ['step_start', 'step_complete', 'error'].includes(event.type)
  )

  return (
    <div className={`bg-card rounded-lg border p-4 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Brain className="h-5 w-5 text-primary" />
          <h3 className="font-semibold">Agent Trace</h3>
        </div>
        
        <div className="flex items-center gap-2">
          {totalTime && (
            <Badge variant="outline" className="text-xs">
              <Clock className="h-3 w-3 mr-1" />
              {formatTiming(totalTime)}
            </Badge>
          )}
          
          {refinementCount > 0 && (
            <Badge variant="info" className="text-xs">
              {refinementCount} refinement{refinementCount !== 1 ? 's' : ''}
            </Badge>
          )}
        </div>
      </div>

      {/* Timeline */}
      <div className="space-y-0">
        {stepEvents.map((event, index) => (
          <TraceStep
            key={`${event.timestamp}-${index}`}
            event={event}
            isLast={index === stepEvents.length - 1}
          />
        ))}
      </div>

      {/* Summary */}
      {stepEvents.length > 0 && (
        <div className="mt-4 pt-3 border-t text-xs text-muted-foreground">
          <div className="flex justify-between items-center">
            <span>
              Completed {stepEvents.filter(e => e.type === 'step_complete').length} of {Math.ceil(stepEvents.length / 2)} steps
            </span>
            
            <span>
              {stepEvents.length} total events
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

/**
 * Compact agent trace for inline display
 */
export function CompactAgentTrace({ 
  traceEvents = [], 
  totalTime,
  className = ""
}) {
  const stepEvents = traceEvents.filter(event => 
    ['step_start', 'step_complete'].includes(event.type)
  )
  
  const completedSteps = stepEvents.filter(e => e.type === 'step_complete')
  
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <div className="flex items-center gap-1">
        {['planner', 'retriever', 'synthesizer', 'verifier'].map(step => {
          const isCompleted = completedSteps.some(e => e.step === step)
          return (
            <div
              key={step}
              className={`
                w-2 h-2 rounded-full
                ${isCompleted ? 'bg-green-500' : 'bg-gray-300'}
              `}
              title={`${step} ${isCompleted ? 'completed' : 'pending'}`}
            />
          )
        })}
      </div>
      
      {totalTime && (
        <Badge variant="outline" className="text-xs">
          {formatTiming(totalTime)}
        </Badge>
      )}
    </div>
  )
}
