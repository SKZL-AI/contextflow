/**
 * ContextFlow API TypeScript Types
 *
 * Type definitions for the ContextFlow Python API interfaces
 * Based on OpenAPI 3.1 specification
 */

// ============================================================================
// Core Types
// ============================================================================

/**
 * Processing strategy options
 */
export type Strategy = 'auto' | 'gsd' | 'ralph' | 'rlm';

/**
 * API error types for categorization
 */
export type ErrorType =
  | 'validation_error'
  | 'provider_error'
  | 'rate_limit_error'
  | 'token_limit_error'
  | 'configuration_error'
  | 'internal_error'
  | 'not_found_error'
  | 'authentication_error';

/**
 * Health status values
 */
export type HealthStatus = 'healthy' | 'degraded' | 'unhealthy';

// ============================================================================
// Token and Cost Tracking
// ============================================================================

/**
 * Token usage statistics for API calls
 */
export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cost_usd: number;
}

// ============================================================================
// Verification
// ============================================================================

/**
 * Details from answer verification process
 */
export interface VerificationDetails {
  passed: boolean;
  score: number;
  issues: string[];
  suggestions: string[];
}

// ============================================================================
// Trajectory Tracking
// ============================================================================

/**
 * A single step in the processing trajectory
 */
export interface TrajectoryStep {
  step_type: string;
  timestamp: string;
  tokens_used: number;
  cost_usd: number;
  duration_ms: number;
  metadata: Record<string, unknown>;
}

// ============================================================================
// Process API
// ============================================================================

/**
 * Request payload for the /process endpoint
 */
export interface ProcessRequest {
  task: string;
  documents?: string[];
  context?: string;
  strategy?: Strategy | string;
  provider?: string;
  model?: string;
  stream?: boolean;
  constraints?: string[];
  config?: Record<string, unknown>;
  session_id?: string;
  max_tokens?: number;
  temperature?: number;
  verify?: boolean;
}

/**
 * Response from the /process endpoint
 */
export interface ProcessResponse {
  success: boolean;
  answer: string;
  strategy_used: Strategy;
  token_usage: TokenUsage;
  execution_time: number;
  verification_passed?: boolean;
  verification_score?: number;
  verification_details?: VerificationDetails;
  trajectory?: TrajectoryStep[];
  sub_agent_count?: number;
  warnings?: string[];
  metadata?: Record<string, unknown>;
  request_id?: string;
  created_at?: string;
}

// ============================================================================
// Analysis API
// ============================================================================

/**
 * Request payload for the /analyze endpoint
 */
export interface AnalyzeRequest {
  documents?: string[];
  context?: string;
  provider?: string;
  include_chunk_suggestion?: boolean;
}

/**
 * Suggestion for context chunking
 */
export interface ChunkSuggestion {
  strategy: string;
  chunk_size: number;
  overlap: number;
  estimated_chunks: number;
  rationale: string;
}

/**
 * Response from the /analyze endpoint
 */
export interface AnalysisResponse {
  token_count: number;
  density: number;
  complexity: string;
  complexity_score: number;
  recommended_strategy: Strategy | string;
  estimated_costs: Record<string, number>;
  estimated_time: number;
  structure_type: string;
  chunk_suggestion?: ChunkSuggestion;
  warnings?: string[];
  metadata?: Record<string, unknown>;
}

// ============================================================================
// Batch Processing
// ============================================================================

/**
 * Request to process multiple tasks in batch
 */
export interface BatchProcessRequest {
  requests: ProcessRequest[];
  parallel?: boolean;
  max_concurrent?: number;
  fail_fast?: boolean;
}

/**
 * Response from batch processing
 */
export interface BatchProcessResponse {
  success: boolean;
  results: (ProcessResponse | ErrorResponse)[];
  total_requests: number;
  successful_count: number;
  failed_count: number;
  total_execution_time: number;
  total_token_usage: TokenUsage;
}

// ============================================================================
// Session Management
// ============================================================================

/**
 * Request to create a new session
 */
export interface CreateSessionRequest {
  documents?: string[];
  context?: string;
  metadata?: Record<string, unknown>;
  ttl_seconds?: number;
}

/**
 * Information about a session
 */
export interface SessionInfo {
  session_id: string;
  created_at: string;
  last_accessed: string;
  document_count: number;
  total_tokens: number;
  chunk_count: number;
  metadata?: Record<string, unknown>;
}

/**
 * Response from session creation
 */
export interface CreateSessionResponse {
  session_id: string;
  session_info: SessionInfo;
}

// ============================================================================
// Search API
// ============================================================================

/**
 * Request to search within session context
 */
export interface SearchRequest {
  query: string;
  session_id?: string;
  max_results?: number;
  include_scores?: boolean;
  threshold?: number;
}

/**
 * Single search result
 */
export interface SearchResult {
  content: string;
  score: number;
  chunk_id: string;
  metadata: Record<string, unknown>;
}

/**
 * Response from context search
 */
export interface SearchResponse {
  results: SearchResult[];
  total_results: number;
  query: string;
  search_time_ms: number;
}

// ============================================================================
// Provider Information
// ============================================================================

/**
 * Information about an LLM provider
 */
export interface ProviderInfo {
  name: string;
  available: boolean;
  models: string[];
  max_context: number;
  supports_streaming: boolean;
  supports_tools: boolean;
  rate_limit_rpm: number;
}

// ============================================================================
// Health Check
// ============================================================================

/**
 * Response from the /health endpoint
 */
export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  providers: ProviderInfo[];
  uptime_seconds: number;
  active_sessions: number;
  memory_usage_mb: number;
  timestamp: string;
}

// ============================================================================
// Error Handling
// ============================================================================

/**
 * Standard error response structure
 */
export interface ErrorResponse {
  success: false;
  error: string;
  error_type: string;
  error_code?: string;
  details?: Record<string, unknown>;
  request_id?: string;
  timestamp?: string;
}

// ============================================================================
// Recent Tasks
// ============================================================================

/**
 * A recent task entry for display in the UI
 */
export interface RecentTask {
  id: string;
  task: string;
  strategy: Strategy;
  tokens: number;
  cost: number;
  status: 'success' | 'warning' | 'error';
  time: string;
}

// ============================================================================
// WebSocket Communication
// ============================================================================

/**
 * Message sent to the WebSocket server
 */
export interface WebSocketMessage {
  type: 'process' | 'analyze' | 'cancel' | 'ping';
  task?: string;
  context?: string;
  documents?: string[];
  strategy?: Strategy;
  request_id?: string;
}

/**
 * Response received from the WebSocket server
 */
export interface WebSocketResponse {
  type: 'chunk' | 'metadata' | 'progress' | 'done' | 'error' | 'connected' | 'pong' | 'cancelled';
  content?: string;
  metadata?: Record<string, unknown>;
  request_id?: string;
  error?: string;
  timestamp?: string;
}
