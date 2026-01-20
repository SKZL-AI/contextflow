/**
 * ContextFlow API Service
 *
 * Axios-based API client for the ContextFlow backend
 */

import axios, { AxiosInstance, AxiosError, AxiosResponse, InternalAxiosRequestConfig } from 'axios';
import type {
  HealthResponse,
  ProviderInfo,
  ProcessRequest,
  ProcessResponse,
  AnalyzeRequest,
  AnalysisResponse,
  ErrorResponse,
  BatchProcessRequest,
  BatchProcessResponse,
  CreateSessionRequest,
  CreateSessionResponse,
  SessionInfo,
  SearchRequest,
  SearchResponse,
} from '../types/api';

// Re-export types for convenience
export type { SessionInfo, SearchResponse } from '../types/api';

// ============================================================================
// API Error Class
// ============================================================================

export class ApiError extends Error {
  public readonly status: number;
  public readonly errorType: string;
  public readonly errorCode?: string;
  public readonly details?: Record<string, unknown>;
  public readonly requestId?: string;

  constructor(
    message: string,
    status: number,
    errorType: string = 'unknown',
    errorCode?: string,
    details?: Record<string, unknown>,
    requestId?: string
  ) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.errorType = errorType;
    this.errorCode = errorCode;
    this.details = details;
    this.requestId = requestId;
  }

  static fromErrorResponse(response: ErrorResponse, status: number): ApiError {
    return new ApiError(
      response.error,
      status,
      response.error_type,
      response.error_code,
      response.details,
      response.request_id
    );
  }
}

// ============================================================================
// Axios Instance Setup
// ============================================================================

const apiClient: AxiosInstance = axios.create({
  baseURL: '/api/v1',
  timeout: 60000, // 60 seconds for long-running requests
  headers: {
    'Content-Type': 'application/json',
  },
});

// ============================================================================
// Request Interceptor
// ============================================================================

apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig): InternalAxiosRequestConfig => {
    // Add timestamp for request tracking
    config.headers.set('X-Request-Time', new Date().toISOString());

    // Log request in development
    if (import.meta.env.DEV) {
      console.debug(`[API] ${config.method?.toUpperCase()} ${config.url}`);
    }

    return config;
  },
  (error: AxiosError): Promise<never> => {
    console.error('[API] Request setup error:', error.message);
    return Promise.reject(new ApiError(
      `Request setup failed: ${error.message}`,
      0,
      'request_setup_error'
    ));
  }
);

// ============================================================================
// Response Interceptor
// ============================================================================

apiClient.interceptors.response.use(
  (response: AxiosResponse): AxiosResponse => {
    // Log response in development
    if (import.meta.env.DEV) {
      console.debug(`[API] Response ${response.status} from ${response.config.url}`);
    }
    return response;
  },
  (error: AxiosError<ErrorResponse>): Promise<never> => {
    // Handle network errors
    if (!error.response) {
      console.error('[API] Network error:', error.message);
      return Promise.reject(new ApiError(
        'Network error: Unable to connect to server',
        0,
        'network_error'
      ));
    }

    const { status, data } = error.response;

    // Handle structured error responses
    if (data && typeof data === 'object' && 'error' in data) {
      console.error(`[API] Error ${status}:`, data.error);
      return Promise.reject(ApiError.fromErrorResponse(data, status));
    }

    // Handle generic HTTP errors
    const errorMessages: Record<number, string> = {
      400: 'Bad request: Invalid parameters',
      401: 'Unauthorized: Authentication required',
      403: 'Forbidden: Access denied',
      404: 'Not found: Resource does not exist',
      422: 'Validation error: Invalid input data',
      429: 'Rate limit exceeded: Too many requests',
      500: 'Server error: Internal server error',
      502: 'Bad gateway: Server unavailable',
      503: 'Service unavailable: Server is overloaded',
      504: 'Gateway timeout: Server took too long to respond',
    };

    const message = errorMessages[status] || `HTTP error ${status}`;
    console.error(`[API] ${message}`);

    return Promise.reject(new ApiError(
      message,
      status,
      'http_error'
    ));
  }
);

// ============================================================================
// API Functions
// ============================================================================

/**
 * Check API health status
 */
export async function getHealth(): Promise<HealthResponse> {
  try {
    const response = await apiClient.get<HealthResponse>('/health');
    return response.data;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError(
      'Failed to check health status',
      0,
      'unknown_error'
    );
  }
}

/**
 * Get list of available LLM providers
 */
export async function getProviders(): Promise<ProviderInfo[]> {
  try {
    const response = await apiClient.get<ProviderInfo[]>('/providers');
    return response.data;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError(
      'Failed to fetch providers',
      0,
      'unknown_error'
    );
  }
}

/**
 * Process a task with the ContextFlow orchestrator
 */
export async function process(request: ProcessRequest): Promise<ProcessResponse> {
  try {
    const response = await apiClient.post<ProcessResponse>('/process', request);
    return response.data;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError(
      'Failed to process task',
      0,
      'unknown_error'
    );
  }
}

/**
 * Process a task with Server-Sent Events streaming
 * Returns an EventSource for real-time updates
 */
export function processStream(request: ProcessRequest): EventSource {
  // Build query string from request parameters
  const params = new URLSearchParams();
  params.set('task', request.task);

  if (request.documents) {
    params.set('documents', JSON.stringify(request.documents));
  }
  if (request.context) {
    params.set('context', request.context);
  }
  if (request.strategy) {
    params.set('strategy', request.strategy);
  }
  if (request.provider) {
    params.set('provider', request.provider);
  }
  if (request.max_tokens !== undefined) {
    params.set('max_tokens', request.max_tokens.toString());
  }
  if (request.temperature !== undefined) {
    params.set('temperature', request.temperature.toString());
  }
  if (request.verify !== undefined) {
    params.set('verify', request.verify.toString());
  }
  if (request.constraints) {
    params.set('constraints', JSON.stringify(request.constraints));
  }

  const url = `/api/v1/process/stream?${params.toString()}`;
  return new EventSource(url);
}

/**
 * Analyze documents/context for strategy recommendation
 */
export async function analyze(request: AnalyzeRequest): Promise<AnalysisResponse> {
  try {
    const response = await apiClient.post<AnalysisResponse>('/analyze', request);
    return response.data;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError(
      'Failed to analyze content',
      0,
      'unknown_error'
    );
  }
}

/**
 * Batch process multiple tasks with parallel execution
 */
export async function batchProcess(request: BatchProcessRequest): Promise<BatchProcessResponse> {
  try {
    const response = await apiClient.post<BatchProcessResponse>('/batch', request);
    return response.data;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError(
      'Failed to batch process tasks',
      0,
      'unknown_error'
    );
  }
}

/**
 * Create a new session for stateful processing
 */
export async function createSession(
  request: CreateSessionRequest
): Promise<CreateSessionResponse> {
  try {
    const response = await apiClient.post<CreateSessionResponse>('/sessions', request);
    return response.data;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError(
      'Failed to create session',
      0,
      'unknown_error'
    );
  }
}

/**
 * Get session information by ID
 */
export async function getSession(sessionId: string): Promise<SessionInfo> {
  try {
    const response = await apiClient.get<SessionInfo>(`/sessions/${sessionId}`);
    return response.data;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError(
      `Failed to get session: ${sessionId}`,
      0,
      'unknown_error'
    );
  }
}

/**
 * Delete a session by ID
 */
export async function deleteSession(sessionId: string): Promise<void> {
  try {
    await apiClient.delete(`/sessions/${sessionId}`);
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError(
      `Failed to delete session: ${sessionId}`,
      0,
      'unknown_error'
    );
  }
}

/**
 * Search within a session's documents using RAG
 */
export async function search(request: SearchRequest): Promise<SearchResponse> {
  try {
    const response = await apiClient.post<SearchResponse>('/search', request);
    return response.data;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError(
      'Failed to perform search',
      0,
      'unknown_error'
    );
  }
}

/**
 * Get API root information
 */
export async function getApiInfo(): Promise<Record<string, unknown>> {
  try {
    const response = await apiClient.get<Record<string, unknown>>('/');
    return response.data;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError(
      'Failed to get API info',
      0,
      'unknown_error'
    );
  }
}

// ============================================================================
// WebSocket Helper
// ============================================================================

/**
 * Create a WebSocket connection to the specified endpoint
 */
export function createWebSocket(endpoint: '/ws/process' | '/ws/stream'): WebSocket {
  // Determine WebSocket protocol based on current page protocol
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;
  const url = `${protocol}//${host}/api/v1${endpoint}`;

  const ws = new WebSocket(url);

  // Add error logging
  ws.onerror = (event) => {
    console.error('[WebSocket] Connection error:', event);
  };

  ws.onclose = (event) => {
    if (event.wasClean) {
      console.debug(`[WebSocket] Connection closed cleanly, code=${event.code}`);
    } else {
      console.warn(`[WebSocket] Connection died, code=${event.code}`);
    }
  };

  return ws;
}

// ============================================================================
// Export API Client for Advanced Usage
// ============================================================================

export { apiClient };
