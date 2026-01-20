/**
 * Streaming Response Hook
 *
 * Custom hook for handling streaming responses from the ContextFlow backend.
 * Manages chunks, accumulates full response, and provides stream control.
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { useWebSocket } from './useWebSocket';
import type { Strategy, WebSocketResponse } from '../types/api';

// ============================================================================
// Types
// ============================================================================

interface StreamMetadata {
  requestId: string;
  strategy?: Strategy;
  startTime: number;
  tokenCount?: number;
  [key: string]: unknown;
}

interface UseStreamingResponseOptions {
  /** Callback when streaming starts */
  onStreamStart?: (requestId: string) => void;
  /** Callback when chunk received */
  onChunk?: (chunk: string, index: number) => void;
  /** Callback when streaming completes */
  onStreamComplete?: (fullResponse: string, metadata?: StreamMetadata) => void;
  /** Callback when error occurs */
  onError?: (error: string) => void;
  /** Callback when stream is cancelled */
  onCancel?: (requestId: string) => void;
}

interface UseStreamingResponseReturn {
  /** Array of received chunks */
  chunks: string[];
  /** Full accumulated response */
  fullResponse: string;
  /** Whether currently streaming */
  isStreaming: boolean;
  /** Whether WebSocket is connected */
  isConnected: boolean;
  /** Current stream request ID */
  requestId: string | null;
  /** Stream metadata */
  metadata: StreamMetadata | null;
  /** Error message if any */
  error: string | null;
  /** Progress percentage (if available) */
  progress: number | null;
  /** Start a new streaming request */
  startStream: (
    task: string,
    strategy: Strategy,
    context?: string,
    documents?: string[]
  ) => string | null;
  /** Cancel the current stream */
  cancelStream: () => boolean;
  /** Reset stream state */
  resetStream: () => void;
}

// ============================================================================
// Hook Implementation
// ============================================================================

/**
 * Hook for handling streaming responses from ContextFlow
 *
 * Manages WebSocket connection for streaming, accumulates response chunks,
 * and provides control methods for stream management.
 *
 * @param options - Configuration options
 *
 * @example
 * ```tsx
 * const {
 *   chunks,
 *   fullResponse,
 *   isStreaming,
 *   startStream,
 *   cancelStream,
 * } = useStreamingResponse({
 *   onChunk: (chunk) => console.log('Chunk:', chunk),
 *   onStreamComplete: (response) => console.log('Done:', response),
 * });
 *
 * const handleStart = () => {
 *   startStream('Summarize this document', 'auto', documentContent);
 * };
 * ```
 */
export function useStreamingResponse(
  options: UseStreamingResponseOptions = {}
): UseStreamingResponseReturn {
  const {
    onStreamStart,
    onChunk,
    onStreamComplete,
    onError,
    onCancel,
  } = options;

  const [chunks, setChunks] = useState<string[]>([]);
  const [fullResponse, setFullResponse] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [requestId, setRequestId] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<StreamMetadata | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<number | null>(null);

  const chunkIndexRef = useRef(0);

  /**
   * Handle incoming WebSocket messages
   */
  const handleMessage = useCallback(
    (message: WebSocketResponse) => {
      // Ignore messages for different requests
      if (requestId && message.request_id && message.request_id !== requestId) {
        return;
      }

      switch (message.type) {
        case 'chunk':
          if (message.content) {
            const chunkContent = message.content;
            const currentIndex = chunkIndexRef.current++;

            setChunks((prev) => [...prev, chunkContent]);
            setFullResponse((prev) => prev + chunkContent);
            onChunk?.(chunkContent, currentIndex);
          }
          break;

        case 'metadata':
          if (message.metadata) {
            setMetadata((prev) => ({
              ...(prev || { requestId: requestId || '', startTime: Date.now() }),
              ...message.metadata,
            }));
          }
          break;

        case 'progress':
          if (message.metadata?.progress !== undefined) {
            setProgress(message.metadata.progress as number);
          }
          break;

        case 'done':
          setIsStreaming(false);
          setProgress(100);

          const finalMetadata: StreamMetadata = {
            requestId: requestId || '',
            startTime: metadata?.startTime || Date.now(),
            ...message.metadata,
          };
          setMetadata(finalMetadata);
          onStreamComplete?.(fullResponse, finalMetadata);
          break;

        case 'error':
          setIsStreaming(false);
          const errorMessage = message.error || 'Unknown streaming error';
          setError(errorMessage);
          onError?.(errorMessage);
          break;

        case 'cancelled':
          setIsStreaming(false);
          if (requestId) {
            onCancel?.(requestId);
          }
          break;

        case 'connected':
          console.debug('[useStreamingResponse] WebSocket connected');
          break;

        case 'pong':
          // Heartbeat response, no action needed
          break;
      }
    },
    [requestId, metadata, fullResponse, onChunk, onStreamComplete, onError, onCancel]
  );

  const {
    isConnected,
    sendMessage,
    connect,
  } = useWebSocket('/ws/stream', {
    autoConnect: true,
    autoReconnect: true,
    onMessage: handleMessage,
    onError: () => {
      if (isStreaming) {
        setError('WebSocket connection error');
        setIsStreaming(false);
        onError?.('WebSocket connection error');
      }
    },
    onDisconnect: (event) => {
      if (isStreaming && event.code !== 1000) {
        setError('Connection lost during streaming');
        setIsStreaming(false);
        onError?.('Connection lost during streaming');
      }
    },
  });

  /**
   * Reset stream state
   */
  const resetStream = useCallback(() => {
    setChunks([]);
    setFullResponse('');
    setIsStreaming(false);
    setRequestId(null);
    setMetadata(null);
    setError(null);
    setProgress(null);
    chunkIndexRef.current = 0;
  }, []);

  /**
   * Start a new streaming request
   */
  const startStream = useCallback(
    (
      task: string,
      strategy: Strategy,
      context?: string,
      documents?: string[]
    ): string | null => {
      // Ensure connection
      if (!isConnected) {
        connect();
        // Wait a bit for connection
        setTimeout(() => {
          if (!isConnected) {
            setError('Failed to connect to WebSocket');
            onError?.('Failed to connect to WebSocket');
          }
        }, 3000);
      }

      // Reset state for new stream
      resetStream();

      // Generate new request ID
      const newRequestId = crypto.randomUUID();
      setRequestId(newRequestId);

      // Initialize metadata
      const newMetadata: StreamMetadata = {
        requestId: newRequestId,
        strategy,
        startTime: Date.now(),
      };
      setMetadata(newMetadata);

      // Start streaming
      setIsStreaming(true);
      setProgress(0);

      // Send process message
      const success = sendMessage({
        type: 'process',
        task,
        strategy,
        context,
        documents,
        request_id: newRequestId,
      });

      if (success) {
        onStreamStart?.(newRequestId);
        return newRequestId;
      } else {
        setIsStreaming(false);
        setError('Failed to send stream request');
        onError?.('Failed to send stream request');
        return null;
      }
    },
    [isConnected, connect, resetStream, sendMessage, onStreamStart, onError]
  );

  /**
   * Cancel the current stream
   */
  const cancelStream = useCallback((): boolean => {
    if (!isStreaming || !requestId) {
      return false;
    }

    const success = sendMessage({
      type: 'cancel',
      request_id: requestId,
    });

    if (success) {
      setIsStreaming(false);
      onCancel?.(requestId);
    }

    return success;
  }, [isStreaming, requestId, sendMessage, onCancel]);

  // Update fullResponse when chunks change (for cases where handleMessage
  // might miss updates due to closure issues)
  useEffect(() => {
    if (chunks.length > 0) {
      setFullResponse(chunks.join(''));
    }
  }, [chunks]);

  return {
    chunks,
    fullResponse,
    isStreaming,
    isConnected,
    requestId,
    metadata,
    error,
    progress,
    startStream,
    cancelStream,
    resetStream,
  };
}

export default useStreamingResponse;
