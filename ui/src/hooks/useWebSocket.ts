/**
 * WebSocket Hook
 *
 * Custom hook for managing WebSocket connections to the ContextFlow backend.
 * Handles connection lifecycle, reconnection, and message handling.
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import { createWebSocket } from '../services/api';
import type { WebSocketResponse, WebSocketMessage } from '../types/api';

// ============================================================================
// Types
// ============================================================================

type WebSocketEndpoint = '/ws/process' | '/ws/stream';

interface UseWebSocketOptions {
  /** Auto-connect on mount (default: true) */
  autoConnect?: boolean;
  /** Auto-reconnect on disconnect (default: true) */
  autoReconnect?: boolean;
  /** Reconnection delay in ms (default: 3000) */
  reconnectDelay?: number;
  /** Maximum reconnection attempts (default: 5) */
  maxReconnectAttempts?: number;
  /** Callback when connected */
  onConnect?: () => void;
  /** Callback when disconnected */
  onDisconnect?: (event: CloseEvent) => void;
  /** Callback when error occurs */
  onError?: (error: Event) => void;
  /** Callback when message received */
  onMessage?: (message: WebSocketResponse) => void;
}

interface UseWebSocketReturn {
  isConnected: boolean;
  isConnecting: boolean;
  lastMessage: WebSocketResponse | null;
  connectionError: string | null;
  sendMessage: (message: WebSocketMessage) => boolean;
  connect: () => void;
  disconnect: () => void;
  reconnect: () => void;
}

// ============================================================================
// Hook Implementation
// ============================================================================

/**
 * Hook for managing WebSocket connections
 *
 * Provides connection management, automatic reconnection, and message handling
 * for real-time communication with the ContextFlow backend.
 *
 * @param endpoint - WebSocket endpoint to connect to
 * @param options - Configuration options
 *
 * @example
 * ```tsx
 * const { isConnected, lastMessage, sendMessage } = useWebSocket('/ws/process');
 *
 * useEffect(() => {
 *   if (lastMessage?.type === 'chunk') {
 *     console.log('Received chunk:', lastMessage.content);
 *   }
 * }, [lastMessage]);
 *
 * const handleSend = () => {
 *   sendMessage({ type: 'process', task: 'Hello' });
 * };
 * ```
 */
export function useWebSocket(
  endpoint: WebSocketEndpoint = '/ws/process',
  options: UseWebSocketOptions = {}
): UseWebSocketReturn {
  const {
    autoConnect = true,
    autoReconnect = true,
    reconnectDelay = 3000,
    maxReconnectAttempts = 5,
    onConnect,
    onDisconnect,
    onError,
    onMessage,
  } = options;

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketResponse | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  /**
   * Clear any pending reconnection timeout
   */
  const clearReconnectTimeout = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  }, []);

  /**
   * Disconnect from WebSocket
   */
  const disconnect = useCallback(() => {
    clearReconnectTimeout();

    if (wsRef.current) {
      // Remove event listeners before closing
      wsRef.current.onopen = null;
      wsRef.current.onclose = null;
      wsRef.current.onerror = null;
      wsRef.current.onmessage = null;

      if (
        wsRef.current.readyState === WebSocket.OPEN ||
        wsRef.current.readyState === WebSocket.CONNECTING
      ) {
        wsRef.current.close(1000, 'Client disconnect');
      }

      wsRef.current = null;
    }

    setIsConnected(false);
    setIsConnecting(false);
  }, [clearReconnectTimeout]);

  /**
   * Connect to WebSocket
   */
  const connect = useCallback(() => {
    // Don't connect if already connected or connecting
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      return;
    }

    // Clean up existing connection
    disconnect();

    if (!mountedRef.current) {
      return;
    }

    setIsConnecting(true);
    setConnectionError(null);

    try {
      const ws = createWebSocket(endpoint);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;

        setIsConnected(true);
        setIsConnecting(false);
        setConnectionError(null);
        reconnectAttemptsRef.current = 0;

        console.debug(`[useWebSocket] Connected to ${endpoint}`);
        onConnect?.();
      };

      ws.onclose = (event: CloseEvent) => {
        if (!mountedRef.current) return;

        setIsConnected(false);
        setIsConnecting(false);

        console.debug(
          `[useWebSocket] Disconnected from ${endpoint}:`,
          event.code,
          event.reason
        );
        onDisconnect?.(event);

        // Auto-reconnect if enabled and not a clean close
        if (
          autoReconnect &&
          event.code !== 1000 &&
          reconnectAttemptsRef.current < maxReconnectAttempts
        ) {
          const delay =
            reconnectDelay * Math.pow(2, reconnectAttemptsRef.current);
          reconnectAttemptsRef.current++;

          console.debug(
            `[useWebSocket] Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts})`
          );

          reconnectTimeoutRef.current = setTimeout(() => {
            if (mountedRef.current) {
              connect();
            }
          }, delay);
        } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
          setConnectionError('Maximum reconnection attempts reached');
        }
      };

      ws.onerror = (event: Event) => {
        if (!mountedRef.current) return;

        console.error(`[useWebSocket] Error on ${endpoint}:`, event);
        setConnectionError('WebSocket connection error');
        onError?.(event);
      };

      ws.onmessage = (event: MessageEvent) => {
        if (!mountedRef.current) return;

        try {
          const message: WebSocketResponse = JSON.parse(event.data);
          setLastMessage(message);
          onMessage?.(message);

          // Handle error messages from server
          if (message.type === 'error' && message.error) {
            console.warn('[useWebSocket] Server error:', message.error);
          }
        } catch (parseError) {
          console.error(
            '[useWebSocket] Failed to parse message:',
            event.data,
            parseError
          );
        }
      };
    } catch (err) {
      setIsConnecting(false);
      setConnectionError(
        err instanceof Error ? err.message : 'Failed to create WebSocket'
      );
      console.error('[useWebSocket] Failed to connect:', err);
    }
  }, [
    endpoint,
    disconnect,
    autoReconnect,
    reconnectDelay,
    maxReconnectAttempts,
    onConnect,
    onDisconnect,
    onError,
    onMessage,
  ]);

  /**
   * Force reconnection
   */
  const reconnect = useCallback(() => {
    reconnectAttemptsRef.current = 0;
    disconnect();
    connect();
  }, [disconnect, connect]);

  /**
   * Send a message through the WebSocket
   */
  const sendMessage = useCallback((message: WebSocketMessage): boolean => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.warn('[useWebSocket] Cannot send message: not connected');
      return false;
    }

    try {
      wsRef.current.send(JSON.stringify(message));
      return true;
    } catch (err) {
      console.error('[useWebSocket] Failed to send message:', err);
      return false;
    }
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    mountedRef.current = true;

    if (autoConnect) {
      connect();
    }

    return () => {
      mountedRef.current = false;
      clearReconnectTimeout();
      disconnect();
    };
  }, [autoConnect, connect, disconnect, clearReconnectTimeout]);

  return {
    isConnected,
    isConnecting,
    lastMessage,
    connectionError,
    sendMessage,
    connect,
    disconnect,
    reconnect,
  };
}

export default useWebSocket;
