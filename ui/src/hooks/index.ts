/**
 * ContextFlow UI Hooks
 *
 * Custom React hooks for the ContextFlow UI application.
 * These hooks provide reusable state management and API integration logic.
 */

// API Hook - For making REST API calls to the ContextFlow backend
export { useContextFlowAPI, default as useContextFlowAPIDefault } from './useContextFlowAPI';

// WebSocket Hook - For managing WebSocket connections
export { useWebSocket, default as useWebSocketDefault } from './useWebSocket';

// Streaming Response Hook - For handling streaming responses
export { useStreamingResponse, default as useStreamingResponseDefault } from './useStreamingResponse';
