/**
 * ContextFlow API Hook
 *
 * Custom hook for making API calls to the ContextFlow backend.
 * Manages loading states, error handling, and integrates with the app store.
 */

import { useState, useCallback } from 'react';
import * as api from '../services/api';
import { useAppStore } from '../stores/appStore';
import type {
  Strategy,
  ProcessResponse,
  HealthResponse,
  ProviderInfo,
  AnalysisResponse,
  RecentTask,
} from '../types/api';

// ============================================================================
// Types
// ============================================================================

interface UseContextFlowAPIReturn {
  loading: boolean;
  error: string | null;
  fetchHealth: () => Promise<HealthResponse | null>;
  fetchProviders: () => Promise<ProviderInfo[] | null>;
  processTask: (
    task: string,
    strategy: Strategy,
    context?: string,
    documents?: string[]
  ) => Promise<ProcessResponse | null>;
  analyzeContext: (
    context: string,
    documents?: string[]
  ) => Promise<AnalysisResponse | null>;
  clearError: () => void;
}

// ============================================================================
// Hook Implementation
// ============================================================================

/**
 * Hook for interacting with the ContextFlow API
 *
 * Provides methods for health checks, provider listing, task processing,
 * and context analysis. Automatically manages loading states and errors.
 *
 * @example
 * ```tsx
 * const { loading, error, fetchHealth, processTask } = useContextFlowAPI();
 *
 * useEffect(() => {
 *   fetchHealth();
 * }, [fetchHealth]);
 *
 * const handleSubmit = async () => {
 *   const result = await processTask('Summarize this', 'auto', context);
 *   if (result) {
 *     console.log('Success:', result.answer);
 *   }
 * };
 * ```
 */
export function useContextFlowAPI(): UseContextFlowAPIReturn {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { setProviders, setHealth, addTask, setCurrentResponse } = useAppStore();

  /**
   * Clear the current error state
   */
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  /**
   * Fetch health status from the API
   */
  const fetchHealth = useCallback(async (): Promise<HealthResponse | null> => {
    setLoading(true);
    setError(null);

    try {
      const health = await api.getHealth();
      setHealth(health);
      return health;
    } catch (err) {
      const message =
        err instanceof api.ApiError
          ? err.message
          : 'Failed to fetch health status';
      setError(message);
      console.error('[useContextFlowAPI] Health check failed:', err);
      return null;
    } finally {
      setLoading(false);
    }
  }, [setHealth]);

  /**
   * Fetch available providers from the API
   */
  const fetchProviders = useCallback(async (): Promise<ProviderInfo[] | null> => {
    setLoading(true);
    setError(null);

    try {
      const providers = await api.getProviders();
      setProviders(providers);
      return providers;
    } catch (err) {
      const message =
        err instanceof api.ApiError
          ? err.message
          : 'Failed to fetch providers';
      setError(message);
      console.error('[useContextFlowAPI] Fetch providers failed:', err);
      return null;
    } finally {
      setLoading(false);
    }
  }, [setProviders]);

  /**
   * Process a task using the ContextFlow orchestrator
   */
  const processTask = useCallback(
    async (
      task: string,
      strategy: Strategy,
      context?: string,
      documents?: string[]
    ): Promise<ProcessResponse | null> => {
      setLoading(true);
      setError(null);

      try {
        const response = await api.process({
          task,
          strategy,
          context,
          documents,
        });

        setCurrentResponse(response);

        // Add to recent tasks
        const recentTask: RecentTask = {
          id: response.request_id || crypto.randomUUID(),
          task: task.slice(0, 100) + (task.length > 100 ? '...' : ''),
          strategy: response.strategy_used,
          tokens: response.token_usage.total_tokens,
          cost: response.token_usage.cost_usd,
          status: response.success
            ? response.verification_passed === false
              ? 'warning'
              : 'success'
            : 'error',
          time: response.created_at || new Date().toISOString(),
        };
        addTask(recentTask);

        return response;
      } catch (err) {
        const message =
          err instanceof api.ApiError
            ? err.message
            : 'Failed to process task';
        setError(message);
        console.error('[useContextFlowAPI] Process task failed:', err);

        // Add failed task to history
        const failedTask: RecentTask = {
          id: crypto.randomUUID(),
          task: task.slice(0, 100) + (task.length > 100 ? '...' : ''),
          strategy,
          tokens: 0,
          cost: 0,
          status: 'error',
          time: new Date().toISOString(),
        };
        addTask(failedTask);

        return null;
      } finally {
        setLoading(false);
      }
    },
    [setCurrentResponse, addTask]
  );

  /**
   * Analyze context/documents for strategy recommendation
   */
  const analyzeContext = useCallback(
    async (
      context: string,
      documents?: string[]
    ): Promise<AnalysisResponse | null> => {
      setLoading(true);
      setError(null);

      try {
        const analysis = await api.analyze({
          context,
          documents,
        });
        return analysis;
      } catch (err) {
        const message =
          err instanceof api.ApiError
            ? err.message
            : 'Failed to analyze context';
        setError(message);
        console.error('[useContextFlowAPI] Analyze context failed:', err);
        return null;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  return {
    loading,
    error,
    fetchHealth,
    fetchProviders,
    processTask,
    analyzeContext,
    clearError,
  };
}

export default useContextFlowAPI;
