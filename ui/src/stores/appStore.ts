/**
 * ContextFlow UI Application State Store
 *
 * Zustand store for managing global application state including
 * UI state, data state, and actions.
 */

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import type {
  Strategy,
  ProcessResponse,
  ProviderInfo,
  RecentTask,
  HealthResponse,
} from '../types/api';

// ============================================================================
// State Interface
// ============================================================================

interface UIState {
  selectedStrategy: Strategy;
  isProcessing: boolean;
  ragEnabled: boolean;
  activeTab: string;
}

interface DataState {
  providers: ProviderInfo[];
  recentTasks: RecentTask[];
  health: HealthResponse | null;
  currentResponse: ProcessResponse | null;
}

interface Actions {
  setStrategy: (strategy: Strategy) => void;
  setProcessing: (processing: boolean) => void;
  toggleRag: () => void;
  setActiveTab: (tab: string) => void;
  setProviders: (providers: ProviderInfo[]) => void;
  addTask: (task: RecentTask) => void;
  setHealth: (health: HealthResponse) => void;
  setCurrentResponse: (response: ProcessResponse | null) => void;
  reset: () => void;
}

export interface AppState extends UIState, DataState, Actions {}

// ============================================================================
// Initial State
// ============================================================================

const initialUIState: UIState = {
  selectedStrategy: 'auto',
  isProcessing: false,
  ragEnabled: false,
  activeTab: 'process',
};

const initialDataState: DataState = {
  providers: [],
  recentTasks: [],
  health: null,
  currentResponse: null,
};

// ============================================================================
// Store Creation
// ============================================================================

export const useAppStore = create<AppState>()(
  immer((set) => ({
    // UI State
    ...initialUIState,

    // Data State
    ...initialDataState,

    // Actions
    setStrategy: (strategy: Strategy) =>
      set((state) => {
        state.selectedStrategy = strategy;
      }),

    setProcessing: (processing: boolean) =>
      set((state) => {
        state.isProcessing = processing;
      }),

    toggleRag: () =>
      set((state) => {
        state.ragEnabled = !state.ragEnabled;
      }),

    setActiveTab: (tab: string) =>
      set((state) => {
        state.activeTab = tab;
      }),

    setProviders: (providers: ProviderInfo[]) =>
      set((state) => {
        state.providers = providers;
      }),

    addTask: (task: RecentTask) =>
      set((state) => {
        state.recentTasks.unshift(task);
        // Keep only the last 50 tasks
        if (state.recentTasks.length > 50) {
          state.recentTasks = state.recentTasks.slice(0, 50);
        }
      }),

    setHealth: (health: HealthResponse) =>
      set((state) => {
        state.health = health;
      }),

    setCurrentResponse: (response: ProcessResponse | null) =>
      set((state) => {
        state.currentResponse = response;
      }),

    reset: () =>
      set((state) => {
        // Reset UI state
        state.selectedStrategy = initialUIState.selectedStrategy;
        state.isProcessing = initialUIState.isProcessing;
        state.ragEnabled = initialUIState.ragEnabled;
        state.activeTab = initialUIState.activeTab;

        // Reset data state
        state.providers = initialDataState.providers;
        state.recentTasks = initialDataState.recentTasks;
        state.health = initialDataState.health;
        state.currentResponse = initialDataState.currentResponse;
      }),
  }))
);

// ============================================================================
// Selector Helpers
// ============================================================================

/**
 * Select only UI-related state (useful for components that only need UI state)
 */
export const selectUIState = (state: AppState): UIState => ({
  selectedStrategy: state.selectedStrategy,
  isProcessing: state.isProcessing,
  ragEnabled: state.ragEnabled,
  activeTab: state.activeTab,
});

/**
 * Select only data-related state (useful for components that only need data)
 */
export const selectDataState = (state: AppState): DataState => ({
  providers: state.providers,
  recentTasks: state.recentTasks,
  health: state.health,
  currentResponse: state.currentResponse,
});
