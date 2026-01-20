import React, { useState, useEffect } from 'react';
import {
  Server,
  Check,
  X,
  Zap,
  Wrench,
  Clock,
  ChevronDown,
  ChevronUp,
  Loader2,
  RefreshCw,
} from 'lucide-react';
import { useAppStore } from '../stores/appStore';
import { useContextFlowAPI } from '../hooks/useContextFlowAPI';
import type { ProviderInfo } from '../types/api';

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Format context window size for display
 */
function formatContextSize(tokens: number): string {
  if (tokens >= 1000000) {
    return `${(tokens / 1000000).toFixed(1)}M`;
  }
  if (tokens >= 1000) {
    return `${(tokens / 1000).toFixed(0)}K`;
  }
  return tokens.toString();
}

/**
 * Format rate limit for display
 */
function formatRateLimit(rpm: number): string {
  if (rpm === 0) return 'Unlimited';
  if (rpm >= 1000) return `${(rpm / 1000).toFixed(1)}K/min`;
  return `${rpm}/min`;
}

// ============================================================================
// Status Indicator Component
// ============================================================================

interface StatusIndicatorProps {
  available: boolean;
  loading?: boolean;
}

const StatusIndicator: React.FC<StatusIndicatorProps> = ({ available, loading }) => {
  if (loading) {
    return (
      <div className="flex items-center gap-2">
        <Loader2 className="w-3 h-3 text-slate-400 animate-spin" />
        <span className="text-xs text-slate-400">Checking...</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <div
        className={`w-2 h-2 rounded-full ${
          available ? 'bg-green-500' : 'bg-slate-500'
        }`}
      />
      <span className={`text-xs ${available ? 'text-green-400' : 'text-slate-500'}`}>
        {available ? 'Available' : 'Unavailable'}
      </span>
    </div>
  );
};

// ============================================================================
// Feature Badge Component
// ============================================================================

interface FeatureBadgeProps {
  icon: React.ReactNode;
  label: string;
  supported: boolean;
}

const FeatureBadge: React.FC<FeatureBadgeProps> = ({ icon, label, supported }) => {
  return (
    <div
      className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${
        supported
          ? 'bg-green-500/10 text-green-400 border border-green-500/20'
          : 'bg-slate-800 text-slate-500 border border-slate-700'
      }`}
    >
      {icon}
      <span className="text-xs font-medium">{label}</span>
      {supported ? (
        <Check className="w-3 h-3" />
      ) : (
        <X className="w-3 h-3" />
      )}
    </div>
  );
};

// ============================================================================
// Model List Component
// ============================================================================

interface ModelListProps {
  models: string[];
}

const ModelList: React.FC<ModelListProps> = ({ models }) => {
  if (models.length === 0) {
    return <p className="text-slate-500 text-sm">No models available</p>;
  }

  return (
    <div className="flex flex-wrap gap-2">
      {models.map((model) => (
        <span
          key={model}
          className="px-2 py-1 text-xs bg-slate-800 text-slate-300 rounded border border-slate-700"
        >
          {model}
        </span>
      ))}
    </div>
  );
};

// ============================================================================
// Provider Card Component
// ============================================================================

interface ProviderCardProps {
  provider: ProviderInfo;
  isSelected: boolean;
  isExpanded: boolean;
  loading: boolean;
  onSelect: () => void;
  onToggleExpand: () => void;
}

const ProviderCard: React.FC<ProviderCardProps> = ({
  provider,
  isSelected,
  isExpanded,
  loading,
  onSelect,
  onToggleExpand,
}) => {
  return (
    <div
      className={`rounded-xl border transition-all duration-200 ${
        isSelected
          ? 'border-violet-500 bg-violet-500/5'
          : 'border-slate-800 bg-slate-900 hover:border-slate-700'
      }`}
    >
      {/* Card Header */}
      <div className="p-4">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            {/* Selection Radio */}
            <button
              onClick={onSelect}
              disabled={!provider.available}
              className={`w-5 h-5 rounded-full border-2 flex items-center justify-center transition-colors ${
                isSelected
                  ? 'border-violet-500 bg-violet-500'
                  : provider.available
                  ? 'border-slate-600 hover:border-slate-500'
                  : 'border-slate-700 cursor-not-allowed'
              }`}
              title={provider.available ? 'Select as default provider' : 'Provider unavailable'}
            >
              {isSelected && <Check className="w-3 h-3 text-white" />}
            </button>

            {/* Provider Icon & Name */}
            <div className="flex items-center gap-2">
              <Server className="w-5 h-5 text-slate-400" />
              <h3 className="font-semibold text-white capitalize">{provider.name}</h3>
            </div>
          </div>

          {/* Status & Expand Button */}
          <div className="flex items-center gap-3">
            <StatusIndicator available={provider.available} loading={loading} />
            <button
              onClick={onToggleExpand}
              className="p-1 rounded hover:bg-slate-800 transition-colors"
              title={isExpanded ? 'Collapse details' : 'Expand details'}
            >
              {isExpanded ? (
                <ChevronUp className="w-4 h-4 text-slate-400" />
              ) : (
                <ChevronDown className="w-4 h-4 text-slate-400" />
              )}
            </button>
          </div>
        </div>

        {/* Quick Info */}
        <div className="mt-3 flex items-center gap-4 text-sm text-slate-400">
          <span>{provider.models.length} models</span>
          <span className="text-slate-600">|</span>
          <span>{formatContextSize(provider.max_context)} context</span>
        </div>
      </div>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="px-4 pb-4 border-t border-slate-800 pt-4 space-y-4">
          {/* Models Section */}
          <div>
            <h4 className="text-xs uppercase tracking-wider text-slate-500 mb-2">
              Available Models
            </h4>
            <ModelList models={provider.models} />
          </div>

          {/* Specs Section */}
          <div>
            <h4 className="text-xs uppercase tracking-wider text-slate-500 mb-2">
              Specifications
            </h4>
            <div className="grid grid-cols-2 gap-3">
              <div className="flex items-center gap-2 text-sm">
                <Clock className="w-4 h-4 text-slate-500" />
                <span className="text-slate-400">Max Context:</span>
                <span className="text-white font-medium">
                  {formatContextSize(provider.max_context)} tokens
                </span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <Clock className="w-4 h-4 text-slate-500" />
                <span className="text-slate-400">Rate Limit:</span>
                <span className="text-white font-medium">
                  {formatRateLimit(provider.rate_limit_rpm)}
                </span>
              </div>
            </div>
          </div>

          {/* Features Section */}
          <div>
            <h4 className="text-xs uppercase tracking-wider text-slate-500 mb-2">
              Capabilities
            </h4>
            <div className="flex flex-wrap gap-2">
              <FeatureBadge
                icon={<Zap className="w-3 h-3" />}
                label="Streaming"
                supported={provider.supports_streaming}
              />
              <FeatureBadge
                icon={<Wrench className="w-3 h-3" />}
                label="Tools/Functions"
                supported={provider.supports_tools}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Empty State Component
// ============================================================================

interface EmptyStateProps {
  onRefresh: () => void;
  loading: boolean;
}

const EmptyState: React.FC<EmptyStateProps> = ({ onRefresh, loading }) => {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <Server className="w-12 h-12 text-slate-600 mb-4" />
      <h3 className="text-lg font-medium text-slate-300 mb-2">No Providers Found</h3>
      <p className="text-slate-500 max-w-sm mb-4">
        Could not retrieve provider information. Make sure the backend is running and try again.
      </p>
      <button
        onClick={onRefresh}
        disabled={loading}
        className="flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-500 disabled:bg-slate-700 text-white rounded-lg transition-colors"
      >
        {loading ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : (
          <RefreshCw className="w-4 h-4" />
        )}
        Refresh
      </button>
    </div>
  );
};

// ============================================================================
// Loading State Component
// ============================================================================

const LoadingState: React.FC = () => {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {[1, 2, 3].map((i) => (
        <div
          key={i}
          className="rounded-xl border border-slate-800 bg-slate-900 p-4 animate-pulse"
        >
          <div className="flex items-center gap-3 mb-3">
            <div className="w-5 h-5 rounded-full bg-slate-800" />
            <div className="h-5 w-24 bg-slate-800 rounded" />
          </div>
          <div className="h-4 w-32 bg-slate-800 rounded mb-2" />
          <div className="h-4 w-40 bg-slate-800 rounded" />
        </div>
      ))}
    </div>
  );
};

// ============================================================================
// Main Providers Component
// ============================================================================

export const Providers: React.FC = () => {
  const { providers } = useAppStore();
  const { fetchProviders, loading, error } = useContextFlowAPI();

  // Local state for UI
  const [expandedProvider, setExpandedProvider] = useState<string | null>(null);
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [initialLoading, setInitialLoading] = useState(true);

  // Fetch providers on mount
  useEffect(() => {
    const loadProviders = async () => {
      await fetchProviders();
      setInitialLoading(false);
    };
    loadProviders();
  }, [fetchProviders]);

  // Set default selected provider (first available)
  useEffect(() => {
    if (providers.length > 0 && selectedProvider === null) {
      const firstAvailable = providers.find((p) => p.available);
      if (firstAvailable) {
        setSelectedProvider(firstAvailable.name);
      }
    }
  }, [providers, selectedProvider]);

  const handleSelectProvider = (providerName: string) => {
    const provider = providers.find((p) => p.name === providerName);
    if (provider?.available) {
      setSelectedProvider(providerName);
    }
  };

  const handleToggleExpand = (providerName: string) => {
    setExpandedProvider((prev) => (prev === providerName ? null : providerName));
  };

  const handleRefresh = () => {
    setInitialLoading(true);
    fetchProviders().then(() => setInitialLoading(false));
  };

  // Stats
  const availableCount = providers.filter((p) => p.available).length;
  const totalModels = providers.reduce((sum, p) => sum + p.models.length, 0);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-lg font-semibold text-white">Provider Management</h1>
            <p className="text-sm text-slate-400 mt-1">
              Configure and monitor LLM providers
            </p>
          </div>
          <button
            onClick={handleRefresh}
            disabled={loading}
            className="flex items-center gap-2 px-3 py-1.5 text-sm bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-300 rounded-lg transition-colors"
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <RefreshCw className="w-4 h-4" />
            )}
            Refresh
          </button>
        </div>

        {/* Stats Row */}
        {providers.length > 0 && (
          <div className="flex items-center gap-6 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-500" />
              <span className="text-slate-400">
                {availableCount} of {providers.length} available
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Server className="w-4 h-4 text-slate-500" />
              <span className="text-slate-400">{totalModels} total models</span>
            </div>
            {selectedProvider && (
              <div className="flex items-center gap-2">
                <Check className="w-4 h-4 text-violet-500" />
                <span className="text-slate-400">
                  Default: <span className="text-violet-400 capitalize">{selectedProvider}</span>
                </span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 text-red-400">
          <p className="font-medium">Error loading providers</p>
          <p className="text-sm mt-1 text-red-400/80">{error}</p>
        </div>
      )}

      {/* Provider Grid */}
      <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800">
        {initialLoading ? (
          <LoadingState />
        ) : providers.length === 0 ? (
          <EmptyState onRefresh={handleRefresh} loading={loading} />
        ) : (
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {providers.map((provider) => (
              <ProviderCard
                key={provider.name}
                provider={provider}
                isSelected={selectedProvider === provider.name}
                isExpanded={expandedProvider === provider.name}
                loading={loading}
                onSelect={() => handleSelectProvider(provider.name)}
                onToggleExpand={() => handleToggleExpand(provider.name)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Providers;
