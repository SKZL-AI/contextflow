import { useState, useEffect } from 'react';
import {
  Settings as SettingsIcon,
  Save,
  RotateCcw,
  Server,
  Brain,
  Palette,
  Globe,
  CheckCircle,
  AlertCircle,
  Loader2,
  Zap,
} from 'lucide-react';
import { useAppStore } from '../stores/appStore';
import { useContextFlowAPI } from '../hooks/useContextFlowAPI';
import type { Strategy } from '../types/api';

// ============================================================================
// Types
// ============================================================================

interface SettingsState {
  defaultProvider: string;
  defaultStrategy: Strategy;
  language: 'de' | 'en';
  darkMode: boolean;
}

interface StrategyInfo {
  value: Strategy;
  label: string;
  description: string;
}

// ============================================================================
// Constants
// ============================================================================

const STORAGE_KEY = 'contextflow-settings';

const DEFAULT_SETTINGS: SettingsState = {
  defaultProvider: 'claude',
  defaultStrategy: 'auto',
  language: 'de',
  darkMode: true,
};

const STRATEGIES: StrategyInfo[] = [
  {
    value: 'auto',
    label: 'Auto',
    description: 'Automatically selects the best strategy based on context size and complexity',
  },
  {
    value: 'gsd',
    label: 'GSD',
    description: 'Get Stuff Done - Direct approach for small contexts (< 10K tokens)',
  },
  {
    value: 'ralph',
    label: 'RALPH',
    description: 'Recursive Abstraction Layer for Processing Hierarchies (10K - 100K tokens)',
  },
  {
    value: 'rlm',
    label: 'RLM',
    description: 'Recursive Language Model - Multi-agent approach for large contexts (> 100K tokens)',
  },
];

const TOKEN_THRESHOLDS = [
  { strategy: 'GSD', range: '< 10,000 tokens', description: 'Direct single-pass processing' },
  { strategy: 'RALPH', range: '10,000 - 100,000 tokens', description: 'Iterative structured processing' },
  { strategy: 'RLM', range: '> 100,000 tokens', description: 'Recursive multi-agent processing' },
];

// ============================================================================
// Settings Component
// ============================================================================

export const Settings: React.FC = () => {
  const { providers, health } = useAppStore();
  const { fetchHealth, loading } = useContextFlowAPI();

  const [settings, setSettings] = useState<SettingsState>(DEFAULT_SETTINGS);
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null);
  const [healthLoading, setHealthLoading] = useState(false);

  // Load settings from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        setSettings({ ...DEFAULT_SETTINGS, ...parsed });
      } catch (e) {
        console.error('Failed to parse stored settings:', e);
      }
    }
  }, []);

  // Auto-hide toast after 3 seconds
  useEffect(() => {
    if (toast) {
      const timer = setTimeout(() => setToast(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [toast]);

  const saveSettings = () => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
      setToast({ message: 'Settings saved successfully', type: 'success' });
    } catch (e) {
      setToast({ message: 'Failed to save settings', type: 'error' });
    }
  };

  const resetSettings = () => {
    setSettings(DEFAULT_SETTINGS);
    localStorage.removeItem(STORAGE_KEY);
    setToast({ message: 'Settings reset to defaults', type: 'success' });
  };

  const checkHealth = async () => {
    setHealthLoading(true);
    const result = await fetchHealth();
    setHealthLoading(false);
    if (result) {
      setToast({ message: 'Backend is healthy', type: 'success' });
    } else {
      setToast({ message: 'Failed to reach backend', type: 'error' });
    }
  };

  const updateSetting = <K extends keyof SettingsState>(key: K, value: SettingsState[K]) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
  };

  // Get available providers for dropdown
  const availableProviders = providers.length > 0
    ? providers.filter(p => p.available)
    : [{ name: 'claude', available: true }]; // Fallback

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 bg-violet-500/20 rounded-lg">
            <SettingsIcon className="w-5 h-5 text-violet-400" />
          </div>
          <h1 className="text-xl font-semibold">Settings</h1>
        </div>
        <p className="text-slate-400 text-sm">
          Configure ContextFlow defaults and preferences
        </p>
      </div>

      {/* Toast Notification */}
      {toast && (
        <div
          className={`fixed top-4 right-4 z-50 flex items-center gap-2 px-4 py-3 rounded-lg shadow-lg transition-all ${
            toast.type === 'success'
              ? 'bg-emerald-500/20 border border-emerald-500/30 text-emerald-400'
              : 'bg-red-500/20 border border-red-500/30 text-red-400'
          }`}
        >
          {toast.type === 'success' ? (
            <CheckCircle className="w-4 h-4" />
          ) : (
            <AlertCircle className="w-4 h-4" />
          )}
          <span className="text-sm font-medium">{toast.message}</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Provider Settings */}
        <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-blue-500/20 rounded-lg">
              <Server className="w-4 h-4 text-blue-400" />
            </div>
            <h2 className="text-lg font-semibold">Provider Settings</h2>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Default Provider
              </label>
              <select
                value={settings.defaultProvider}
                onChange={(e) => updateSetting('defaultProvider', e.target.value)}
                className="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent transition-all"
              >
                {availableProviders.map((provider) => (
                  <option key={provider.name} value={provider.name}>
                    {provider.name.charAt(0).toUpperCase() + provider.name.slice(1)}
                  </option>
                ))}
              </select>
              <p className="text-xs text-slate-500 mt-1">
                Currently selected: {settings.defaultProvider}
              </p>
            </div>
          </div>
        </div>

        {/* Strategy Settings */}
        <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-purple-500/20 rounded-lg">
              <Brain className="w-4 h-4 text-purple-400" />
            </div>
            <h2 className="text-lg font-semibold">Strategy Settings</h2>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Default Strategy
              </label>
              <select
                value={settings.defaultStrategy}
                onChange={(e) => updateSetting('defaultStrategy', e.target.value as Strategy)}
                className="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent transition-all"
              >
                {STRATEGIES.map((strategy) => (
                  <option key={strategy.value} value={strategy.value}>
                    {strategy.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Strategy Description */}
            <div className="p-3 bg-slate-800/50 rounded-lg border border-slate-700/50">
              <p className="text-sm text-slate-400">
                {STRATEGIES.find((s) => s.value === settings.defaultStrategy)?.description}
              </p>
            </div>
          </div>
        </div>

        {/* Display Settings */}
        <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-amber-500/20 rounded-lg">
              <Palette className="w-4 h-4 text-amber-400" />
            </div>
            <h2 className="text-lg font-semibold">Display Settings</h2>
          </div>

          <div className="space-y-4">
            {/* Language Toggle */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Globe className="w-4 h-4 text-slate-400" />
                <div>
                  <span className="text-sm font-medium text-slate-300">Language</span>
                  <p className="text-xs text-slate-500">Interface language (coming soon)</p>
                </div>
              </div>
              <div className="flex items-center gap-2 bg-slate-800 rounded-lg p-1">
                <button
                  onClick={() => updateSetting('language', 'de')}
                  className={`px-3 py-1.5 text-sm rounded-md transition-all ${
                    settings.language === 'de'
                      ? 'bg-violet-500 text-white'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  DE
                </button>
                <button
                  onClick={() => updateSetting('language', 'en')}
                  className={`px-3 py-1.5 text-sm rounded-md transition-all ${
                    settings.language === 'en'
                      ? 'bg-violet-500 text-white'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  EN
                </button>
              </div>
            </div>

            {/* Theme Toggle */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Palette className="w-4 h-4 text-slate-400" />
                <div>
                  <span className="text-sm font-medium text-slate-300">Dark Mode</span>
                  <p className="text-xs text-slate-500">Theme preference</p>
                </div>
              </div>
              <button
                onClick={() => updateSetting('darkMode', !settings.darkMode)}
                className={`relative w-12 h-6 rounded-full transition-colors ${
                  settings.darkMode ? 'bg-violet-500' : 'bg-slate-700'
                }`}
              >
                <span
                  className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                    settings.darkMode ? 'left-7' : 'left-1'
                  }`}
                />
              </button>
            </div>
          </div>
        </div>

        {/* API Configuration */}
        <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-emerald-500/20 rounded-lg">
              <Server className="w-4 h-4 text-emerald-400" />
            </div>
            <h2 className="text-lg font-semibold">API Configuration</h2>
          </div>

          <div className="space-y-4">
            {/* API Endpoint */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                API Endpoint
              </label>
              <input
                type="text"
                value="/api/v1"
                readOnly
                className="w-full px-4 py-2.5 bg-slate-800/50 border border-slate-700 rounded-lg text-slate-400 cursor-not-allowed"
              />
            </div>

            {/* Health Status */}
            <div className="flex items-center justify-between">
              <div>
                <span className="text-sm font-medium text-slate-300">Backend Status</span>
                <div className="flex items-center gap-2 mt-1">
                  {health ? (
                    <>
                      <span
                        className={`w-2 h-2 rounded-full ${
                          health.status === 'healthy'
                            ? 'bg-emerald-500'
                            : health.status === 'degraded'
                            ? 'bg-amber-500'
                            : 'bg-red-500'
                        }`}
                      />
                      <span className="text-xs text-slate-400 capitalize">{health.status}</span>
                    </>
                  ) : (
                    <>
                      <span className="w-2 h-2 rounded-full bg-slate-500" />
                      <span className="text-xs text-slate-500">Unknown</span>
                    </>
                  )}
                </div>
              </div>
              <button
                onClick={checkHealth}
                disabled={healthLoading || loading}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {healthLoading || loading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Zap className="w-4 h-4" />
                )}
                Check Health
              </button>
            </div>

            {/* Backend Version */}
            {health && (
              <div className="p-3 bg-slate-800/50 rounded-lg border border-slate-700/50">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-400">Backend Version</span>
                  <span className="text-white font-mono">{health.version}</span>
                </div>
                <div className="flex items-center justify-between text-sm mt-2">
                  <span className="text-slate-400">Active Sessions</span>
                  <span className="text-white">{health.active_sessions}</span>
                </div>
                <div className="flex items-center justify-between text-sm mt-2">
                  <span className="text-slate-400">Memory Usage</span>
                  <span className="text-white">{health.memory_usage_mb.toFixed(1)} MB</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Token Limits Info */}
      <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 bg-cyan-500/20 rounded-lg">
            <Zap className="w-4 h-4 text-cyan-400" />
          </div>
          <h2 className="text-lg font-semibold">Strategy Token Thresholds</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {TOKEN_THRESHOLDS.map((threshold, index) => (
            <div
              key={threshold.strategy}
              className="p-4 bg-slate-800/50 rounded-xl border border-slate-700/50"
            >
              <div className="flex items-center gap-2 mb-2">
                <span
                  className={`px-2 py-0.5 text-xs font-semibold rounded ${
                    index === 0
                      ? 'bg-emerald-500/20 text-emerald-400'
                      : index === 1
                      ? 'bg-amber-500/20 text-amber-400'
                      : 'bg-red-500/20 text-red-400'
                  }`}
                >
                  {threshold.strategy}
                </span>
              </div>
              <p className="text-sm font-medium text-white mb-1">{threshold.range}</p>
              <p className="text-xs text-slate-400">{threshold.description}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Save/Reset Buttons */}
      <div className="flex items-center justify-end gap-3">
        <button
          onClick={resetSettings}
          className="px-6 py-2.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
        >
          <RotateCcw className="w-4 h-4" />
          Reset to Defaults
        </button>
        <button
          onClick={saveSettings}
          className="px-6 py-2.5 bg-violet-600 hover:bg-violet-500 rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
        >
          <Save className="w-4 h-4" />
          Save Settings
        </button>
      </div>
    </div>
  );
};

export default Settings;
