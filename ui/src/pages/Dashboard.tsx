import { useState, useEffect } from 'react';
import { Brain } from 'lucide-react';
import {
  MetricsGrid,
  StrategySelector,
  TaskInput,
  ProviderStatus,
  RecentTasks,
  TokenUsageChart,
  StrategyDistribution,
} from '../components/dashboard';
import type { Strategy } from '../components/dashboard';
import { useContextFlowAPI } from '../hooks/useContextFlowAPI';

export function Dashboard() {
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy>('auto');
  const [isProcessing, setIsProcessing] = useState(false);
  const [ragEnabled, setRagEnabled] = useState(false);

  const { processTask, fetchProviders, fetchHealth } = useContextFlowAPI();

  // Initialize data on mount
  useEffect(() => {
    fetchHealth();
    fetchProviders();
  }, [fetchHealth, fetchProviders]);

  const handleStrategySelect = (strategy: Strategy) => {
    setSelectedStrategy(strategy);
  };

  const handleTaskSubmit = async (task: string) => {
    setIsProcessing(true);
    try {
      await processTask(task, selectedStrategy);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleToggleRag = () => {
    setRagEnabled(!ragEnabled);
  };

  return (
    <div className="space-y-6">
      {/* Metrics Grid at top */}
      <MetricsGrid />

      {/* Strategy Selection and Provider Status */}
      <div className="grid grid-cols-3 gap-6">
        {/* Strategy Selection Panel */}
        <div className="col-span-2 bg-slate-900 rounded-2xl p-6 border border-slate-800">
          <div className="flex items-center gap-2 mb-6">
            <Brain className="w-5 h-5 text-violet-500" />
            <h2 className="text-lg font-semibold">Strategie-Auswahl</h2>
          </div>

          <StrategySelector
            selectedStrategy={selectedStrategy}
            onSelect={handleStrategySelect}
          />

          <div className="mt-6">
            <TaskInput
              onSubmit={handleTaskSubmit}
              isProcessing={isProcessing}
              ragEnabled={ragEnabled}
              onToggleRag={handleToggleRag}
            />
          </div>
        </div>

        {/* Provider Status */}
        <div className="col-span-1">
          <ProviderStatus />
        </div>
      </div>

      {/* Recent Tasks - Full Width */}
      <RecentTasks />

      {/* Charts Row */}
      <div className="grid grid-cols-2 gap-6">
        <TokenUsageChart />
        <StrategyDistribution />
      </div>
    </div>
  );
}

export default Dashboard;
