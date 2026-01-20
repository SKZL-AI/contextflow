import { useState } from 'react';
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

export function Dashboard() {
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy>('auto');
  const [isProcessing, setIsProcessing] = useState(false);
  const [ragEnabled, setRagEnabled] = useState(false);

  const handleStrategySelect = (strategy: Strategy) => {
    setSelectedStrategy(strategy);
  };

  const handleTaskSubmit = async (task: string) => {
    setIsProcessing(true);
    try {
      // TODO: Implement task submission logic via API
      console.log('Submitting task:', task, 'with strategy:', selectedStrategy);
      await new Promise((resolve) => setTimeout(resolve, 2000));
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
