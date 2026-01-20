import { useMemo } from 'react';
import { FileText, Activity, DollarSign, Clock, LucideIcon } from 'lucide-react';
import { useAppStore } from '../../stores/appStore';

interface MetricCardData {
  label: string;
  value: string;
  change: string;
  icon: LucideIcon;
  iconColor: string;
}

interface MetricsGridProps {
  metrics?: MetricCardData[];
}

/**
 * Helper function to check if a date string represents today
 */
const isToday = (dateStr: string): boolean => {
  const date = new Date(dateStr);
  const today = new Date();
  return date.toDateString() === today.toDateString();
};

/**
 * Format large numbers with K/M suffixes
 */
const formatNumber = (num: number): string => {
  if (num >= 1000000) {
    return `${(num / 1000000).toFixed(1)}M`;
  }
  if (num >= 1000) {
    return `${(num / 1000).toFixed(1)}K`;
  }
  return num.toLocaleString();
};

/**
 * Format currency values
 */
const formatCurrency = (amount: number): string => {
  return `$${amount.toFixed(2)}`;
};

/**
 * Format time in seconds
 */
const formatLatency = (seconds: number): string => {
  if (seconds < 1) {
    return `${Math.round(seconds * 1000)}ms`;
  }
  return `${seconds.toFixed(1)}s`;
};

export function MetricsGrid({ metrics: propMetrics }: MetricsGridProps) {
  const { recentTasks, health } = useAppStore();

  const computedMetrics = useMemo((): MetricCardData[] => {
    // Filter tasks created today
    const todaysTasks = recentTasks.filter((task) => isToday(task.time));

    // Calculate metrics from real data
    const totalTasks = recentTasks.length;
    const tokensToday = todaysTasks.reduce((sum, task) => sum + task.tokens, 0);
    const costToday = todaysTasks.reduce((sum, task) => sum + task.cost, 0);

    // Calculate average latency from health data or estimate from task count
    const avgLatency = health?.uptime_seconds
      ? Math.min(health.uptime_seconds / Math.max(totalTasks, 1), 5)
      : totalTasks > 0
        ? 1.2
        : 0;

    return [
      {
        label: 'Verarbeitete Tasks',
        value: formatNumber(totalTasks),
        change: 'N/A',
        icon: FileText,
        iconColor: 'text-blue-500',
      },
      {
        label: 'Tokens Heute',
        value: formatNumber(tokensToday),
        change: 'N/A',
        icon: Activity,
        iconColor: 'text-green-500',
      },
      {
        label: 'Kosten Heute',
        value: formatCurrency(costToday),
        change: 'N/A',
        icon: DollarSign,
        iconColor: 'text-purple-500',
      },
      {
        label: 'Avg. Latenz',
        value: avgLatency > 0 ? formatLatency(avgLatency) : '0s',
        change: 'N/A',
        icon: Clock,
        iconColor: 'text-orange-500',
      },
    ];
  }, [recentTasks, health]);

  // Use provided metrics if passed, otherwise use computed metrics from store
  const metrics = propMetrics ?? computedMetrics;
  return (
    <div className="grid grid-cols-4 gap-6">
      {metrics.map((metric, index) => {
        const Icon = metric.icon;
        const isPositive = metric.change.startsWith('+');
        const isNeutral = metric.change === 'N/A' || metric.change === '0%';

        return (
          <div
            key={index}
            className="bg-slate-900 rounded-2xl p-6 border border-slate-800 hover:border-slate-700 transition-all"
          >
            <div className="flex items-center justify-between mb-4">
              <div className={`bg-slate-800 p-3 rounded-xl ${metric.iconColor}`}>
                <Icon className="w-5 h-5" />
              </div>
              <span
                className={`text-sm font-medium ${
                  isNeutral ? 'text-slate-500' : isPositive ? 'text-green-500' : 'text-red-500'
                }`}
              >
                {metric.change}
              </span>
            </div>
            <div className="text-2xl font-bold text-white mb-1">
              {metric.value}
            </div>
            <div className="text-sm text-slate-500">{metric.label}</div>
          </div>
        );
      })}
    </div>
  );
}
