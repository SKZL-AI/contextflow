import { useMemo } from 'react';
import { Layers } from 'lucide-react';
import { useAppStore } from '../../stores/appStore';

export function StrategyDistribution() {
  const { recentTasks } = useAppStore();

  const { strategies, total, circumference } = useMemo(() => {
    const circumference = 2 * Math.PI * 40; // 251.2 for r=40

    // Count strategies (normalize to lowercase for comparison)
    const gsdCount = recentTasks.filter(t =>
      t.strategy.toLowerCase() === 'gsd'
    ).length;
    const ralphCount = recentTasks.filter(t =>
      t.strategy.toLowerCase() === 'ralph'
    ).length;
    const rlmCount = recentTasks.filter(t =>
      t.strategy.toLowerCase() === 'rlm'
    ).length;
    const autoCount = recentTasks.filter(t =>
      t.strategy.toLowerCase() === 'auto'
    ).length;

    const total = gsdCount + ralphCount + rlmCount + autoCount || 1; // Avoid division by 0

    const strategies = [
      {
        name: 'GSD',
        count: gsdCount,
        percentage: Math.round((gsdCount / total) * 100),
        color: 'bg-green-500',
        strokeColor: '#22c55e'
      },
      {
        name: 'RALPH',
        count: ralphCount,
        percentage: Math.round((ralphCount / total) * 100),
        color: 'bg-blue-500',
        strokeColor: '#3b82f6'
      },
      {
        name: 'RLM',
        count: rlmCount,
        percentage: Math.round((rlmCount / total) * 100),
        color: 'bg-orange-500',
        strokeColor: '#f97316'
      },
    ];

    // Add AUTO if there are any
    if (autoCount > 0) {
      strategies.push({
        name: 'AUTO',
        count: autoCount,
        percentage: Math.round((autoCount / total) * 100),
        color: 'bg-purple-500',
        strokeColor: '#a855f7'
      });
    }

    return { strategies, total, circumference };
  }, [recentTasks]);

  // Calculate SVG segments
  let offset = 0;
  const segments = strategies.map(s => {
    const dashLength = (s.percentage / 100) * circumference;
    const segment = {
      ...s,
      dashArray: `${dashLength} ${circumference}`,
      dashOffset: -offset
    };
    offset += dashLength;
    return segment;
  });

  // Check if there are no tasks
  const hasData = recentTasks.length > 0;

  return (
    <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800">
      <div className="flex items-center gap-2 mb-6">
        <Layers className="w-5 h-5 text-amber-500" />
        <h3 className="text-lg font-semibold">Strategie-Verteilung</h3>
      </div>

      <div className="flex items-center justify-center h-48">
        {/* Donut Chart */}
        <div className="relative w-40 h-40">
          <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
            {/* Background circle */}
            <circle
              cx="50"
              cy="50"
              r="40"
              fill="none"
              stroke="#1e293b"
              strokeWidth="12"
            />
            {/* Dynamic segments */}
            {hasData && segments.map((segment) => (
              <circle
                key={segment.name}
                cx="50"
                cy="50"
                r="40"
                fill="none"
                stroke={segment.strokeColor}
                strokeWidth="12"
                strokeDasharray={segment.dashArray}
                strokeDashoffset={segment.dashOffset}
                strokeLinecap="round"
              />
            ))}
          </svg>
          {/* Center text */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            {hasData ? (
              <>
                <span className="text-2xl font-bold">{total.toLocaleString()}</span>
                <span className="text-xs text-slate-500">Tasks</span>
              </>
            ) : (
              <span className="text-sm text-slate-500">Keine Daten</span>
            )}
          </div>
        </div>

        {/* Legend */}
        <div className="ml-8 space-y-3">
          {hasData ? (
            strategies.map(({ name, percentage, color }) => (
              <div key={name} className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${color}`} />
                <span className="text-sm">{name} ({percentage}%)</span>
              </div>
            ))
          ) : (
            <span className="text-sm text-slate-500">Noch keine Tasks vorhanden</span>
          )}
        </div>
      </div>
    </div>
  );
}
