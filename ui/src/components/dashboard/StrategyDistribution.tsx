import { Layers } from 'lucide-react';

const strategies = [
  { name: 'GSD', percentage: 30, color: 'bg-green-500' },
  { name: 'RALPH', percentage: 40, color: 'bg-blue-500' },
  { name: 'RLM', percentage: 30, color: 'bg-orange-500' },
];

export function StrategyDistribution() {
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
            {/* GSD segment (30%) */}
            <circle
              cx="50"
              cy="50"
              r="40"
              fill="none"
              stroke="#22c55e"
              strokeWidth="12"
              strokeDasharray="75.4 251.2"
              strokeDashoffset="0"
              strokeLinecap="round"
            />
            {/* RALPH segment (40%) */}
            <circle
              cx="50"
              cy="50"
              r="40"
              fill="none"
              stroke="#3b82f6"
              strokeWidth="12"
              strokeDasharray="100.5 251.2"
              strokeDashoffset="-75.4"
              strokeLinecap="round"
            />
            {/* RLM segment (30%) */}
            <circle
              cx="50"
              cy="50"
              r="40"
              fill="none"
              stroke="#f97316"
              strokeWidth="12"
              strokeDasharray="75.4 251.2"
              strokeDashoffset="-175.9"
              strokeLinecap="round"
            />
          </svg>
          {/* Center text */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-2xl font-bold">1.2K</span>
            <span className="text-xs text-slate-500">Tasks</span>
          </div>
        </div>

        {/* Legend */}
        <div className="ml-8 space-y-3">
          {strategies.map(({ name, percentage, color }) => (
            <div key={name} className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${color}`} />
              <span className="text-sm">{name} ({percentage}%)</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
