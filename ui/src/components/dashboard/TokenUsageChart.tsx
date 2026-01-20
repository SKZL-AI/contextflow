import { TrendingUp } from 'lucide-react';

const tokenData = [
  { day: 'Mo', height: 40 },
  { day: 'Di', height: 65 },
  { day: 'Mi', height: 45 },
  { day: 'Do', height: 80 },
  { day: 'Fr', height: 55 },
  { day: 'Sa', height: 90 },
  { day: 'So', height: 70 },
];

export function TokenUsageChart() {
  return (
    <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800">
      <div className="flex items-center gap-2 mb-6">
        <TrendingUp className="w-5 h-5 text-cyan-500" />
        <h3 className="text-lg font-semibold">Token-Verbrauch (7 Tage)</h3>
      </div>

      <div className="h-48 flex items-end justify-between gap-2">
        {tokenData.map(({ day, height }) => (
          <div key={day} className="flex-1 flex flex-col items-center gap-2">
            <div
              className="w-full bg-gradient-to-t from-violet-500 to-purple-400 rounded-t-lg transition-all hover:opacity-80"
              style={{ height: `${height}%` }}
            />
            <span className="text-xs text-slate-500">{day}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
