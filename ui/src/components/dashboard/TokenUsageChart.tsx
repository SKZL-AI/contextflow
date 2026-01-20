import { useMemo } from 'react';
import { TrendingUp } from 'lucide-react';
import { useAppStore } from '../../stores/appStore';

export function TokenUsageChart() {
  const { recentTasks } = useAppStore();

  const tokenData = useMemo(() => {
    // Get day names in German (Mo, Di, Mi, Do, Fr, Sa, So)
    const dayNames = ['So', 'Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa'];

    // Create array for last 7 days
    const last7Days: { day: string; tokens: number; date: Date }[] = [];
    for (let i = 6; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      last7Days.push({
        day: dayNames[date.getDay()],
        tokens: 0,
        date: date
      });
    }

    // Aggregate tokens per day
    recentTasks.forEach(task => {
      const taskDate = new Date(task.time);
      const dayIndex = last7Days.findIndex(d =>
        d.date.toDateString() === taskDate.toDateString()
      );
      if (dayIndex !== -1) {
        last7Days[dayIndex].tokens += task.tokens;
      }
    });

    // Calculate heights relative to max
    const maxTokens = Math.max(...last7Days.map(d => d.tokens), 1);
    return last7Days.map(d => ({
      day: d.day,
      height: Math.max((d.tokens / maxTokens) * 100, d.tokens > 0 ? 5 : 0) // At least 5% height if has tokens
    }));
  }, [recentTasks]);

  const hasData = tokenData.some(d => d.height > 0);

  return (
    <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800">
      <div className="flex items-center gap-2 mb-6">
        <TrendingUp className="w-5 h-5 text-cyan-500" />
        <h3 className="text-lg font-semibold">Token-Verbrauch (7 Tage)</h3>
      </div>

      <div className="h-48 flex items-end justify-between gap-2 relative">
        {!hasData && (
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-slate-500 text-sm">Keine Daten</span>
          </div>
        )}
        {tokenData.map(({ day, height }) => (
          <div key={day} className="flex-1 flex flex-col items-center gap-2">
            <div
              className={`w-full rounded-t-lg transition-all hover:opacity-80 ${
                height > 0
                  ? 'bg-gradient-to-t from-violet-500 to-purple-400'
                  : 'bg-slate-800'
              }`}
              style={{ height: height > 0 ? `${height}%` : '2px' }}
            />
            <span className="text-xs text-slate-500">{day}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
