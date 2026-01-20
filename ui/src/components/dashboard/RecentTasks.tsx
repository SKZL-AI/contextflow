import { Activity, ChevronRight, CheckCircle2, AlertCircle, XCircle } from 'lucide-react';
import { useAppStore } from '../../stores/appStore';

const strategyStyles: Record<string, string> = {
  gsd: 'bg-green-500/20 text-green-400',
  GSD: 'bg-green-500/20 text-green-400',
  ralph: 'bg-blue-500/20 text-blue-400',
  RALPH: 'bg-blue-500/20 text-blue-400',
  rlm: 'bg-orange-500/20 text-orange-400',
  RLM: 'bg-orange-500/20 text-orange-400',
  auto: 'bg-purple-500/20 text-purple-400',
};

const getRelativeTime = (isoTime: string): string => {
  const diff = Date.now() - new Date(isoTime).getTime();
  const minutes = Math.floor(diff / 60000);
  if (minutes < 1) return 'Just now';
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
};

export function RecentTasks() {
  const { recentTasks } = useAppStore();

  return (
    <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-green-500" />
          <h2 className="text-lg font-semibold">Letzte Verarbeitungen</h2>
        </div>
        <a
          href="#"
          className="flex items-center gap-1 text-sm text-slate-400 hover:text-slate-300 transition-colors"
        >
          Alle anzeigen
          <ChevronRight className="w-4 h-4" />
        </a>
      </div>

      <div className="rounded-xl border border-slate-800 overflow-hidden">
        {recentTasks.length === 0 ? (
          <div className="px-4 py-8 text-center text-slate-500">
            Keine Tasks vorhanden
          </div>
        ) : (
          <table className="w-full">
            <thead className="bg-slate-800/50">
              <tr>
                <th className="text-left px-4 py-3 text-sm font-medium text-slate-400">Task</th>
                <th className="text-left px-4 py-3 text-sm font-medium text-slate-400">Strategie</th>
                <th className="text-left px-4 py-3 text-sm font-medium text-slate-400">Tokens</th>
                <th className="text-left px-4 py-3 text-sm font-medium text-slate-400">Kosten</th>
                <th className="text-left px-4 py-3 text-sm font-medium text-slate-400">Status</th>
                <th className="text-left px-4 py-3 text-sm font-medium text-slate-400">Zeit</th>
              </tr>
            </thead>
            <tbody>
              {recentTasks.map((task) => (
                <tr key={task.id} className="hover:bg-slate-800/30 transition-colors">
                  <td className="px-4 py-3 font-medium">{task.task}</td>
                  <td className="px-4 py-3">
                    <span
                      className={`px-2 py-1 rounded-lg text-xs font-medium ${strategyStyles[task.strategy] || 'bg-slate-500/20 text-slate-400'}`}
                    >
                      {task.strategy.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-slate-400">{task.tokens.toLocaleString()}</td>
                  <td className="px-4 py-3 text-slate-400">${task.cost.toFixed(2)}</td>
                  <td className="px-4 py-3">
                    {task.status === 'success' ? (
                      <CheckCircle2 className="w-5 h-5 text-green-500" />
                    ) : task.status === 'error' ? (
                      <XCircle className="w-5 h-5 text-red-500" />
                    ) : (
                      <AlertCircle className="w-5 h-5 text-yellow-500" />
                    )}
                  </td>
                  <td className="px-4 py-3 text-sm text-slate-500">{getRelativeTime(task.time)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
