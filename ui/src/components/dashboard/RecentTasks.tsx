import { Activity, ChevronRight, CheckCircle2, AlertCircle } from 'lucide-react';

type Strategy = 'GSD' | 'RALPH' | 'RLM';
type Status = 'success' | 'warning';

interface Task {
  name: string;
  strategy: Strategy;
  tokens: number;
  cost: number;
  status: Status;
  time: string;
}

const tasks: Task[] = [
  { name: 'Dokumenten-Analyse', strategy: 'RALPH', tokens: 45200, cost: 0.42, status: 'success', time: '2m ago' },
  { name: 'Code Review', strategy: 'GSD', tokens: 8500, cost: 0.08, status: 'success', time: '5m ago' },
  { name: 'Research Summary', strategy: 'RLM', tokens: 156000, cost: 1.85, status: 'success', time: '12m ago' },
  { name: 'API Docs Generation', strategy: 'RALPH', tokens: 32000, cost: 0.31, status: 'warning', time: '18m ago' },
];

const strategyStyles: Record<Strategy, string> = {
  GSD: 'bg-green-500/20 text-green-400',
  RALPH: 'bg-blue-500/20 text-blue-400',
  RLM: 'bg-orange-500/20 text-orange-400',
};

export function RecentTasks() {
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
            {tasks.map((task, index) => (
              <tr key={index} className="hover:bg-slate-800/30 transition-colors">
                <td className="px-4 py-3 font-medium">{task.name}</td>
                <td className="px-4 py-3">
                  <span
                    className={`px-2 py-1 rounded-lg text-xs font-medium ${strategyStyles[task.strategy]}`}
                  >
                    {task.strategy}
                  </span>
                </td>
                <td className="px-4 py-3 text-slate-400">{task.tokens.toLocaleString()}</td>
                <td className="px-4 py-3 text-slate-400">${task.cost.toFixed(2)}</td>
                <td className="px-4 py-3">
                  {task.status === 'success' ? (
                    <CheckCircle2 className="w-5 h-5 text-green-500" />
                  ) : (
                    <AlertCircle className="w-5 h-5 text-yellow-500" />
                  )}
                </td>
                <td className="px-4 py-3 text-sm text-slate-500">{task.time}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
