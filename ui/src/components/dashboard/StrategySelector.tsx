import { Brain, Zap, Layers, Cpu, LucideIcon } from 'lucide-react';

export type Strategy = 'auto' | 'gsd' | 'ralph' | 'rlm';

interface StrategyOption {
  id: Strategy;
  name: string;
  description: string;
  icon: LucideIcon;
  gradient: string;
}

interface StrategySelectorProps {
  selectedStrategy: Strategy;
  onSelect: (strategy: Strategy) => void;
}

const strategies: StrategyOption[] = [
  {
    id: 'auto',
    name: 'Auto',
    description: 'Automatische Auswahl',
    icon: Brain,
    gradient: 'from-violet-500 to-purple-600',
  },
  {
    id: 'gsd',
    name: 'GSD',
    description: '<10K Tokens',
    icon: Zap,
    gradient: 'from-green-500 to-emerald-600',
  },
  {
    id: 'ralph',
    name: 'RALPH',
    description: '10K-100K Tokens',
    icon: Layers,
    gradient: 'from-blue-500 to-cyan-600',
  },
  {
    id: 'rlm',
    name: 'RLM',
    description: '>100K Tokens',
    icon: Cpu,
    gradient: 'from-orange-500 to-red-600',
  },
];

export function StrategySelector({ selectedStrategy, onSelect }: StrategySelectorProps) {
  return (
    <div className="grid grid-cols-4 gap-4">
      {strategies.map((strategy) => {
        const Icon = strategy.icon;
        const isSelected = selectedStrategy === strategy.id;

        return (
          <button
            key={strategy.id}
            onClick={() => onSelect(strategy.id)}
            className={`relative p-4 rounded-xl border transition-all text-left ${
              isSelected
                ? 'border-violet-500 bg-violet-500/10'
                : 'border-slate-700 hover:border-slate-600 bg-slate-800/50'
            }`}
          >
            {isSelected && (
              <div className="absolute top-2 right-2 w-2 h-2 rounded-full bg-violet-500" />
            )}
            <div
              className={`w-10 h-10 rounded-lg bg-gradient-to-br ${strategy.gradient} flex items-center justify-center mb-3`}
            >
              <Icon className="w-5 h-5 text-white" />
            </div>
            <div className="font-semibold text-white">{strategy.name}</div>
            <div className="text-xs text-slate-500">{strategy.description}</div>
          </button>
        );
      })}
    </div>
  );
}
