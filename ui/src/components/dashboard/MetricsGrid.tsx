import { FileText, Activity, DollarSign, Clock, LucideIcon } from 'lucide-react';

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

const defaultMetrics: MetricCardData[] = [
  {
    label: 'Verarbeitete Tasks',
    value: '1,284',
    change: '+12%',
    icon: FileText,
    iconColor: 'text-blue-500',
  },
  {
    label: 'Tokens Heute',
    value: '2.4M',
    change: '+8%',
    icon: Activity,
    iconColor: 'text-green-500',
  },
  {
    label: 'Kosten Heute',
    value: '$23.45',
    change: '-5%',
    icon: DollarSign,
    iconColor: 'text-purple-500',
  },
  {
    label: 'Avg. Latenz',
    value: '1.1s',
    change: '-15%',
    icon: Clock,
    iconColor: 'text-orange-500',
  },
];

export function MetricsGrid({ metrics = defaultMetrics }: MetricsGridProps) {
  return (
    <div className="grid grid-cols-4 gap-6">
      {metrics.map((metric, index) => {
        const Icon = metric.icon;
        const isPositive = metric.change.startsWith('+');

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
                  isPositive ? 'text-green-500' : 'text-red-500'
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
