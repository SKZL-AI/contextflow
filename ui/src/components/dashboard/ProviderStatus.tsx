import { Server } from 'lucide-react';
import { useAppStore } from '../../stores/appStore';

export function ProviderStatus() {
  const { providers } = useAppStore();

  const displayProviders = providers.map(p => ({
    name: p.name,
    model: p.models[0] || 'N/A',
    status: p.available ? 'active' as const : 'inactive' as const,
    latency: `${p.rate_limit_rpm} rpm`
  }));
  return (
    <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800">
      <div className="flex items-center gap-2 mb-6">
        <Server className="w-5 h-5 text-blue-500" />
        <h2 className="text-lg font-semibold">Provider Status</h2>
      </div>

      <div className="space-y-3">
        {displayProviders.length === 0 ? (
          <div className="text-center text-slate-500 py-4">
            Keine Provider verfügbar
          </div>
        ) : (
          displayProviders.map((provider) => (
            <div
              key={provider.name}
              className="flex items-center justify-between p-3 bg-slate-800/50 rounded-xl"
            >
              <div className="flex items-center gap-3">
                <div
                  className={`w-2 h-2 rounded-full ${
                    provider.status === 'active' ? 'bg-green-500' : 'bg-slate-600'
                  }`}
                />
                <div>
                  <div className="font-medium">{provider.name}</div>
                  <div className="text-xs text-slate-500">{provider.model}</div>
                </div>
              </div>
              <div className="text-sm text-slate-400">{provider.latency}</div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
