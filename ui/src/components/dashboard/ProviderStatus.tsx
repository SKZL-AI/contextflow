import { Server } from 'lucide-react';

interface Provider {
  name: string;
  model: string;
  status: 'active' | 'inactive';
  latency: string;
}

const providers: Provider[] = [
  { name: 'Claude', model: 'claude-sonnet-4-20250514', status: 'active', latency: '1.2s' },
  { name: 'OpenAI', model: 'gpt-4-turbo', status: 'active', latency: '0.8s' },
  { name: 'Gemini', model: 'gemini-1.5-pro', status: 'active', latency: '0.9s' },
  { name: 'Ollama', model: 'llama3', status: 'inactive', latency: '-' },
];

export function ProviderStatus() {
  return (
    <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800">
      <div className="flex items-center gap-2 mb-6">
        <Server className="w-5 h-5 text-blue-500" />
        <h2 className="text-lg font-semibold">Provider Status</h2>
      </div>

      <div className="space-y-3">
        {providers.map((provider) => (
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
        ))}
      </div>
    </div>
  );
}
