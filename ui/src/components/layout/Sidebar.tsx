import { NavLink } from 'react-router-dom';
import {
  Brain,
  BarChart3,
  Play,
  Server,
  Database,
  Layers,
  Settings,
} from 'lucide-react';

const navItems = [
  { path: '/', label: 'Dashboard', icon: BarChart3 },
  { path: '/process', label: 'Verarbeitung', icon: Play },
  { path: '/providers', label: 'Provider', icon: Server },
  { path: '/rag', label: 'RAG System', icon: Database },
  { path: '/sessions', label: 'Sessions', icon: Layers },
  { path: '/settings', label: 'Einstellungen', icon: Settings },
];

function Sidebar() {
  return (
    <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col h-full">
      {/* Logo Section */}
      <div className="p-4 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="bg-gradient-to-br from-violet-500 to-purple-600 p-2 rounded-lg">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-white font-semibold text-lg">ContextFlow</h1>
            <span className="text-slate-400 text-xs">v0.1.0</span>
          </div>
        </div>
      </div>

      {/* Navigation Section */}
      <nav className="flex-1 p-4">
        <ul className="space-y-1">
          {navItems.map((item) => (
            <li key={item.path}>
              <NavLink
                to={item.path}
                end={item.path === '/'}
                className={({ isActive }) =>
                  `w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-colors ${
                    isActive
                      ? 'bg-gradient-to-r from-violet-500/20 to-purple-500/20 text-violet-400 border border-violet-500/30'
                      : 'text-slate-400 hover:bg-slate-800 hover:text-white border border-transparent'
                  }`
                }
              >
                <item.icon className="w-5 h-5" />
                <span>{item.label}</span>
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      {/* Bottom Section */}
      <div className="p-4 border-t border-slate-800">
        <div className="bg-gradient-to-br from-violet-500/10 to-purple-500/10 rounded-xl p-4">
          <span className="text-slate-400 text-xs uppercase tracking-wide">
            Aktiver Provider
          </span>
          <div className="flex items-center gap-2 mt-2">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
            </span>
            <span className="text-white text-sm font-medium">Claude Sonnet</span>
          </div>
        </div>
      </div>
    </aside>
  );
}

export { Sidebar };
