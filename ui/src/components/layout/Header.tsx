import { useLocation } from 'react-router-dom';

interface RouteInfo {
  title: string;
  subtitle: string;
}

const routeMap: Record<string, RouteInfo> = {
  '/': {
    title: 'Dashboard',
    subtitle: 'Willkommen zurück! Hier ist dein Überblick.',
  },
  '/process': {
    title: 'Verarbeitung',
    subtitle: 'Tasks erstellen und verarbeiten',
  },
  '/providers': {
    title: 'Provider',
    subtitle: 'LLM Provider verwalten',
  },
  '/rag': {
    title: 'RAG System',
    subtitle: 'Retrieval Augmented Generation',
  },
  '/sessions': {
    title: 'Sessions',
    subtitle: 'Aktive Sessions verwalten',
  },
  '/settings': {
    title: 'Einstellungen',
    subtitle: 'Konfiguration anpassen',
  },
};

function Header() {
  const location = useLocation();

  const currentRoute = routeMap[location.pathname] || {
    title: 'ContextFlow',
    subtitle: 'AI-powered context management',
  };

  return (
    <header className="sticky top-0 z-10 bg-slate-950/80 backdrop-blur-xl border-b border-slate-800 px-8 py-4">
      <div className="flex items-center justify-between">
        {/* Left side - Page title and subtitle */}
        <div>
          <h2 className="text-2xl font-bold">{currentRoute.title}</h2>
          <p className="text-slate-500">{currentRoute.subtitle}</p>
        </div>

        {/* Right side - System status badge */}
        <div className="flex items-center gap-2 px-4 py-2 bg-slate-800 rounded-xl">
          <div className="w-2 h-2 rounded-full bg-green-500" />
          <span>Alle Systeme aktiv</span>
        </div>
      </div>
    </header>
  );
}

export { Header };
