import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Dashboard, Process, Providers, RAG, Sessions, Settings } from './pages'
import { Sidebar, Header } from './components/layout'

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-slate-950 text-white flex">
        <Sidebar />
        <div className="flex-1 overflow-auto">
          <Header />
          <main className="p-8">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/process" element={<Process />} />
              <Route path="/providers" element={<Providers />} />
              <Route path="/rag" element={<RAG />} />
              <Route path="/sessions" element={<Sessions />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  )
}

export default App
