import { useState } from 'react'
import { FileText, Database, Play } from 'lucide-react'

interface TaskInputProps {
  onSubmit: (task: string) => void
  isProcessing: boolean
  ragEnabled?: boolean
  onToggleRag?: () => void
}

export function TaskInput({ onSubmit, isProcessing, ragEnabled = false, onToggleRag }: TaskInputProps) {
  const [task, setTask] = useState('')

  const handleSubmit = () => {
    if (task.trim() && !isProcessing) {
      onSubmit(task)
    }
  }

  return (
    <div className="space-y-4">
      <div className="bg-slate-800 rounded-xl p-4">
        <textarea
          value={task}
          onChange={(e) => setTask(e.target.value)}
          placeholder="Gib deinen Task ein oder lade ein Dokument hoch..."
          className="w-full bg-transparent text-white placeholder-slate-500 resize-none outline-none h-24"
          disabled={isProcessing}
        />
      </div>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button className="flex items-center gap-2 px-4 py-2 bg-slate-800 rounded-xl hover:bg-slate-700 transition-all">
            <FileText className="w-4 h-4" />
            <span className="text-sm">Dokument</span>
          </button>
          <button
            onClick={onToggleRag}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all ${
              ragEnabled
                ? 'bg-violet-500/20 text-violet-400 border border-violet-500/30'
                : 'bg-slate-800 hover:bg-slate-700'
            }`}
          >
            <Database className="w-4 h-4" />
            <span className="text-sm">RAG aktivieren</span>
          </button>
        </div>
        <button
          onClick={handleSubmit}
          disabled={isProcessing || !task.trim()}
          className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-violet-500 to-purple-600 rounded-xl font-semibold hover:opacity-90 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isProcessing ? (
            <>
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>Verarbeite...</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              <span>Starten</span>
            </>
          )}
        </button>
      </div>
    </div>
  )
}
