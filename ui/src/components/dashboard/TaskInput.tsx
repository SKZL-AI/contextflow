import { useState, useRef, ChangeEvent } from 'react'
import { FileText, Database, Play, X, Upload } from 'lucide-react'

interface TaskInputProps {
  onSubmit: (task: string, documents?: string[]) => void
  isProcessing: boolean
  ragEnabled?: boolean
  onToggleRag?: () => void
}

export function TaskInput({ onSubmit, isProcessing, ragEnabled = false, onToggleRag }: TaskInputProps) {
  const [task, setTask] = useState('')
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [isReadingFiles, setIsReadingFiles] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files)
      setSelectedFiles(prev => [...prev, ...newFiles])
    }
    // Reset input so the same file can be selected again
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleRemoveFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index))
  }

  const handleSubmit = async () => {
    if ((!task.trim() && selectedFiles.length === 0) || isProcessing) {
      return
    }

    setIsReadingFiles(true)
    try {
      // Read file contents
      let documents: string[] | undefined
      if (selectedFiles.length > 0) {
        documents = await Promise.all(
          selectedFiles.map(file => file.text())
        )
      }

      onSubmit(task, documents)

      // Clear after submit
      setTask('')
      setSelectedFiles([])
    } catch (error) {
      console.error('Error reading files:', error)
    } finally {
      setIsReadingFiles(false)
    }
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const isSubmitDisabled = isProcessing || isReadingFiles || (!task.trim() && selectedFiles.length === 0)

  return (
    <div className="space-y-4">
      {/* Task Input */}
      <div className="bg-slate-800 rounded-xl p-4">
        <textarea
          value={task}
          onChange={(e) => setTask(e.target.value)}
          placeholder="Gib deinen Task ein oder lade ein Dokument hoch..."
          className="w-full bg-transparent text-white placeholder-slate-500 resize-none outline-none h-24"
          disabled={isProcessing}
        />
      </div>

      {/* Selected Files */}
      {selectedFiles.length > 0 && (
        <div className="bg-slate-800/50 rounded-xl p-3 space-y-2">
          <div className="text-xs text-slate-400 mb-2">
            {selectedFiles.length} Dokument{selectedFiles.length > 1 ? 'e' : ''} ausgewählt
          </div>
          {selectedFiles.map((file, index) => (
            <div
              key={`${file.name}-${index}`}
              className="flex items-center justify-between bg-slate-700/50 rounded-lg px-3 py-2"
            >
              <div className="flex items-center gap-2 min-w-0">
                <FileText className="w-4 h-4 text-violet-400 flex-shrink-0" />
                <span className="text-sm truncate">{file.name}</span>
                <span className="text-xs text-slate-500 flex-shrink-0">
                  ({formatFileSize(file.size)})
                </span>
              </div>
              <button
                onClick={() => handleRemoveFile(index)}
                className="p-1 hover:bg-slate-600 rounded transition-colors flex-shrink-0"
                disabled={isProcessing}
              >
                <X className="w-4 h-4 text-slate-400 hover:text-red-400" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          {/* Hidden File Input */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".txt,.md,.pdf,.doc,.docx,.json,.csv,.xml,.html,.py,.js,.ts,.tsx,.jsx"
            onChange={handleFileSelect}
            className="hidden"
            disabled={isProcessing}
          />

          {/* Document Upload Button */}
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isProcessing}
            className="flex items-center gap-2 px-4 py-2 bg-slate-800 rounded-xl hover:bg-slate-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Upload className="w-4 h-4" />
            <span className="text-sm">Dokument</span>
          </button>

          {/* RAG Toggle */}
          <button
            onClick={onToggleRag}
            disabled={isProcessing}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all disabled:opacity-50 ${
              ragEnabled
                ? 'bg-violet-500/20 text-violet-400 border border-violet-500/30'
                : 'bg-slate-800 hover:bg-slate-700'
            }`}
          >
            <Database className="w-4 h-4" />
            <span className="text-sm">RAG aktivieren</span>
          </button>
        </div>

        {/* Submit Button */}
        <button
          onClick={handleSubmit}
          disabled={isSubmitDisabled}
          className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-violet-500 to-purple-600 rounded-xl font-semibold hover:opacity-90 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isProcessing || isReadingFiles ? (
            <>
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>{isReadingFiles ? 'Lese Dateien...' : 'Verarbeite...'}</span>
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
