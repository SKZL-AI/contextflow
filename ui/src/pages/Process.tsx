import { useState, useRef, useEffect, ChangeEvent } from 'react';
import {
  Play,
  Square,
  Copy,
  FileText,
  X,
  Upload,
  Check,
  AlertCircle,
  Brain,
  Zap,
  Layers,
  Target,
} from 'lucide-react';
import { processStream } from '../services/api';
import type { Strategy } from '../types/api';

interface StrategyOption {
  id: Strategy;
  name: string;
  description: string;
  icon: React.ReactNode;
}

interface StreamMetadata {
  tokenCount: number;
  estimatedCost: number;
  strategyUsed: Strategy | null;
  processingTime: number;
  verificationPassed: boolean | null;
}

const strategies: StrategyOption[] = [
  {
    id: 'auto',
    name: 'Auto',
    description: 'Automatische Strategie-Auswahl',
    icon: <Brain className="w-5 h-5" />,
  },
  {
    id: 'gsd',
    name: 'GSD',
    description: 'Kurze Kontexte (<10K Tokens)',
    icon: <Zap className="w-5 h-5" />,
  },
  {
    id: 'ralph',
    name: 'RALPH',
    description: 'Mittlere Kontexte (10K-100K)',
    icon: <Layers className="w-5 h-5" />,
  },
  {
    id: 'rlm',
    name: 'RLM',
    description: 'Lange Kontexte (>100K)',
    icon: <Target className="w-5 h-5" />,
  },
];

export const Process: React.FC = () => {
  // Task input state
  const [task, setTask] = useState('');
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy>('auto');
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isReadingFiles, setIsReadingFiles] = useState(false);

  // Processing state
  const [isProcessing, setIsProcessing] = useState(false);
  const [output, setOutput] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [copySuccess, setCopySuccess] = useState(false);

  // Metadata state
  const [metadata, setMetadata] = useState<StreamMetadata>({
    tokenCount: 0,
    estimatedCost: 0,
    strategyUsed: null,
    processingTime: 0,
    verificationPassed: null,
  });

  // Refs
  const fileInputRef = useRef<HTMLInputElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const outputRef = useRef<HTMLDivElement>(null);
  const startTimeRef = useRef<number>(0);
  const timerIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
      }
    };
  }, []);

  // Auto-scroll output panel
  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [output]);

  const handleFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files);
      setSelectedFiles((prev) => [...prev, ...newFiles]);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleRemoveFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatTime = (ms: number): string => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  const formatCost = (cost: number): string => {
    if (cost < 0.01) return `$${cost.toFixed(4)}`;
    return `$${cost.toFixed(2)}`;
  };

  const startProcessing = async () => {
    if ((!task.trim() && selectedFiles.length === 0) || isProcessing) {
      return;
    }

    // Reset state
    setIsProcessing(true);
    setOutput('');
    setError(null);
    setCopySuccess(false);
    setMetadata({
      tokenCount: 0,
      estimatedCost: 0,
      strategyUsed: null,
      processingTime: 0,
      verificationPassed: null,
    });

    // Read files if any
    setIsReadingFiles(true);
    let documents: string[] | undefined;
    try {
      if (selectedFiles.length > 0) {
        documents = await Promise.all(selectedFiles.map((file) => file.text()));
      }
    } catch (err) {
      setError('Fehler beim Lesen der Dateien');
      setIsProcessing(false);
      setIsReadingFiles(false);
      return;
    }
    setIsReadingFiles(false);

    // Start timer
    startTimeRef.current = Date.now();
    timerIntervalRef.current = setInterval(() => {
      setMetadata((prev) => ({
        ...prev,
        processingTime: Date.now() - startTimeRef.current,
      }));
    }, 100);

    // Create EventSource for streaming
    try {
      const eventSource = processStream({
        task,
        strategy: selectedStrategy,
        documents,
      });

      eventSourceRef.current = eventSource;

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'chunk') {
            setOutput((prev) => prev + data.content);
            // Update token count estimate during streaming
            if (data.tokens) {
              setMetadata((prev) => ({
                ...prev,
                tokenCount: prev.tokenCount + data.tokens,
              }));
            }
          } else if (data.type === 'metadata') {
            // Update metadata from stream
            setMetadata((prev) => ({
              ...prev,
              tokenCount: data.token_count ?? prev.tokenCount,
              estimatedCost: data.cost_usd ?? prev.estimatedCost,
              strategyUsed: data.strategy_used ?? prev.strategyUsed,
            }));
          } else if (data.type === 'done') {
            // Final metadata update
            if (timerIntervalRef.current) {
              clearInterval(timerIntervalRef.current);
            }
            setMetadata((prev) => ({
              ...prev,
              tokenCount: data.total_tokens ?? prev.tokenCount,
              estimatedCost: data.cost_usd ?? prev.estimatedCost,
              strategyUsed: data.strategy_used ?? prev.strategyUsed,
              processingTime: data.execution_time ?? Date.now() - startTimeRef.current,
              verificationPassed: data.verification_passed ?? null,
            }));
            setIsProcessing(false);
            eventSource.close();
          } else if (data.type === 'error') {
            setError(data.error || 'Ein Fehler ist aufgetreten');
            setIsProcessing(false);
            if (timerIntervalRef.current) {
              clearInterval(timerIntervalRef.current);
            }
            eventSource.close();
          }
        } catch (parseError) {
          console.error('Error parsing SSE data:', parseError);
        }
      };

      eventSource.onerror = () => {
        setError('Verbindungsfehler zum Server');
        setIsProcessing(false);
        if (timerIntervalRef.current) {
          clearInterval(timerIntervalRef.current);
        }
        eventSource.close();
      };
    } catch (err) {
      setError('Fehler beim Starten der Verarbeitung');
      setIsProcessing(false);
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
      }
    }
  };

  const cancelProcessing = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
    }
    setIsProcessing(false);
  };

  const copyOutput = async () => {
    try {
      await navigator.clipboard.writeText(output);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const isStartDisabled =
    isProcessing || isReadingFiles || (!task.trim() && selectedFiles.length === 0);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center gap-3">
        <div className="p-2 bg-violet-500/20 rounded-xl">
          <Brain className="w-6 h-6 text-violet-400" />
        </div>
        <div>
          <h1 className="text-2xl font-bold">Task-Verarbeitung</h1>
          <p className="text-slate-400 text-sm">
            Verarbeite Tasks mit Streaming-Ausgabe
          </p>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Left Column: Input Section */}
        <div className="col-span-2 space-y-6">
          {/* Task Input */}
          <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800">
            <h2 className="text-lg font-semibold mb-4">Task-Eingabe</h2>

            <textarea
              value={task}
              onChange={(e) => setTask(e.target.value)}
              placeholder="Beschreibe deinen Task hier..."
              className="w-full bg-slate-800 rounded-xl p-4 text-white placeholder-slate-500 resize-none outline-none focus:ring-2 focus:ring-violet-500/50 h-32 mb-4"
              disabled={isProcessing}
            />

            {/* Strategy Selector */}
            <div className="mb-4">
              <label className="text-sm text-slate-400 mb-2 block">Strategie</label>
              <div className="grid grid-cols-4 gap-2">
                {strategies.map((strategy) => (
                  <button
                    key={strategy.id}
                    onClick={() => setSelectedStrategy(strategy.id)}
                    disabled={isProcessing}
                    className={`flex flex-col items-center gap-1 p-3 rounded-xl transition-all ${
                      selectedStrategy === strategy.id
                        ? 'bg-violet-500/20 text-violet-400 border border-violet-500/50'
                        : 'bg-slate-800 hover:bg-slate-700 border border-transparent'
                    } disabled:opacity-50`}
                  >
                    {strategy.icon}
                    <span className="text-sm font-medium">{strategy.name}</span>
                    <span className="text-xs text-slate-500 text-center">
                      {strategy.description}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* File Upload */}
            <div className="mb-4">
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".txt,.md,.pdf,.doc,.docx,.json,.csv,.xml,.html,.py,.js,.ts,.tsx,.jsx"
                onChange={handleFileSelect}
                className="hidden"
                disabled={isProcessing}
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isProcessing}
                className="flex items-center gap-2 px-4 py-2 bg-slate-800 rounded-xl hover:bg-slate-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Upload className="w-4 h-4" />
                <span className="text-sm">Dokumente hochladen</span>
              </button>
            </div>

            {/* Selected Files */}
            {selectedFiles.length > 0 && (
              <div className="bg-slate-800/50 rounded-xl p-3 space-y-2 mb-4">
                <div className="text-xs text-slate-400 mb-2">
                  {selectedFiles.length} Dokument{selectedFiles.length > 1 ? 'e' : ''}{' '}
                  ausgewählt
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
            <div className="flex items-center gap-3">
              <button
                onClick={startProcessing}
                disabled={isStartDisabled}
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

              {isProcessing && (
                <button
                  onClick={cancelProcessing}
                  className="flex items-center gap-2 px-4 py-3 bg-red-500/20 text-red-400 rounded-xl hover:bg-red-500/30 transition-all"
                >
                  <Square className="w-4 h-4" />
                  <span>Abbrechen</span>
                </button>
              )}
            </div>
          </div>

          {/* Output Panel */}
          <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">Ausgabe</h2>
              {output && (
                <button
                  onClick={copyOutput}
                  className="flex items-center gap-2 px-3 py-1.5 bg-slate-800 rounded-lg hover:bg-slate-700 transition-all text-sm"
                >
                  {copySuccess ? (
                    <>
                      <Check className="w-4 h-4 text-green-400" />
                      <span className="text-green-400">Kopiert!</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-4 h-4" />
                      <span>Kopieren</span>
                    </>
                  )}
                </button>
              )}
            </div>

            {/* Error Display */}
            {error && (
              <div className="flex items-center gap-2 mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400">
                <AlertCircle className="w-5 h-5 flex-shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {/* Output Area */}
            <div
              ref={outputRef}
              className="bg-slate-950 rounded-xl p-4 min-h-[300px] max-h-[500px] overflow-y-auto font-mono text-sm leading-relaxed"
            >
              {output ? (
                <pre className="whitespace-pre-wrap break-words text-slate-200">
                  {output}
                  {isProcessing && (
                    <span className="inline-block w-2 h-4 bg-violet-400 ml-0.5 animate-pulse" />
                  )}
                </pre>
              ) : (
                <div className="text-slate-500 text-center py-12">
                  {isProcessing ? (
                    <div className="flex flex-col items-center gap-3">
                      <div className="w-8 h-8 border-3 border-slate-700 border-t-violet-500 rounded-full animate-spin" />
                      <span>Warte auf Antwort...</span>
                    </div>
                  ) : (
                    'Die Ausgabe erscheint hier...'
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Column: Metadata Panel */}
        <div className="col-span-1">
          <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800 sticky top-6">
            <h2 className="text-lg font-semibold mb-4">Metadaten</h2>

            <div className="space-y-4">
              {/* Token Count */}
              <div className="bg-slate-800/50 rounded-xl p-4">
                <div className="text-xs text-slate-400 mb-1">Token</div>
                <div className="text-2xl font-bold text-violet-400">
                  {metadata.tokenCount.toLocaleString()}
                </div>
              </div>

              {/* Estimated Cost */}
              <div className="bg-slate-800/50 rounded-xl p-4">
                <div className="text-xs text-slate-400 mb-1">Geschätzte Kosten</div>
                <div className="text-2xl font-bold text-emerald-400">
                  {formatCost(metadata.estimatedCost)}
                </div>
              </div>

              {/* Strategy Used */}
              <div className="bg-slate-800/50 rounded-xl p-4">
                <div className="text-xs text-slate-400 mb-1">Verwendete Strategie</div>
                <div className="text-lg font-semibold">
                  {metadata.strategyUsed ? (
                    <span className="text-violet-400 uppercase">
                      {metadata.strategyUsed}
                    </span>
                  ) : (
                    <span className="text-slate-500">-</span>
                  )}
                </div>
              </div>

              {/* Processing Time */}
              <div className="bg-slate-800/50 rounded-xl p-4">
                <div className="text-xs text-slate-400 mb-1">Verarbeitungszeit</div>
                <div className="text-lg font-semibold">
                  {metadata.processingTime > 0 ? (
                    formatTime(metadata.processingTime)
                  ) : (
                    <span className="text-slate-500">-</span>
                  )}
                </div>
              </div>

              {/* Verification Status */}
              <div className="bg-slate-800/50 rounded-xl p-4">
                <div className="text-xs text-slate-400 mb-1">Verifizierung</div>
                {metadata.verificationPassed === null ? (
                  <span className="text-slate-500">-</span>
                ) : metadata.verificationPassed ? (
                  <div className="flex items-center gap-2 text-green-400">
                    <Check className="w-5 h-5" />
                    <span className="font-semibold">Bestanden</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-2 text-red-400">
                    <AlertCircle className="w-5 h-5" />
                    <span className="font-semibold">Fehlgeschlagen</span>
                  </div>
                )}
              </div>
            </div>

            {/* Processing Status Indicator */}
            {isProcessing && (
              <div className="mt-4 flex items-center gap-2 text-violet-400 text-sm">
                <div className="w-2 h-2 bg-violet-400 rounded-full animate-pulse" />
                <span>Streaming aktiv...</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Process;
