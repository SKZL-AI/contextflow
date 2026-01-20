import { useState, useRef, useCallback, ChangeEvent, DragEvent } from 'react';
import {
  Database,
  Upload,
  Search,
  Trash2,
  FileText,
  X,
  CheckCircle,
  AlertCircle,
  File,
  Loader2,
} from 'lucide-react';
import * as api from '../services/api';
import type { SessionInfo, SearchResponse } from '../types/api';

// ============================================================================
// Types
// ============================================================================

interface UploadedDocument {
  id: string;
  name: string;
  size: number;
  uploadedAt: Date;
  content: string;
  chunksCount?: number;
}

interface UploadProgress {
  fileName: string;
  progress: number;
  status: 'uploading' | 'processing' | 'complete' | 'error';
  error?: string;
}

interface SearchResultWithHighlight {
  content: string;
  score: number;
  chunk_id: string;
  metadata: Record<string, unknown>;
  highlightedContent?: string;
}

// ============================================================================
// Helper Functions
// ============================================================================

const ACCEPTED_FORMATS = ['.txt', '.md', '.pdf', '.doc', '.docx'];
const ACCEPTED_MIME_TYPES = [
  'text/plain',
  'text/markdown',
  'application/pdf',
  'application/msword',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
];

function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(date: Date): string {
  return new Intl.DateTimeFormat('de-DE', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}

function generateId(): string {
  return Math.random().toString(36).substring(2, 15);
}

function highlightText(content: string, query: string): string {
  if (!query.trim()) return content;
  const words = query.trim().split(/\s+/).filter(w => w.length > 2);
  if (words.length === 0) return content;

  const pattern = new RegExp(`(${words.map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi');
  return content.replace(pattern, '<mark class="bg-violet-500/30 text-violet-200 px-0.5 rounded">$1</mark>');
}

function isAcceptedFile(file: File): boolean {
  const extension = '.' + file.name.split('.').pop()?.toLowerCase();
  return ACCEPTED_FORMATS.includes(extension) || ACCEPTED_MIME_TYPES.includes(file.type);
}

// ============================================================================
// Component
// ============================================================================

export const RAG: React.FC = () => {
  // State
  const [documents, setDocuments] = useState<UploadedDocument[]>([]);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress[]>([]);
  const [sessionInfo, setSessionInfo] = useState<SessionInfo | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResultWithHighlight[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // ============================================================================
  // File Reading
  // ============================================================================

  const readFileContent = useCallback(async (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target?.result as string;
        resolve(content);
      };
      reader.onerror = () => reject(new Error('Failed to read file'));

      // For PDF and Word files, we read as text (backend should handle conversion)
      // In a production app, you'd want proper parsing libraries
      if (file.type === 'application/pdf' || file.type.includes('word')) {
        reader.readAsText(file); // This won't work perfectly for binary formats
        // In production, use pdf.js or mammoth.js for proper parsing
      } else {
        reader.readAsText(file);
      }
    });
  }, []);

  // ============================================================================
  // Document Upload
  // ============================================================================

  const processFiles = useCallback(async (files: FileList | File[]) => {
    const fileArray = Array.from(files).filter(isAcceptedFile);

    if (fileArray.length === 0) {
      setError('No valid files selected. Accepted formats: ' + ACCEPTED_FORMATS.join(', '));
      return;
    }

    setError(null);
    const newProgress: UploadProgress[] = fileArray.map(f => ({
      fileName: f.name,
      progress: 0,
      status: 'uploading',
    }));
    setUploadProgress(prev => [...prev, ...newProgress]);

    const newDocuments: UploadedDocument[] = [];

    for (let i = 0; i < fileArray.length; i++) {
      const file = fileArray[i];
      const progressIndex = uploadProgress.length + i;

      try {
        // Simulate upload progress
        setUploadProgress(prev => prev.map((p, idx) =>
          idx === progressIndex ? { ...p, progress: 30 } : p
        ));

        // Read file content
        const content = await readFileContent(file);

        setUploadProgress(prev => prev.map((p, idx) =>
          idx === progressIndex ? { ...p, progress: 70, status: 'processing' } : p
        ));

        // Estimate chunks (roughly 1 chunk per 500 chars for demo)
        const estimatedChunks = Math.ceil(content.length / 500);

        const doc: UploadedDocument = {
          id: generateId(),
          name: file.name,
          size: file.size,
          uploadedAt: new Date(),
          content,
          chunksCount: estimatedChunks,
        };

        newDocuments.push(doc);

        setUploadProgress(prev => prev.map((p, idx) =>
          idx === progressIndex ? { ...p, progress: 100, status: 'complete' } : p
        ));
      } catch (err) {
        setUploadProgress(prev => prev.map((p, idx) =>
          idx === progressIndex ? {
            ...p,
            status: 'error',
            error: err instanceof Error ? err.message : 'Upload failed'
          } : p
        ));
      }
    }

    if (newDocuments.length > 0) {
      setDocuments(prev => [...prev, ...newDocuments]);
    }

    // Clear completed uploads after delay
    setTimeout(() => {
      setUploadProgress(prev => prev.filter(p => p.status !== 'complete'));
    }, 2000);
  }, [uploadProgress.length, readFileContent]);

  // ============================================================================
  // Event Handlers
  // ============================================================================

  const handleFileSelect = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFiles(e.target.files);
      e.target.value = ''; // Reset input
    }
  }, [processFiles]);

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFiles(e.dataTransfer.files);
    }
  }, [processFiles]);

  const handleDeleteDocument = useCallback((docId: string) => {
    setDocuments(prev => prev.filter(d => d.id !== docId));
    // If session exists and we delete all documents, clear the session
    setSessionInfo(prev => {
      if (prev && documents.length <= 1) {
        return null;
      }
      return prev;
    });
  }, [documents.length]);

  const handleCreateSession = useCallback(async () => {
    if (documents.length === 0) {
      setError('Please upload at least one document first.');
      return;
    }

    setIsCreatingSession(true);
    setError(null);

    try {
      const documentContents = documents.map(d => d.content);
      const response = await api.createSession({
        documents: documentContents,
        metadata: {
          name: 'RAG Session',
          document_names: documents.map(d => d.name),
        },
      });

      setSessionInfo(response.session_info);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create session');
    } finally {
      setIsCreatingSession(false);
    }
  }, [documents]);

  const handleSearch = useCallback(async () => {
    if (!searchQuery.trim()) {
      setError('Please enter a search query.');
      return;
    }

    if (!sessionInfo) {
      setError('Please create a session first by indexing your documents.');
      return;
    }

    setIsSearching(true);
    setError(null);
    setSearchResults([]);

    try {
      const response: SearchResponse = await api.search({
        query: searchQuery,
        session_id: sessionInfo.session_id,
        max_results: 10,
        include_scores: true,
      });

      const resultsWithHighlight: SearchResultWithHighlight[] = response.results.map(r => ({
        ...r,
        highlightedContent: highlightText(r.content, searchQuery),
      }));

      setSearchResults(resultsWithHighlight);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setIsSearching(false);
    }
  }, [searchQuery, sessionInfo]);

  const handleClearUploadError = useCallback((index: number) => {
    setUploadProgress(prev => prev.filter((_, i) => i !== index));
  }, []);

  // ============================================================================
  // Render
  // ============================================================================

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Database className="w-6 h-6 text-violet-500" />
          <h1 className="text-2xl font-bold">RAG Document Management</h1>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="bg-red-900/20 border border-red-800 rounded-xl p-4 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="text-red-200">{error}</p>
          </div>
          <button
            onClick={() => setError(null)}
            className="text-red-400 hover:text-red-300 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      )}

      {/* Session Info Card */}
      <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Database className="w-5 h-5 text-violet-500" />
          Session Info
        </h2>

        <div className="grid grid-cols-4 gap-4">
          <div className="bg-slate-800/50 rounded-xl p-4">
            <p className="text-slate-400 text-sm mb-1">Session ID</p>
            <p className="font-mono text-sm truncate">
              {sessionInfo?.session_id || '-'}
            </p>
          </div>
          <div className="bg-slate-800/50 rounded-xl p-4">
            <p className="text-slate-400 text-sm mb-1">Total Documents</p>
            <p className="text-2xl font-bold text-violet-400">
              {sessionInfo?.document_count ?? documents.length}
            </p>
          </div>
          <div className="bg-slate-800/50 rounded-xl p-4">
            <p className="text-slate-400 text-sm mb-1">Total Tokens</p>
            <p className="text-2xl font-bold text-emerald-400">
              {sessionInfo?.total_tokens?.toLocaleString() ?? '-'}
            </p>
          </div>
          <div className="bg-slate-800/50 rounded-xl p-4">
            <p className="text-slate-400 text-sm mb-1">Total Chunks</p>
            <p className="text-2xl font-bold text-amber-400">
              {sessionInfo?.chunk_count ?? documents.reduce((sum, d) => sum + (d.chunksCount || 0), 0)}
            </p>
          </div>
        </div>

        {!sessionInfo && documents.length > 0 && (
          <div className="mt-4">
            <button
              onClick={handleCreateSession}
              disabled={isCreatingSession}
              className="px-4 py-2 bg-violet-600 hover:bg-violet-500 disabled:bg-violet-800 disabled:cursor-not-allowed rounded-lg font-medium transition-colors flex items-center gap-2"
            >
              {isCreatingSession ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Creating Session...
                </>
              ) : (
                <>
                  <Database className="w-4 h-4" />
                  Create RAG Session
                </>
              )}
            </button>
          </div>
        )}
      </div>

      {/* Two Column Layout */}
      <div className="grid grid-cols-2 gap-6">
        {/* Left Column: Upload & Documents */}
        <div className="space-y-6">
          {/* Document Upload Section */}
          <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Upload className="w-5 h-5 text-violet-500" />
              Document Upload
            </h2>

            {/* Drag & Drop Zone */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`
                border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all
                ${isDragOver
                  ? 'border-violet-500 bg-violet-500/10'
                  : 'border-slate-700 hover:border-slate-600 hover:bg-slate-800/50'
                }
              `}
            >
              <Upload className={`w-12 h-12 mx-auto mb-4 ${isDragOver ? 'text-violet-400' : 'text-slate-500'}`} />
              <p className="text-slate-300 mb-2">
                Drag & drop files here or click to browse
              </p>
              <p className="text-slate-500 text-sm">
                Accepted formats: {ACCEPTED_FORMATS.join(', ')}
              </p>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept={ACCEPTED_FORMATS.join(',')}
              onChange={handleFileSelect}
              className="hidden"
            />

            {/* Upload Progress */}
            {uploadProgress.length > 0 && (
              <div className="mt-4 space-y-2">
                {uploadProgress.map((progress, index) => (
                  <div key={index} className="bg-slate-800 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <FileText className="w-4 h-4 text-slate-400" />
                        <span className="text-sm truncate max-w-[200px]">{progress.fileName}</span>
                      </div>
                      {progress.status === 'complete' && (
                        <CheckCircle className="w-4 h-4 text-emerald-400" />
                      )}
                      {progress.status === 'error' && (
                        <button onClick={() => handleClearUploadError(index)}>
                          <X className="w-4 h-4 text-red-400" />
                        </button>
                      )}
                      {(progress.status === 'uploading' || progress.status === 'processing') && (
                        <Loader2 className="w-4 h-4 text-violet-400 animate-spin" />
                      )}
                    </div>
                    {progress.status !== 'error' ? (
                      <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-violet-500 transition-all duration-300"
                          style={{ width: `${progress.progress}%` }}
                        />
                      </div>
                    ) : (
                      <p className="text-red-400 text-sm">{progress.error}</p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Document List */}
          <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <File className="w-5 h-5 text-violet-500" />
              Uploaded Documents
              <span className="text-sm font-normal text-slate-400">({documents.length})</span>
            </h2>

            {documents.length === 0 ? (
              <div className="text-center py-8 text-slate-500">
                <FileText className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>No documents uploaded yet</p>
              </div>
            ) : (
              <div className="overflow-hidden rounded-xl border border-slate-800">
                <table className="w-full">
                  <thead className="bg-slate-800/50">
                    <tr>
                      <th className="text-left px-4 py-3 text-sm font-medium text-slate-400">Name</th>
                      <th className="text-left px-4 py-3 text-sm font-medium text-slate-400">Size</th>
                      <th className="text-left px-4 py-3 text-sm font-medium text-slate-400">Uploaded</th>
                      <th className="text-left px-4 py-3 text-sm font-medium text-slate-400">Chunks</th>
                      <th className="px-4 py-3"></th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    {documents.map(doc => (
                      <tr key={doc.id} className="hover:bg-slate-800/30 transition-colors">
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-2">
                            <FileText className="w-4 h-4 text-slate-400" />
                            <span className="truncate max-w-[150px]" title={doc.name}>
                              {doc.name}
                            </span>
                          </div>
                        </td>
                        <td className="px-4 py-3 text-slate-400 text-sm">
                          {formatFileSize(doc.size)}
                        </td>
                        <td className="px-4 py-3 text-slate-400 text-sm">
                          {formatDate(doc.uploadedAt)}
                        </td>
                        <td className="px-4 py-3">
                          <span className="px-2 py-1 bg-violet-500/20 text-violet-300 rounded-md text-sm">
                            {doc.chunksCount || '-'}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <button
                            onClick={() => handleDeleteDocument(doc.id)}
                            className="p-1.5 text-slate-400 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
                            title="Delete document"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Search */}
        <div className="space-y-6">
          {/* Search Test Section */}
          <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Search className="w-5 h-5 text-violet-500" />
              Search Test
            </h2>

            {/* Search Input */}
            <div className="flex gap-3 mb-4">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                  placeholder="Enter search query..."
                  className="w-full pl-10 pr-4 py-3 bg-slate-800 border border-slate-700 rounded-xl focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500 transition-colors"
                />
              </div>
              <button
                onClick={handleSearch}
                disabled={isSearching || !sessionInfo}
                className="px-6 py-3 bg-violet-600 hover:bg-violet-500 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-xl font-medium transition-colors flex items-center gap-2"
              >
                {isSearching ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Searching...
                  </>
                ) : (
                  <>
                    <Search className="w-4 h-4" />
                    Search
                  </>
                )}
              </button>
            </div>

            {!sessionInfo && (
              <p className="text-amber-400 text-sm mb-4 flex items-center gap-2">
                <AlertCircle className="w-4 h-4" />
                Create a session first to enable search
              </p>
            )}

            {/* Search Results */}
            <div className="space-y-3 max-h-[500px] overflow-y-auto">
              {searchResults.length === 0 && !isSearching && sessionInfo && (
                <div className="text-center py-8 text-slate-500">
                  <Search className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>Enter a query to search your documents</p>
                </div>
              )}

              {searchResults.map((result, index) => (
                <div
                  key={result.chunk_id || index}
                  className="bg-slate-800/50 rounded-xl p-4 border border-slate-700"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-slate-500 font-mono">
                      Chunk: {result.chunk_id}
                    </span>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-400">Relevance:</span>
                      <span className={`
                        px-2 py-0.5 rounded-full text-xs font-medium
                        ${result.score >= 0.8
                          ? 'bg-emerald-500/20 text-emerald-300'
                          : result.score >= 0.5
                            ? 'bg-amber-500/20 text-amber-300'
                            : 'bg-slate-600/50 text-slate-300'
                        }
                      `}>
                        {(result.score * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  {/* Highlighted Content */}
                  <div
                    className="text-sm text-slate-300 leading-relaxed"
                    dangerouslySetInnerHTML={{
                      __html: result.highlightedContent || result.content
                    }}
                  />

                  {/* Metadata */}
                  {Object.keys(result.metadata || {}).length > 0 && (
                    <div className="mt-3 pt-3 border-t border-slate-700">
                      <p className="text-xs text-slate-500 mb-1">Metadata:</p>
                      <pre className="text-xs text-slate-400 bg-slate-900/50 rounded p-2 overflow-x-auto">
                        {JSON.stringify(result.metadata, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RAG;
