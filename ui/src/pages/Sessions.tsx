import { useState, useEffect, useCallback } from 'react';
import {
  Layers,
  Plus,
  Trash2,
  Copy,
  Check,
  RefreshCw,
  Clock,
  FileText,
  Hash,
  Loader2,
  AlertCircle,
  X,
} from 'lucide-react';
import * as api from '../services/api';
import type { SessionInfo } from '../types/api';

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Format timestamp for display
 */
function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleString('de-DE', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/**
 * Truncate session ID for display
 */
function truncateId(id: string, length: number = 12): string {
  if (id.length <= length) return id;
  return `${id.substring(0, length)}...`;
}

/**
 * Format number with locale
 */
function formatNumber(num: number): string {
  return num.toLocaleString('de-DE');
}

// ============================================================================
// Types
// ============================================================================

interface CreateSessionModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCreate: (name: string, description: string) => Promise<void>;
  isCreating: boolean;
}

interface SessionDetailsProps {
  session: SessionInfo;
  isActive: boolean;
  onSetActive: () => void;
  onClose: () => void;
}

// ============================================================================
// Create Session Modal Component
// ============================================================================

const CreateSessionModal: React.FC<CreateSessionModalProps> = ({
  isOpen,
  onClose,
  onCreate,
  isCreating,
}) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await onCreate(name, description);
    setName('');
    setDescription('');
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-slate-900 rounded-2xl border border-slate-800 p-6 w-full max-w-md shadow-xl">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-white">Create New Session</h2>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5 text-slate-400" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Session Name (optional)
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My Session"
              className="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Description (optional)
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Session description..."
              rows={3}
              className="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent resize-none"
            />
          </div>

          <div className="flex gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isCreating}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-violet-600 hover:bg-violet-500 disabled:bg-slate-700 text-white rounded-lg transition-colors"
            >
              {isCreating ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Creating...
                </>
              ) : (
                <>
                  <Plus className="w-4 h-4" />
                  Create
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

// ============================================================================
// Session Details Panel Component
// ============================================================================

const SessionDetailsPanel: React.FC<SessionDetailsProps> = ({
  session,
  isActive,
  onSetActive,
  onClose,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopyId = async () => {
    try {
      await navigator.clipboard.writeText(session.session_id);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <Layers className="w-5 h-5 text-violet-500" />
          <h2 className="text-lg font-semibold text-white">Session Details</h2>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded hover:bg-slate-800 transition-colors"
        >
          <X className="w-5 h-5 text-slate-400" />
        </button>
      </div>

      {/* Active Badge */}
      {isActive && (
        <div className="mb-4 px-3 py-2 bg-violet-500/10 border border-violet-500/20 rounded-lg">
          <span className="text-violet-400 text-sm font-medium">
            Currently Active Session
          </span>
        </div>
      )}

      {/* Session ID */}
      <div className="mb-6">
        <label className="block text-xs uppercase tracking-wider text-slate-500 mb-2">
          Session ID
        </label>
        <div className="flex items-center gap-2">
          <code className="flex-1 px-3 py-2 bg-slate-800 rounded-lg text-sm text-slate-300 font-mono overflow-x-auto">
            {session.session_id}
          </code>
          <button
            onClick={handleCopyId}
            className="p-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors"
            title="Copy Session ID"
          >
            {copied ? (
              <Check className="w-4 h-4 text-green-400" />
            ) : (
              <Copy className="w-4 h-4 text-slate-400" />
            )}
          </button>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="p-4 bg-slate-800/50 rounded-xl">
          <div className="flex items-center gap-2 text-slate-400 mb-1">
            <Clock className="w-4 h-4" />
            <span className="text-xs uppercase tracking-wider">Created</span>
          </div>
          <p className="text-white font-medium">{formatTimestamp(session.created_at)}</p>
        </div>

        <div className="p-4 bg-slate-800/50 rounded-xl">
          <div className="flex items-center gap-2 text-slate-400 mb-1">
            <Clock className="w-4 h-4" />
            <span className="text-xs uppercase tracking-wider">Last Accessed</span>
          </div>
          <p className="text-white font-medium">{formatTimestamp(session.last_accessed)}</p>
        </div>

        <div className="p-4 bg-slate-800/50 rounded-xl">
          <div className="flex items-center gap-2 text-slate-400 mb-1">
            <FileText className="w-4 h-4" />
            <span className="text-xs uppercase tracking-wider">Documents</span>
          </div>
          <p className="text-white font-medium">{formatNumber(session.document_count)}</p>
        </div>

        <div className="p-4 bg-slate-800/50 rounded-xl">
          <div className="flex items-center gap-2 text-slate-400 mb-1">
            <Hash className="w-4 h-4" />
            <span className="text-xs uppercase tracking-wider">Tokens</span>
          </div>
          <p className="text-white font-medium">{formatNumber(session.total_tokens)}</p>
        </div>
      </div>

      {/* Chunks */}
      <div className="mb-6 p-4 bg-slate-800/50 rounded-xl">
        <div className="flex items-center gap-2 text-slate-400 mb-1">
          <Layers className="w-4 h-4" />
          <span className="text-xs uppercase tracking-wider">Chunks</span>
        </div>
        <p className="text-white font-medium">{formatNumber(session.chunk_count)}</p>
      </div>

      {/* Metadata */}
      {session.metadata && Object.keys(session.metadata).length > 0 && (
        <div className="mb-6">
          <label className="block text-xs uppercase tracking-wider text-slate-500 mb-2">
            Metadata
          </label>
          <div className="p-4 bg-slate-800/50 rounded-xl">
            <pre className="text-sm text-slate-300 font-mono overflow-x-auto">
              {JSON.stringify(session.metadata, null, 2)}
            </pre>
          </div>
        </div>
      )}

      {/* Set as Active Button */}
      {!isActive && (
        <button
          onClick={onSetActive}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-violet-600 hover:bg-violet-500 text-white rounded-lg transition-colors"
        >
          <Check className="w-4 h-4" />
          Set as Active Session
        </button>
      )}
    </div>
  );
};

// ============================================================================
// Loading State Component
// ============================================================================

const LoadingState: React.FC = () => {
  return (
    <div className="animate-pulse space-y-3">
      {[1, 2, 3].map((i) => (
        <div
          key={i}
          className="h-16 bg-slate-800 rounded-xl"
        />
      ))}
    </div>
  );
};

// ============================================================================
// Empty State Component
// ============================================================================

interface EmptyStateProps {
  onCreateNew: () => void;
}

const EmptyState: React.FC<EmptyStateProps> = ({ onCreateNew }) => {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <Layers className="w-12 h-12 text-slate-600 mb-4" />
      <h3 className="text-lg font-medium text-slate-300 mb-2">No Sessions Found</h3>
      <p className="text-slate-500 max-w-sm mb-6">
        Create your first session to start managing context and documents.
      </p>
      <button
        onClick={onCreateNew}
        className="flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-500 text-white rounded-lg transition-colors"
      >
        <Plus className="w-4 h-4" />
        Create Session
      </button>
    </div>
  );
};

// ============================================================================
// Session Row Component
// ============================================================================

interface SessionRowProps {
  session: SessionInfo;
  isSelected: boolean;
  isActive: boolean;
  onSelect: () => void;
  onDelete: () => void;
  isDeleting: boolean;
}

const SessionRow: React.FC<SessionRowProps> = ({
  session,
  isSelected,
  isActive,
  onSelect,
  onDelete,
  isDeleting,
}) => {
  return (
    <div
      onClick={onSelect}
      className={`p-4 rounded-xl border cursor-pointer transition-all duration-200 ${
        isSelected
          ? 'border-violet-500 bg-violet-500/5'
          : isActive
          ? 'border-violet-500/50 bg-violet-500/5 hover:bg-violet-500/10'
          : 'border-slate-800 bg-slate-900 hover:border-slate-700 hover:bg-slate-800/50'
      }`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4 flex-1 min-w-0">
          {/* Active indicator */}
          <div
            className={`w-2 h-2 rounded-full flex-shrink-0 ${
              isActive ? 'bg-violet-500' : 'bg-slate-600'
            }`}
          />

          {/* Session ID */}
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <code className="text-sm font-mono text-slate-300">
                {truncateId(session.session_id)}
              </code>
              {isActive && (
                <span className="px-2 py-0.5 text-xs font-medium bg-violet-500/20 text-violet-400 rounded">
                  Active
                </span>
              )}
              {typeof session.metadata?.name === 'string' && session.metadata.name && (
                <span className="text-sm text-slate-400">
                  - {session.metadata.name}
                </span>
              )}
            </div>
            <div className="text-xs text-slate-500 mt-1">
              Created {formatTimestamp(session.created_at)}
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="hidden md:flex items-center gap-6 text-sm text-slate-400">
          <div className="flex items-center gap-2" title="Documents">
            <FileText className="w-4 h-4" />
            <span>{session.document_count}</span>
          </div>
          <div className="flex items-center gap-2" title="Tokens">
            <Hash className="w-4 h-4" />
            <span>{formatNumber(session.total_tokens)}</span>
          </div>
          <div className="flex items-center gap-2" title="Chunks">
            <Layers className="w-4 h-4" />
            <span>{session.chunk_count}</span>
          </div>
        </div>

        {/* Delete Button */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          disabled={isDeleting}
          className="ml-4 p-2 text-slate-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors disabled:opacity-50"
          title="Delete Session"
        >
          {isDeleting ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Trash2 className="w-4 h-4" />
          )}
        </button>
      </div>
    </div>
  );
};

// ============================================================================
// Success Message Component
// ============================================================================

interface SuccessMessageProps {
  message: string;
  onDismiss: () => void;
}

const SuccessMessage: React.FC<SuccessMessageProps> = ({ message, onDismiss }) => {
  useEffect(() => {
    const timer = setTimeout(onDismiss, 5000);
    return () => clearTimeout(timer);
  }, [onDismiss]);

  return (
    <div className="flex items-center justify-between p-4 bg-green-500/10 border border-green-500/20 rounded-xl text-green-400">
      <div className="flex items-center gap-3">
        <Check className="w-5 h-5" />
        <span>{message}</span>
      </div>
      <button onClick={onDismiss} className="p-1 hover:bg-green-500/20 rounded">
        <X className="w-4 h-4" />
      </button>
    </div>
  );
};

// ============================================================================
// Main Sessions Component
// ============================================================================

export const Sessions: React.FC = () => {
  // State
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [selectedSession, setSelectedSession] = useState<SessionInfo | null>(null);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // Load sessions from localStorage on mount
  const loadSessions = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Load session IDs from localStorage
      const storedSessionIds = localStorage.getItem('contextflow_sessions');
      const sessionIds: string[] = storedSessionIds ? JSON.parse(storedSessionIds) : [];

      // Fetch each session's details
      const sessionPromises = sessionIds.map(async (id) => {
        try {
          return await api.getSession(id);
        } catch {
          // Session no longer exists, remove from stored list
          return null;
        }
      });

      const results = await Promise.all(sessionPromises);
      const validSessions = results.filter((s): s is SessionInfo => s !== null);

      // Update stored session IDs to only include valid ones
      const validIds = validSessions.map((s) => s.session_id);
      localStorage.setItem('contextflow_sessions', JSON.stringify(validIds));

      setSessions(validSessions);

      // Load active session ID
      const storedActiveId = localStorage.getItem('contextflow_active_session');
      if (storedActiveId && validIds.includes(storedActiveId)) {
        setActiveSessionId(storedActiveId);
      }
    } catch (err) {
      console.error('Failed to load sessions:', err);
      setError('Failed to load sessions. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  // Create new session
  const handleCreateSession = async (name: string, description: string) => {
    setIsCreating(true);
    setError(null);

    try {
      const metadata: Record<string, unknown> = {};
      if (name.trim()) metadata.name = name.trim();
      if (description.trim()) metadata.description = description.trim();

      const response = await api.createSession({
        metadata: Object.keys(metadata).length > 0 ? metadata : undefined,
      });

      // Add to local storage
      const storedSessionIds = localStorage.getItem('contextflow_sessions');
      const sessionIds: string[] = storedSessionIds ? JSON.parse(storedSessionIds) : [];
      sessionIds.unshift(response.session_id);
      localStorage.setItem('contextflow_sessions', JSON.stringify(sessionIds));

      // Add to state
      setSessions((prev) => [response.session_info, ...prev]);
      setShowCreateModal(false);
      setSuccessMessage(`Session "${name || response.session_id}" created successfully!`);
    } catch (err) {
      console.error('Failed to create session:', err);
      setError('Failed to create session. Please try again.');
    } finally {
      setIsCreating(false);
    }
  };

  // Delete session
  const handleDeleteSession = async (sessionId: string) => {
    setDeletingSessionId(sessionId);
    setError(null);

    try {
      await api.deleteSession(sessionId);

      // Remove from local storage
      const storedSessionIds = localStorage.getItem('contextflow_sessions');
      const sessionIds: string[] = storedSessionIds ? JSON.parse(storedSessionIds) : [];
      const updatedIds = sessionIds.filter((id) => id !== sessionId);
      localStorage.setItem('contextflow_sessions', JSON.stringify(updatedIds));

      // Remove from state
      setSessions((prev) => prev.filter((s) => s.session_id !== sessionId));

      // Clear selection if deleted session was selected
      if (selectedSession?.session_id === sessionId) {
        setSelectedSession(null);
      }

      // Clear active if deleted session was active
      if (activeSessionId === sessionId) {
        setActiveSessionId(null);
        localStorage.removeItem('contextflow_active_session');
      }

      setSuccessMessage('Session deleted successfully!');
    } catch (err) {
      console.error('Failed to delete session:', err);
      setError('Failed to delete session. Please try again.');
    } finally {
      setDeletingSessionId(null);
    }
  };

  // Set active session
  const handleSetActive = (sessionId: string) => {
    setActiveSessionId(sessionId);
    localStorage.setItem('contextflow_active_session', sessionId);
    setSuccessMessage('Active session updated!');
  };

  // Stats
  const totalDocuments = sessions.reduce((sum, s) => sum + s.document_count, 0);
  const totalTokens = sessions.reduce((sum, s) => sum + s.total_tokens, 0);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-lg font-semibold text-white">Session Management</h1>
            <p className="text-sm text-slate-400 mt-1">
              Manage context sessions for persistent document processing
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={loadSessions}
              disabled={isLoading}
              className="flex items-center gap-2 px-3 py-1.5 text-sm bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-300 rounded-lg transition-colors"
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <RefreshCw className="w-4 h-4" />
              )}
              Refresh
            </button>
            <button
              onClick={() => setShowCreateModal(true)}
              className="flex items-center gap-2 px-4 py-1.5 text-sm bg-violet-600 hover:bg-violet-500 text-white rounded-lg transition-colors"
            >
              <Plus className="w-4 h-4" />
              New Session
            </button>
          </div>
        </div>

        {/* Stats Row */}
        {sessions.length > 0 && (
          <div className="flex items-center gap-6 text-sm">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-slate-500" />
              <span className="text-slate-400">
                {sessions.length} session{sessions.length !== 1 ? 's' : ''}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <FileText className="w-4 h-4 text-slate-500" />
              <span className="text-slate-400">{formatNumber(totalDocuments)} documents</span>
            </div>
            <div className="flex items-center gap-2">
              <Hash className="w-4 h-4 text-slate-500" />
              <span className="text-slate-400">{formatNumber(totalTokens)} tokens</span>
            </div>
            {activeSessionId && (
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-violet-500" />
                <span className="text-slate-400">
                  Active: <span className="text-violet-400">{truncateId(activeSessionId)}</span>
                </span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Success Message */}
      {successMessage && (
        <SuccessMessage
          message={successMessage}
          onDismiss={() => setSuccessMessage(null)}
        />
      )}

      {/* Error Display */}
      {error && (
        <div className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <p>{error}</p>
          <button
            onClick={() => setError(null)}
            className="ml-auto p-1 hover:bg-red-500/20 rounded"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Session List */}
        <div className="lg:col-span-2 bg-slate-900 rounded-2xl p-6 border border-slate-800">
          <div className="flex items-center gap-2 mb-4">
            <Layers className="w-5 h-5 text-violet-500" />
            <h2 className="text-lg font-semibold text-white">Sessions</h2>
          </div>

          {isLoading ? (
            <LoadingState />
          ) : sessions.length === 0 ? (
            <EmptyState onCreateNew={() => setShowCreateModal(true)} />
          ) : (
            <div className="space-y-3">
              {sessions.map((session) => (
                <SessionRow
                  key={session.session_id}
                  session={session}
                  isSelected={selectedSession?.session_id === session.session_id}
                  isActive={activeSessionId === session.session_id}
                  onSelect={() => setSelectedSession(session)}
                  onDelete={() => handleDeleteSession(session.session_id)}
                  isDeleting={deletingSessionId === session.session_id}
                />
              ))}
            </div>
          )}
        </div>

        {/* Details Panel */}
        <div className="lg:col-span-1">
          {selectedSession ? (
            <SessionDetailsPanel
              session={selectedSession}
              isActive={activeSessionId === selectedSession.session_id}
              onSetActive={() => handleSetActive(selectedSession.session_id)}
              onClose={() => setSelectedSession(null)}
            />
          ) : (
            <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800">
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <Layers className="w-12 h-12 text-slate-700 mb-4" />
                <p className="text-slate-500">
                  Select a session to view details
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Create Session Modal */}
      <CreateSessionModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onCreate={handleCreateSession}
        isCreating={isCreating}
      />
    </div>
  );
};

export default Sessions;
