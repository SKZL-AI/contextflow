"""
Session Manager for ContextFlow.

Inspired by Claude-Mem: Persistent memory across sessions.

Features:
- SQLite-backed session history
- Observation tracking and compression
- Relevant context retrieval for future sessions
- Session lifecycle management
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any

from contextflow.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class ObservationType(str, Enum):
    """Types of observations that can be recorded during a session."""

    TASK = "task"  # Task executed
    RESULT = "result"  # Task result
    ERROR = "error"  # Error occurred
    INSIGHT = "insight"  # Learned insight
    PREFERENCE = "preference"  # User preference
    CONTEXT = "context"  # Context snippet
    VERIFICATION = "verification"  # Verification result


# Default token estimation: ~4 characters per token
CHARS_PER_TOKEN = 4


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Observation:
    """
    Single observation in a session.

    Observations are the atomic units of memory - they capture
    what happened during a session for future reference.
    """

    id: str
    session_id: str
    type: ObservationType
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 1.0  # For retrieval ranking
    compressed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert observation to dictionary for serialization."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "type": self.type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "token_count": self.token_count,
            "metadata": self.metadata,
            "relevance_score": self.relevance_score,
            "compressed": self.compressed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        """Create observation from dictionary."""
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            type=ObservationType(data["type"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            token_count=data.get("token_count", 0),
            metadata=data.get("metadata", {}),
            relevance_score=data.get("relevance_score", 1.0),
            compressed=data.get("compressed", False),
        )


@dataclass
class Session:
    """
    Session with observations.

    A session represents a single interaction period with the system.
    It contains multiple observations and can be summarized for
    efficient retrieval.
    """

    id: str
    started_at: datetime
    ended_at: datetime | None = None
    observations: list[Observation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    summary: str | None = None
    total_tokens: int = 0

    @property
    def is_active(self) -> bool:
        """Check if session is still active."""
        return self.ended_at is None

    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        end = self.ended_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "id": self.id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "observations": [obs.to_dict() for obs in self.observations],
            "metadata": self.metadata,
            "summary": self.summary,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        """Create session from dictionary."""
        return cls(
            id=data["id"],
            started_at=datetime.fromisoformat(data["started_at"]),
            ended_at=(datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None),
            observations=[Observation.from_dict(obs) for obs in data.get("observations", [])],
            metadata=data.get("metadata", {}),
            summary=data.get("summary"),
            total_tokens=data.get("total_tokens", 0),
        )


@dataclass
class SessionContext:
    """
    Context retrieved for a new session.

    Contains relevant observations from past sessions that may
    be useful for the current task.
    """

    observations: list[Observation]
    total_tokens: int
    sessions_referenced: int
    retrieval_query: str

    def to_prompt_format(self) -> str:
        """
        Format context for inclusion in a prompt.

        Returns:
            Formatted string with relevant observations.
        """
        if not self.observations:
            return ""

        lines = ["## Relevant Context from Previous Sessions\n"]

        # Group by session
        by_session: dict[str, list[Observation]] = {}
        for obs in self.observations:
            if obs.session_id not in by_session:
                by_session[obs.session_id] = []
            by_session[obs.session_id].append(obs)

        for session_id, obs_list in by_session.items():
            lines.append(f"### Session {session_id[:8]}...")
            for obs in obs_list:
                type_label = obs.type.value.upper()
                lines.append(f"- [{type_label}] {obs.content}")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# Session Manager
# =============================================================================


class SessionManager:
    """
    Manages sessions and observations for persistent memory.

    Inspired by Claude-Mem pattern for cross-session context.

    Usage:
        manager = SessionManager(db_path="~/.contextflow/sessions.db")

        # Start new session
        session = await manager.start_session(metadata={"project": "myapp"})

        # Add observations during work
        await manager.add_observation(
            session.id,
            ObservationType.TASK,
            "Analyzed API endpoints"
        )

        await manager.add_observation(
            session.id,
            ObservationType.INSIGHT,
            "The API uses REST with JWT auth"
        )

        # End session
        await manager.end_session(session.id)

        # Later: Get relevant context for new session
        context = await manager.get_relevant_context(
            query="How does authentication work?",
            max_tokens=2000
        )
    """

    def __init__(
        self,
        db_path: str = "~/.contextflow/sessions.db",
        max_observations_per_session: int = 1000,
        compress_threshold_tokens: int = 5000,
    ):
        """
        Initialize SessionManager.

        Args:
            db_path: Path to SQLite database
            max_observations_per_session: Max observations before compression
            compress_threshold_tokens: Token threshold for compression
        """
        self.db_path = Path(db_path).expanduser()
        self.max_observations_per_session = max_observations_per_session
        self.compress_threshold_tokens = compress_threshold_tokens

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-safe connection management
        self._lock = Lock()
        self._connection: sqlite3.Connection | None = None

        # Active sessions cache
        self._active_sessions: dict[str, Session] = {}

        # Initialize database
        self._init_db()

        logger.info(
            "SessionManager initialized",
            db_path=str(self.db_path),
            max_observations=max_observations_per_session,
        )

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                isolation_level="DEFERRED",
            )
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Sessions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    metadata TEXT DEFAULT '{}',
                    summary TEXT,
                    total_tokens INTEGER DEFAULT 0
                )
            """
            )

            # Observations table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS observations (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    token_count INTEGER DEFAULT 0,
                    metadata TEXT DEFAULT '{}',
                    relevance_score REAL DEFAULT 1.0,
                    compressed INTEGER DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                        ON DELETE CASCADE
                )
            """
            )

            # Indexes for efficient retrieval
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_observations_session
                ON observations(session_id)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_observations_type
                ON observations(type)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_observations_timestamp
                ON observations(timestamp DESC)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_started
                ON sessions(started_at DESC)
            """
            )

            conn.commit()

    async def start_session(self, metadata: dict[str, Any] | None = None) -> Session:
        """
        Start a new session.

        Args:
            metadata: Optional metadata for the session

        Returns:
            Newly created Session
        """
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()

        session = Session(
            id=session_id,
            started_at=now,
            metadata=metadata or {},
        )

        # Store in database
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO sessions (id, started_at, metadata)
                VALUES (?, ?, ?)
                """,
                (session_id, now.isoformat(), json.dumps(session.metadata)),
            )
            conn.commit()

        # Cache active session
        self._active_sessions[session_id] = session

        logger.info("Session started", session_id=session_id[:8])
        return session

    async def end_session(
        self,
        session_id: str,
        generate_summary: bool = True,
    ) -> Session:
        """
        End a session and optionally generate summary.

        Args:
            session_id: Session ID to end
            generate_summary: Whether to generate a summary

        Returns:
            Updated Session with end time and summary

        Raises:
            ValueError: If session not found
        """
        session = await self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        now = datetime.utcnow()
        session.ended_at = now

        # Generate summary if requested
        if generate_summary and session.observations:
            session.summary = await self._generate_session_summary(session)

        # Calculate total tokens
        session.total_tokens = sum(obs.token_count for obs in session.observations)

        # Update database
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE sessions
                SET ended_at = ?, summary = ?, total_tokens = ?
                WHERE id = ?
                """,
                (
                    now.isoformat(),
                    session.summary,
                    session.total_tokens,
                    session_id,
                ),
            )
            conn.commit()

        # Remove from active cache
        self._active_sessions.pop(session_id, None)

        logger.info(
            "Session ended",
            session_id=session_id[:8],
            duration_seconds=session.duration_seconds,
            total_tokens=session.total_tokens,
        )
        return session

    async def add_observation(
        self,
        session_id: str,
        obs_type: ObservationType,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Observation:
        """
        Add an observation to a session.

        Args:
            session_id: Session to add observation to
            obs_type: Type of observation
            content: Observation content
            metadata: Optional metadata

        Returns:
            Created Observation

        Raises:
            ValueError: If session not found or not active
        """
        # Verify session exists
        session = await self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        if not session.is_active:
            raise ValueError(f"Session is not active: {session_id}")

        # Estimate token count
        token_count = len(content) // CHARS_PER_TOKEN

        observation = Observation(
            id=str(uuid.uuid4()),
            session_id=session_id,
            type=obs_type,
            content=content,
            timestamp=datetime.utcnow(),
            token_count=token_count,
            metadata=metadata or {},
        )

        # Store in database
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO observations
                (id, session_id, type, content, timestamp, token_count,
                 metadata, relevance_score, compressed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation.id,
                    session_id,
                    obs_type.value,
                    content,
                    observation.timestamp.isoformat(),
                    token_count,
                    json.dumps(observation.metadata),
                    observation.relevance_score,
                    0,
                ),
            )
            conn.commit()

        # Update cached session
        if session_id in self._active_sessions:
            self._active_sessions[session_id].observations.append(observation)

        # Check if compression needed
        await self._check_compression(session_id)

        logger.debug(
            "Observation added",
            session_id=session_id[:8],
            type=obs_type.value,
            tokens=token_count,
        )
        return observation

    async def get_session(self, session_id: str) -> Session | None:
        """
        Get session by ID.

        Args:
            session_id: Session ID to retrieve

        Returns:
            Session if found, None otherwise
        """
        # Check cache first
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]

        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Get session
            cursor.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            # Get observations
            cursor.execute(
                "SELECT * FROM observations WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            )
            obs_rows = cursor.fetchall()

        observations = [
            Observation(
                id=obs["id"],
                session_id=obs["session_id"],
                type=ObservationType(obs["type"]),
                content=obs["content"],
                timestamp=datetime.fromisoformat(obs["timestamp"]),
                token_count=obs["token_count"],
                metadata=json.loads(obs["metadata"]),
                relevance_score=obs["relevance_score"],
                compressed=bool(obs["compressed"]),
            )
            for obs in obs_rows
        ]

        return Session(
            id=row["id"],
            started_at=datetime.fromisoformat(row["started_at"]),
            ended_at=(datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None),
            observations=observations,
            metadata=json.loads(row["metadata"]),
            summary=row["summary"],
            total_tokens=row["total_tokens"],
        )

    async def get_observations(
        self,
        session_id: str,
        obs_type: ObservationType | None = None,
    ) -> list[Observation]:
        """
        Get observations for a session.

        Args:
            session_id: Session ID
            obs_type: Optional filter by observation type

        Returns:
            List of observations
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            if obs_type is not None:
                cursor.execute(
                    """
                    SELECT * FROM observations
                    WHERE session_id = ? AND type = ?
                    ORDER BY timestamp
                    """,
                    (session_id, obs_type.value),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM observations
                    WHERE session_id = ?
                    ORDER BY timestamp
                    """,
                    (session_id,),
                )

            rows = cursor.fetchall()

        return [
            Observation(
                id=row["id"],
                session_id=row["session_id"],
                type=ObservationType(row["type"]),
                content=row["content"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                token_count=row["token_count"],
                metadata=json.loads(row["metadata"]),
                relevance_score=row["relevance_score"],
                compressed=bool(row["compressed"]),
            )
            for row in rows
        ]

    async def get_relevant_context(
        self,
        query: str,
        max_tokens: int = 2000,
        session_limit: int = 10,
    ) -> SessionContext:
        """
        Get relevant context from past sessions.

        Uses keyword matching to find relevant observations.
        For production, consider integrating with RAG for semantic search.

        Args:
            query: Query to match against observations
            max_tokens: Maximum tokens to return
            session_limit: Max sessions to search

        Returns:
            SessionContext with relevant observations
        """
        # Get recent sessions
        sessions = await self.list_sessions(limit=session_limit)

        # Score and collect observations
        scored_observations: list[tuple[float, Observation]] = []

        for session in sessions:
            for obs in session.observations:
                score = self._keyword_match(query, obs.content)
                # Also check metadata
                if obs.metadata:
                    metadata_str = json.dumps(obs.metadata)
                    score = max(score, self._keyword_match(query, metadata_str))

                if score > 0:
                    # Boost insights and preferences
                    if obs.type in (ObservationType.INSIGHT, ObservationType.PREFERENCE):
                        score *= 1.5

                    scored_observations.append((score, obs))

        # Sort by score
        scored_observations.sort(key=lambda x: x[0], reverse=True)

        # Collect observations up to token limit
        selected: list[Observation] = []
        total_tokens = 0
        sessions_referenced: set[str] = set()

        for score, obs in scored_observations:
            if total_tokens + obs.token_count > max_tokens:
                continue
            obs.relevance_score = score
            selected.append(obs)
            total_tokens += obs.token_count
            sessions_referenced.add(obs.session_id)

        logger.info(
            "Retrieved relevant context",
            query=query[:50],
            observations_found=len(selected),
            total_tokens=total_tokens,
            sessions_referenced=len(sessions_referenced),
        )

        return SessionContext(
            observations=selected,
            total_tokens=total_tokens,
            sessions_referenced=len(sessions_referenced),
            retrieval_query=query,
        )

    async def _compress_observations(
        self,
        observations: list[Observation],
    ) -> Observation:
        """
        Compress multiple observations into one.

        Creates a summary observation from multiple related observations.

        Args:
            observations: Observations to compress

        Returns:
            Single compressed observation
        """
        if not observations:
            raise ValueError("No observations to compress")

        session_id = observations[0].session_id

        # Group by type
        by_type: dict[ObservationType, list[str]] = {}
        for obs in observations:
            if obs.type not in by_type:
                by_type[obs.type] = []
            by_type[obs.type].append(obs.content)

        # Create summary content
        summary_parts = []
        for obs_type, contents in by_type.items():
            summary_parts.append(f"[{obs_type.value.upper()}] ({len(contents)} items)")
            # Keep first and last items
            if len(contents) <= 3:
                for content in contents:
                    summary_parts.append(f"  - {content[:100]}")
            else:
                summary_parts.append(f"  - {contents[0][:100]}")
                summary_parts.append(f"  - ... ({len(contents) - 2} more)")
                summary_parts.append(f"  - {contents[-1][:100]}")

        summary = "\n".join(summary_parts)
        token_count = len(summary) // CHARS_PER_TOKEN

        compressed = Observation(
            id=str(uuid.uuid4()),
            session_id=session_id,
            type=ObservationType.CONTEXT,
            content=summary,
            timestamp=datetime.utcnow(),
            token_count=token_count,
            metadata={
                "compressed_from": len(observations),
                "original_types": list(by_type.keys()),
            },
            compressed=True,
        )

        return compressed

    async def _generate_session_summary(self, session: Session) -> str:
        """
        Generate summary of session.

        Creates a concise summary of what happened during the session.

        Args:
            session: Session to summarize

        Returns:
            Summary string
        """
        if not session.observations:
            return "Empty session"

        # Count by type
        type_counts: dict[str, int] = {}
        for obs in session.observations:
            type_name = obs.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        # Get insights and important observations
        insights = [obs for obs in session.observations if obs.type == ObservationType.INSIGHT]
        errors = [obs for obs in session.observations if obs.type == ObservationType.ERROR]

        # Build summary
        parts = []
        parts.append(f"Session duration: {session.duration_seconds:.1f}s")
        parts.append(f"Observations: {len(session.observations)}")
        parts.append(f"Types: {', '.join(f'{k}:{v}' for k, v in type_counts.items())}")

        if insights:
            parts.append(f"Key insights: {len(insights)}")
            for insight in insights[:3]:  # Max 3 insights in summary
                parts.append(f"  - {insight.content[:100]}")

        if errors:
            parts.append(f"Errors encountered: {len(errors)}")

        return "\n".join(parts)

    def _keyword_match(self, query: str, content: str) -> float:
        """
        Calculate keyword match score.

        Simple keyword matching for relevance scoring.

        Args:
            query: Search query
            content: Content to match against

        Returns:
            Score from 0.0 to 1.0
        """
        query_lower = query.lower()
        content_lower = content.lower()

        # Extract words
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())

        if not query_words:
            return 0.0

        # Calculate overlap
        matches = query_words & content_words
        score = len(matches) / len(query_words)

        # Boost for exact phrase match
        if query_lower in content_lower:
            score = min(1.0, score + 0.3)

        return score

    async def _check_compression(self, session_id: str) -> None:
        """
        Check if session needs compression and apply if needed.

        Args:
            session_id: Session to check
        """
        session = await self.get_session(session_id)
        if session is None:
            return

        # Check thresholds
        if len(session.observations) < self.max_observations_per_session:
            total_tokens = sum(obs.token_count for obs in session.observations)
            if total_tokens < self.compress_threshold_tokens:
                return

        # Get non-compressed observations
        uncompressed = [obs for obs in session.observations if not obs.compressed]
        if len(uncompressed) < 10:  # Minimum batch for compression
            return

        # Compress older observations (keep last 20)
        to_compress = uncompressed[:-20]
        if not to_compress:
            return

        compressed = await self._compress_observations(to_compress)

        # Store compressed observation
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Delete original observations
            ids_to_delete = [obs.id for obs in to_compress]
            cursor.execute(
                f"""
                DELETE FROM observations
                WHERE id IN ({','.join('?' * len(ids_to_delete))})
                """,
                ids_to_delete,
            )

            # Insert compressed
            cursor.execute(
                """
                INSERT INTO observations
                (id, session_id, type, content, timestamp, token_count,
                 metadata, relevance_score, compressed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    compressed.id,
                    session_id,
                    compressed.type.value,
                    compressed.content,
                    compressed.timestamp.isoformat(),
                    compressed.token_count,
                    json.dumps(compressed.metadata),
                    compressed.relevance_score,
                    1,
                ),
            )

            conn.commit()

        logger.info(
            "Compressed observations",
            session_id=session_id[:8],
            compressed_count=len(to_compress),
        )

    async def list_sessions(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Session]:
        """
        List recent sessions.

        Args:
            limit: Maximum sessions to return
            offset: Offset for pagination

        Returns:
            List of sessions (newest first)
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT id FROM sessions
                ORDER BY started_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            rows = cursor.fetchall()

        sessions = []
        for row in rows:
            session = await self.get_session(row["id"])
            if session:
                sessions.append(session)

        return sessions

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and its observations.

        Args:
            session_id: Session to delete

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Delete observations first (foreign key)
            cursor.execute(
                "DELETE FROM observations WHERE session_id = ?",
                (session_id,),
            )

            # Delete session
            cursor.execute(
                "DELETE FROM sessions WHERE id = ?",
                (session_id,),
            )

            deleted = cursor.rowcount > 0
            conn.commit()

        # Remove from cache
        self._active_sessions.pop(session_id, None)

        if deleted:
            logger.info("Session deleted", session_id=session_id[:8])

        return deleted

    async def search_observations(
        self,
        query: str,
        obs_type: ObservationType | None = None,
        limit: int = 50,
    ) -> list[Observation]:
        """
        Search observations across all sessions.

        Args:
            query: Search query
            obs_type: Optional filter by type
            limit: Maximum results

        Returns:
            List of matching observations
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Use LIKE for simple search
            search_pattern = f"%{query}%"

            if obs_type is not None:
                cursor.execute(
                    """
                    SELECT * FROM observations
                    WHERE content LIKE ? AND type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (search_pattern, obs_type.value, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM observations
                    WHERE content LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (search_pattern, limit),
                )

            rows = cursor.fetchall()

        results = [
            Observation(
                id=row["id"],
                session_id=row["session_id"],
                type=ObservationType(row["type"]),
                content=row["content"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                token_count=row["token_count"],
                metadata=json.loads(row["metadata"]),
                relevance_score=row["relevance_score"],
                compressed=bool(row["compressed"]),
            )
            for row in rows
        ]

        # Re-score based on keyword match
        for obs in results:
            obs.relevance_score = self._keyword_match(query, obs.content)

        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        return results

    def get_stats(self) -> dict[str, Any]:
        """
        Get session manager statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Total sessions
            cursor.execute("SELECT COUNT(*) FROM sessions")
            total_sessions = cursor.fetchone()[0]

            # Active sessions
            cursor.execute("SELECT COUNT(*) FROM sessions WHERE ended_at IS NULL")
            active_sessions = cursor.fetchone()[0]

            # Total observations
            cursor.execute("SELECT COUNT(*) FROM observations")
            total_observations = cursor.fetchone()[0]

            # Observations by type
            cursor.execute("SELECT type, COUNT(*) FROM observations GROUP BY type")
            by_type = {row[0]: row[1] for row in cursor.fetchall()}

            # Total tokens
            cursor.execute("SELECT SUM(token_count) FROM observations")
            total_tokens = cursor.fetchone()[0] or 0

            # Compressed observations
            cursor.execute("SELECT COUNT(*) FROM observations WHERE compressed = 1")
            compressed_count = cursor.fetchone()[0]

        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "total_observations": total_observations,
            "observations_by_type": by_type,
            "total_tokens": total_tokens,
            "compressed_observations": compressed_count,
            "db_path": str(self.db_path),
        }

    async def cleanup_old_sessions(
        self,
        older_than_days: int = 30,
    ) -> int:
        """
        Remove old sessions.

        Args:
            older_than_days: Delete sessions older than this many days

        Returns:
            Number of sessions deleted
        """
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        cutoff_str = cutoff.isoformat()

        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Get sessions to delete
            cursor.execute(
                """
                SELECT id FROM sessions
                WHERE started_at < ? AND ended_at IS NOT NULL
                """,
                (cutoff_str,),
            )
            session_ids = [row[0] for row in cursor.fetchall()]

            if not session_ids:
                return 0

            # Delete observations
            cursor.execute(
                f"""
                DELETE FROM observations
                WHERE session_id IN ({','.join('?' * len(session_ids))})
                """,
                session_ids,
            )

            # Delete sessions
            cursor.execute(
                f"""
                DELETE FROM sessions
                WHERE id IN ({','.join('?' * len(session_ids))})
                """,
                session_ids,
            )

            deleted_count = cursor.rowcount
            conn.commit()

        logger.info(
            "Cleaned up old sessions",
            deleted_count=deleted_count,
            older_than_days=older_than_days,
        )

        return deleted_count

    def close(self) -> None:
        """Close database connection."""
        with self._lock:
            if self._connection:
                self._connection.close()
                self._connection = None


# =============================================================================
# Convenience Functions
# =============================================================================


# Global default manager instance
_default_manager: SessionManager | None = None
_manager_lock = Lock()


async def quick_session(
    db_path: str = "~/.contextflow/sessions.db",
) -> SessionManager:
    """
    Create session manager with default settings.

    Args:
        db_path: Path to database

    Returns:
        Configured SessionManager
    """
    return SessionManager(db_path=db_path)


def get_default_session_manager() -> SessionManager:
    """
    Get or create default session manager.

    Returns:
        Default SessionManager instance
    """
    global _default_manager

    with _manager_lock:
        if _default_manager is None:
            _default_manager = SessionManager()
        return _default_manager


def reset_default_session_manager() -> None:
    """Reset the default session manager (useful for testing)."""
    global _default_manager

    with _manager_lock:
        if _default_manager is not None:
            _default_manager.close()
            _default_manager = None
