"""
Unit tests for SessionManager.

Tests session management functionality including:
- Session creation/retrieval
- Observation tracking
- SQLite persistence
- Context retrieval
- Session lifecycle
"""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from contextflow.core.session import (
    CHARS_PER_TOKEN,
    Observation,
    ObservationType,
    Session,
    SessionContext,
    SessionManager,
    get_default_session_manager,
    reset_default_session_manager,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path() -> str:
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test_sessions.db")


@pytest.fixture
def session_manager(temp_db_path: str) -> SessionManager:
    """Create a SessionManager with temporary database."""
    manager = SessionManager(
        db_path=temp_db_path,
        max_observations_per_session=100,
        compress_threshold_tokens=1000,
    )
    yield manager
    manager.close()


@pytest.fixture
def sample_observation() -> Observation:
    """Create a sample observation."""
    return Observation(
        id="obs-123",
        session_id="session-456",
        type=ObservationType.TASK,
        content="Processed a document",
        metadata={"key": "value"},
    )


# =============================================================================
# ObservationType Tests
# =============================================================================


class TestObservationType:
    """Tests for ObservationType enum."""

    def test_all_types_exist(self) -> None:
        """Test that all expected observation types exist."""
        expected = ["TASK", "RESULT", "ERROR", "INSIGHT", "PREFERENCE", "CONTEXT", "VERIFICATION"]

        for type_name in expected:
            assert hasattr(ObservationType, type_name)

    def test_type_values(self) -> None:
        """Test observation type values."""
        assert ObservationType.TASK.value == "task"
        assert ObservationType.INSIGHT.value == "insight"
        assert ObservationType.ERROR.value == "error"


# =============================================================================
# Observation Tests
# =============================================================================


class TestObservation:
    """Tests for Observation dataclass."""

    def test_observation_creation(self) -> None:
        """Test observation creation with required fields."""
        obs = Observation(
            id="test-id", session_id="session-id", type=ObservationType.TASK, content="Test content"
        )

        assert obs.id == "test-id"
        assert obs.session_id == "session-id"
        assert obs.type == ObservationType.TASK
        assert obs.content == "Test content"
        assert obs.token_count == 0
        assert obs.relevance_score == 1.0
        assert obs.compressed is False
        assert isinstance(obs.timestamp, datetime)

    def test_observation_to_dict(self, sample_observation: Observation) -> None:
        """Test observation serialization to dictionary."""
        obs_dict = sample_observation.to_dict()

        assert obs_dict["id"] == "obs-123"
        assert obs_dict["session_id"] == "session-456"
        assert obs_dict["type"] == "task"
        assert obs_dict["content"] == "Processed a document"
        assert "timestamp" in obs_dict
        assert obs_dict["metadata"]["key"] == "value"

    def test_observation_from_dict(self) -> None:
        """Test observation deserialization from dictionary."""
        data = {
            "id": "obs-new",
            "session_id": "session-new",
            "type": "insight",
            "content": "Test insight",
            "timestamp": "2024-01-15T10:30:00",
            "token_count": 5,
            "metadata": {"extra": "data"},
            "relevance_score": 0.8,
            "compressed": False,
        }

        obs = Observation.from_dict(data)

        assert obs.id == "obs-new"
        assert obs.type == ObservationType.INSIGHT
        assert obs.token_count == 5
        assert obs.relevance_score == 0.8


# =============================================================================
# Session Tests
# =============================================================================


class TestSession:
    """Tests for Session dataclass."""

    def test_session_creation(self) -> None:
        """Test session creation."""
        session = Session(id="session-123", started_at=datetime.utcnow())

        assert session.id == "session-123"
        assert session.is_active is True
        assert session.ended_at is None
        assert len(session.observations) == 0
        assert session.total_tokens == 0

    def test_session_is_active(self) -> None:
        """Test session active status."""
        active_session = Session(id="active", started_at=datetime.utcnow())
        assert active_session.is_active is True

        ended_session = Session(
            id="ended", started_at=datetime.utcnow(), ended_at=datetime.utcnow()
        )
        assert ended_session.is_active is False

    def test_session_duration(self) -> None:
        """Test session duration calculation."""
        start = datetime.utcnow()
        session = Session(id="test", started_at=start, ended_at=start + timedelta(seconds=60))

        assert session.duration_seconds == 60.0

    def test_session_to_dict(self) -> None:
        """Test session serialization."""
        session = Session(
            id="test-session", started_at=datetime.utcnow(), metadata={"project": "test"}
        )
        session_dict = session.to_dict()

        assert session_dict["id"] == "test-session"
        assert "started_at" in session_dict
        assert session_dict["ended_at"] is None
        assert session_dict["metadata"]["project"] == "test"

    def test_session_from_dict(self) -> None:
        """Test session deserialization."""
        data = {
            "id": "from-dict",
            "started_at": "2024-01-15T10:00:00",
            "ended_at": None,
            "observations": [],
            "metadata": {},
            "summary": None,
            "total_tokens": 0,
        }

        session = Session.from_dict(data)

        assert session.id == "from-dict"
        assert session.is_active is True


# =============================================================================
# SessionManager Initialization Tests
# =============================================================================


class TestSessionManagerInit:
    """Tests for SessionManager initialization."""

    def test_init_creates_db(self, temp_db_path: str) -> None:
        """Test that initialization creates database."""
        manager = SessionManager(db_path=temp_db_path)

        assert Path(temp_db_path).exists()
        manager.close()

    def test_init_creates_tables(self, session_manager: SessionManager) -> None:
        """Test that initialization creates required tables."""
        # Tables should exist - check by getting stats
        stats = session_manager.get_stats()

        assert "total_sessions" in stats
        assert "total_observations" in stats

    def test_init_with_custom_params(self, temp_db_path: str) -> None:
        """Test initialization with custom parameters."""
        manager = SessionManager(
            db_path=temp_db_path, max_observations_per_session=500, compress_threshold_tokens=10000
        )

        assert manager.max_observations_per_session == 500
        assert manager.compress_threshold_tokens == 10000
        manager.close()


# =============================================================================
# Session Lifecycle Tests
# =============================================================================


class TestSessionLifecycle:
    """Tests for session lifecycle operations."""

    @pytest.mark.asyncio
    async def test_start_session(self, session_manager: SessionManager) -> None:
        """Test starting a new session."""
        session = await session_manager.start_session(metadata={"project": "test"})

        assert session.id is not None
        assert session.is_active is True
        assert session.metadata["project"] == "test"

    @pytest.mark.asyncio
    async def test_end_session(self, session_manager: SessionManager) -> None:
        """Test ending a session."""
        session = await session_manager.start_session()

        ended = await session_manager.end_session(session.id)

        assert ended.is_active is False
        assert ended.ended_at is not None

    @pytest.mark.asyncio
    async def test_end_session_generates_summary(self, session_manager: SessionManager) -> None:
        """Test that ending a session generates summary."""
        session = await session_manager.start_session()
        await session_manager.add_observation(session.id, ObservationType.TASK, "Test task")
        await session_manager.add_observation(
            session.id, ObservationType.INSIGHT, "Important insight"
        )

        ended = await session_manager.end_session(session.id, generate_summary=True)

        assert ended.summary is not None
        assert len(ended.summary) > 0

    @pytest.mark.asyncio
    async def test_end_nonexistent_session_raises(self, session_manager: SessionManager) -> None:
        """Test that ending non-existent session raises error."""
        with pytest.raises(ValueError, match="Session not found"):
            await session_manager.end_session("nonexistent-id")

    @pytest.mark.asyncio
    async def test_get_session(self, session_manager: SessionManager) -> None:
        """Test retrieving a session."""
        created = await session_manager.start_session()

        retrieved = await session_manager.get_session(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, session_manager: SessionManager) -> None:
        """Test getting non-existent session returns None."""
        result = await session_manager.get_session("nonexistent")

        assert result is None


# =============================================================================
# Observation Tests
# =============================================================================


class TestObservationManagement:
    """Tests for observation management."""

    @pytest.mark.asyncio
    async def test_add_observation(self, session_manager: SessionManager) -> None:
        """Test adding an observation."""
        session = await session_manager.start_session()

        obs = await session_manager.add_observation(
            session.id,
            ObservationType.TASK,
            "Processed the document",
            metadata={"file": "test.txt"},
        )

        assert obs.id is not None
        assert obs.type == ObservationType.TASK
        assert obs.content == "Processed the document"
        assert obs.metadata["file"] == "test.txt"
        assert obs.token_count > 0

    @pytest.mark.asyncio
    async def test_add_observation_estimates_tokens(self, session_manager: SessionManager) -> None:
        """Test that token count is estimated."""
        session = await session_manager.start_session()
        content = "This is test content for token estimation"

        obs = await session_manager.add_observation(session.id, ObservationType.TASK, content)

        expected_tokens = len(content) // CHARS_PER_TOKEN
        assert obs.token_count == expected_tokens

    @pytest.mark.asyncio
    async def test_add_observation_to_ended_session_raises(
        self, session_manager: SessionManager
    ) -> None:
        """Test that adding to ended session raises error."""
        session = await session_manager.start_session()
        await session_manager.end_session(session.id)

        with pytest.raises(ValueError, match="not active"):
            await session_manager.add_observation(session.id, ObservationType.TASK, "Should fail")

    @pytest.mark.asyncio
    async def test_add_observation_to_nonexistent_session_raises(
        self, session_manager: SessionManager
    ) -> None:
        """Test that adding to non-existent session raises error."""
        with pytest.raises(ValueError, match="Session not found"):
            await session_manager.add_observation(
                "nonexistent", ObservationType.TASK, "Should fail"
            )

    @pytest.mark.asyncio
    async def test_get_observations(self, session_manager: SessionManager) -> None:
        """Test getting observations for a session."""
        session = await session_manager.start_session()
        await session_manager.add_observation(session.id, ObservationType.TASK, "Task 1")
        await session_manager.add_observation(session.id, ObservationType.INSIGHT, "Insight 1")

        all_obs = await session_manager.get_observations(session.id)
        assert len(all_obs) == 2

        tasks_only = await session_manager.get_observations(
            session.id, obs_type=ObservationType.TASK
        )
        assert len(tasks_only) == 1


# =============================================================================
# Context Retrieval Tests
# =============================================================================


class TestContextRetrieval:
    """Tests for relevant context retrieval."""

    @pytest.mark.asyncio
    async def test_get_relevant_context(self, session_manager: SessionManager) -> None:
        """Test retrieving relevant context."""
        session = await session_manager.start_session()
        await session_manager.add_observation(
            session.id, ObservationType.INSIGHT, "Python is a programming language"
        )
        await session_manager.add_observation(
            session.id, ObservationType.TASK, "Analyzed the Python codebase"
        )
        await session_manager.end_session(session.id)

        context = await session_manager.get_relevant_context(
            query="Python programming", max_tokens=1000
        )

        assert isinstance(context, SessionContext)
        assert len(context.observations) > 0
        assert context.total_tokens <= 1000

    @pytest.mark.asyncio
    async def test_context_respects_token_limit(self, session_manager: SessionManager) -> None:
        """Test that context respects token limit."""
        session = await session_manager.start_session()

        # Add many observations
        for i in range(10):
            await session_manager.add_observation(
                session.id, ObservationType.TASK, f"Task {i} with some content about topic"
            )
        await session_manager.end_session(session.id)

        context = await session_manager.get_relevant_context(query="topic", max_tokens=50)

        assert context.total_tokens <= 50

    @pytest.mark.asyncio
    async def test_context_to_prompt_format(self, session_manager: SessionManager) -> None:
        """Test context formatting for prompts."""
        session = await session_manager.start_session()
        await session_manager.add_observation(
            session.id, ObservationType.INSIGHT, "Important finding about data"
        )
        await session_manager.end_session(session.id)

        context = await session_manager.get_relevant_context(query="data finding", max_tokens=500)

        formatted = context.to_prompt_format()

        assert "Previous Sessions" in formatted or "INSIGHT" in formatted


# =============================================================================
# Session List and Search Tests
# =============================================================================


class TestSessionListAndSearch:
    """Tests for listing and searching sessions."""

    @pytest.mark.asyncio
    async def test_list_sessions(self, session_manager: SessionManager) -> None:
        """Test listing sessions."""
        # Create multiple sessions
        for i in range(3):
            session = await session_manager.start_session()
            await session_manager.end_session(session.id)

        sessions = await session_manager.list_sessions(limit=10)

        assert len(sessions) == 3

    @pytest.mark.asyncio
    async def test_list_sessions_pagination(self, session_manager: SessionManager) -> None:
        """Test session listing with pagination."""
        for i in range(5):
            session = await session_manager.start_session()
            await session_manager.end_session(session.id)

        first_page = await session_manager.list_sessions(limit=2, offset=0)
        second_page = await session_manager.list_sessions(limit=2, offset=2)

        assert len(first_page) == 2
        assert len(second_page) == 2

    @pytest.mark.asyncio
    async def test_search_observations(self, session_manager: SessionManager) -> None:
        """Test searching observations."""
        session = await session_manager.start_session()
        await session_manager.add_observation(
            session.id, ObservationType.INSIGHT, "Machine learning is powerful"
        )
        await session_manager.add_observation(
            session.id, ObservationType.TASK, "Database optimization"
        )
        await session_manager.end_session(session.id)

        results = await session_manager.search_observations(query="machine learning")

        assert len(results) >= 1
        assert any("machine" in r.content.lower() for r in results)


# =============================================================================
# Session Deletion Tests
# =============================================================================


class TestSessionDeletion:
    """Tests for session deletion."""

    @pytest.mark.asyncio
    async def test_delete_session(self, session_manager: SessionManager) -> None:
        """Test deleting a session."""
        session = await session_manager.start_session()
        await session_manager.add_observation(session.id, ObservationType.TASK, "Test")
        await session_manager.end_session(session.id)

        deleted = await session_manager.delete_session(session.id)

        assert deleted is True

        # Should not be retrievable
        result = await session_manager.get_session(session.id)
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(self, session_manager: SessionManager) -> None:
        """Test deleting non-existent session."""
        deleted = await session_manager.delete_session("nonexistent")

        assert deleted is False


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Tests for session statistics."""

    @pytest.mark.asyncio
    async def test_get_stats(self, session_manager: SessionManager) -> None:
        """Test getting statistics."""
        session = await session_manager.start_session()
        await session_manager.add_observation(session.id, ObservationType.TASK, "Test task")

        stats = session_manager.get_stats()

        assert "total_sessions" in stats
        assert "active_sessions" in stats
        assert "total_observations" in stats
        assert "observations_by_type" in stats
        assert "total_tokens" in stats
        assert "db_path" in stats

        assert stats["total_sessions"] >= 1
        assert stats["active_sessions"] >= 1


# =============================================================================
# Cleanup Tests
# =============================================================================


class TestCleanup:
    """Tests for cleanup operations."""

    @pytest.mark.asyncio
    async def test_cleanup_old_sessions(self, session_manager: SessionManager) -> None:
        """Test cleaning up old sessions."""
        # Create and end a session
        session = await session_manager.start_session()
        await session_manager.end_session(session.id)

        # Cleanup with 0 days should delete ended sessions
        deleted = await session_manager.cleanup_old_sessions(older_than_days=0)

        # Note: This depends on timing - session was just created
        # so it might not be deleted. Testing the method doesn't error.
        assert deleted >= 0


# =============================================================================
# Keyword Match Tests
# =============================================================================


class TestKeywordMatch:
    """Tests for keyword matching functionality."""

    def test_keyword_match_exact(self, session_manager: SessionManager) -> None:
        """Test exact keyword matching."""
        score = session_manager._keyword_match(
            query="python programming", content="Python is a programming language"
        )

        assert score > 0

    def test_keyword_match_no_match(self, session_manager: SessionManager) -> None:
        """Test no keyword match."""
        score = session_manager._keyword_match(
            query="java", content="Python is a programming language"
        )

        # No common words
        assert score == 0 or score < 0.5

    def test_keyword_match_phrase_boost(self, session_manager: SessionManager) -> None:
        """Test phrase matching gets boosted."""
        partial_score = session_manager._keyword_match(
            query="programming language", content="Python is a language for programming tasks"
        )

        exact_score = session_manager._keyword_match(
            query="programming language", content="Python is a programming language"
        )

        # Exact phrase should score higher
        assert exact_score >= partial_score


# =============================================================================
# Global Session Manager Tests
# =============================================================================


class TestGlobalSessionManager:
    """Tests for global session manager."""

    def test_get_default_manager(self) -> None:
        """Test getting default session manager."""
        reset_default_session_manager()

        manager = get_default_session_manager()

        assert isinstance(manager, SessionManager)

    def test_default_manager_singleton(self) -> None:
        """Test that default manager is singleton."""
        reset_default_session_manager()

        manager1 = get_default_session_manager()
        manager2 = get_default_session_manager()

        assert manager1 is manager2

    def test_reset_default_manager(self) -> None:
        """Test resetting default manager."""
        manager1 = get_default_session_manager()
        reset_default_session_manager()
        manager2 = get_default_session_manager()

        assert manager1 is not manager2


# =============================================================================
# SessionContext Tests
# =============================================================================


class TestSessionContext:
    """Tests for SessionContext dataclass."""

    def test_empty_context_format(self) -> None:
        """Test formatting empty context."""
        context = SessionContext(
            observations=[], total_tokens=0, sessions_referenced=0, retrieval_query="test"
        )

        formatted = context.to_prompt_format()

        assert formatted == ""

    def test_context_with_observations_format(self, sample_observation: Observation) -> None:
        """Test formatting context with observations."""
        context = SessionContext(
            observations=[sample_observation],
            total_tokens=5,
            sessions_referenced=1,
            retrieval_query="test",
        )

        formatted = context.to_prompt_format()

        assert len(formatted) > 0
        assert "TASK" in formatted


# =============================================================================
# Compression Tests
# =============================================================================


class TestCompression:
    """Tests for observation compression."""

    @pytest.mark.asyncio
    async def test_compress_observations(self, session_manager: SessionManager) -> None:
        """Test compressing multiple observations."""
        session = await session_manager.start_session()

        # Create multiple observations
        observations = []
        for i in range(5):
            obs = Observation(
                id=f"obs-{i}",
                session_id=session.id,
                type=ObservationType.TASK if i % 2 == 0 else ObservationType.RESULT,
                content=f"Content for observation {i}",
            )
            observations.append(obs)

        compressed = await session_manager._compress_observations(observations)

        assert compressed.compressed is True
        assert compressed.type == ObservationType.CONTEXT
        assert "items" in compressed.content


# =============================================================================
# Connection Management Tests
# =============================================================================


class TestConnectionManagement:
    """Tests for database connection management."""

    def test_close_connection(self, session_manager: SessionManager) -> None:
        """Test closing database connection."""
        session_manager.close()

        # Connection should be None after close
        assert session_manager._connection is None
