import pytest
from src.agents.state import SessionState, Message
from src.agents.checkpoint import InMemoryCheckpointManager

pytestmark = pytest.mark.asyncio

@pytest.fixture
def checkpoint_manager() -> InMemoryCheckpointManager:
    """Returns a fresh InMemoryCheckpointManager instance for each test."""
    return InMemoryCheckpointManager()

@pytest.fixture
def sample_session_state() -> SessionState:
    """Returns a sample SessionState object."""
    return SessionState(
        conversation_history=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!")
        ],
        thread_id="thread_123",
        original_request_info={"language": "python"}
    )

# --- Test Cases ---

async def test_save_and_load_state(checkpoint_manager: InMemoryCheckpointManager, sample_session_state: SessionState):
    thread_id = "test_thread_1"

    # Save state
    await checkpoint_manager.save_state(thread_id, sample_session_state)

    # Load state
    loaded_state = await checkpoint_manager.load_state(thread_id)

    assert loaded_state is not None
    assert loaded_state.thread_id == sample_session_state.thread_id
    assert len(loaded_state.conversation_history) == len(sample_session_state.conversation_history)
    assert loaded_state.conversation_history[0].content == "Hello"
    assert loaded_state.original_request_info == {"language": "python"}

async def test_load_non_existent_state(checkpoint_manager: InMemoryCheckpointManager):
    loaded_state = await checkpoint_manager.load_state("non_existent_thread")
    assert loaded_state is None

async def test_save_state_is_deep_copy(checkpoint_manager: InMemoryCheckpointManager, sample_session_state: SessionState):
    thread_id = "test_deep_copy"

    original_content = sample_session_state.conversation_history[0].content

    await checkpoint_manager.save_state(thread_id, sample_session_state)

    # Modify the original state object AFTER saving
    sample_session_state.conversation_history[0].content = "Modified content"
    sample_session_state.add_message(role="user", content="Another message")

    loaded_state = await checkpoint_manager.load_state(thread_id)
    assert loaded_state is not None
    # Check that loaded state reflects the state AT THE TIME OF SAVING, not the modified original
    assert loaded_state.conversation_history[0].content == original_content
    assert len(loaded_state.conversation_history) == 2 # Not 3

async def test_load_state_is_deep_copy(checkpoint_manager: InMemoryCheckpointManager, sample_session_state: SessionState):
    thread_id = "test_load_deep_copy"
    await checkpoint_manager.save_state(thread_id, sample_session_state)

    loaded_state_1 = await checkpoint_manager.load_state(thread_id)
    assert loaded_state_1 is not None

    # Modify the loaded state
    loaded_state_1.add_message(role="user", content="New message in loaded_state_1")
    loaded_state_1.conversation_history[0].content = "Changed in loaded_state_1"

    # Load the state again
    loaded_state_2 = await checkpoint_manager.load_state(thread_id)
    assert loaded_state_2 is not None

    # Ensure loaded_state_2 is not affected by modifications to loaded_state_1
    assert len(loaded_state_2.conversation_history) == 2
    assert loaded_state_2.conversation_history[0].content == "Hello" # Original content

async def test_delete_state(checkpoint_manager: InMemoryCheckpointManager, sample_session_state: SessionState):
    thread_id = "test_delete_thread"

    # Save then delete
    await checkpoint_manager.save_state(thread_id, sample_session_state)
    deleted = await checkpoint_manager.delete_state(thread_id)
    assert deleted is True

    # Try to load deleted state
    loaded_state = await checkpoint_manager.load_state(thread_id)
    assert loaded_state is None

async def test_delete_non_existent_state(checkpoint_manager: InMemoryCheckpointManager):
    deleted = await checkpoint_manager.delete_state("non_existent_for_delete")
    assert deleted is False

async def test_save_state_type_error(checkpoint_manager: InMemoryCheckpointManager):
    with pytest.raises(TypeError, match="state must be an instance of SessionState"):
        await checkpoint_manager.save_state("thread_type_error", "not_a_session_state_object") # type: ignore
```
