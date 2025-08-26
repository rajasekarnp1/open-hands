import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch

from src.api.server import app

# Pytest marker for async tests
pytestmark = pytest.mark.asyncio

@pytest.fixture
def client() -> TestClient:
    """Returns a TestClient instance for the FastAPI app."""
    # In a real app, you might have a more complex setup,
    # but for basic endpoint testing, this is sufficient.
    # The lifespan context manager in the app will run,
    # so we need to ensure its dependencies are available or mocked.

    # For now, we assume the app can be initialized for testing.
    # If the lifespan function has heavy dependencies, they should be mocked here.
    with patch('src.api.server.aggregator_instance', new_callable=AsyncMock) as mock_aggregator:
        # You can configure the mock aggregator here if needed for specific tests
        # e.g., mock_aggregator.chat_completion.return_value = ...
        yield TestClient(app)

# A simple test to ensure the root endpoint is working.
# More tests for chat completions, etc., can be added here.
def test_root_endpoint(client: TestClient):
    """Tests the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "LLM API Aggregator",
        "version": "1.0.0",
        "status": "running"
    }

# Add more tests for the remaining, non-agent endpoints as needed.
# For example, a basic test for /v1/chat/completions
# This would require mocking the aggregator_instance.
# The patch in the client fixture helps with this.

# async def test_chat_completions_endpoint(client: TestClient):
#     # The mock for aggregator_instance is active from the client fixture
#     # We would need to configure the return value on the mock
#     # This is just a placeholder example.
#     pass
