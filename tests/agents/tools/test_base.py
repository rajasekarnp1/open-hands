import pytest
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from src.agents.tools.base import ToolParameter, ToolDefinition, openhands_tool, _parse_type_hint
from src.agents.tools.registry import global_tool_registry, ToolRegistry

# Sample Pydantic model for testing parameter type parsing
class SamplePydanticParam(BaseModel):
    sub_param_str: str = Field(..., description="A sub string parameter.")
    sub_param_int: Optional[int] = Field(None, description="An optional sub integer parameter.")

# --- Tests for _parse_type_hint ---

def test_parse_type_hint_basic():
    assert _parse_type_hint(str) == "string"
    assert _parse_type_hint(int) == "integer"
    assert _parse_type_hint(float) == "number"
    assert _parse_type_hint(bool) == "boolean"
    assert _parse_type_hint(list) == "array"
    assert _parse_type_hint(dict) == "object"
    assert _parse_type_hint(Any) == "any"

def test_parse_type_hint_complex():
    assert _parse_type_hint(Optional[str]) == "string" # Optional unwraps to base type
    assert _parse_type_hint(List[str]) == "array (items type: string)"
    assert _parse_type_hint(Dict[str, int]) == "object" # Dicts are objects, details would be in properties
    assert _parse_type_hint(Optional[List[int]]) == "array (items type: integer)"

def test_parse_type_hint_pydantic_model():
    assert _parse_type_hint(SamplePydanticParam) == "object"

# --- Tests for @openhands_tool decorator ---

# Clear the global registry before each test run in this module to ensure isolation
@pytest.fixture(autouse=True)
def clear_global_registry():
    original_tools = global_tool_registry._tools.copy()
    global_tool_registry._tools.clear()
    yield
    global_tool_registry._tools = original_tools # Restore

def test_openhands_tool_registration_and_definition():
    @openhands_tool
    def sample_tool_for_test(
        param_str: str,
        param_int: int = 10,
        param_bool: Optional[bool] = None,
        project_directory: Optional[str] = None, # Internal
        tool_call_id: Optional[str] = "id_abc" # Internal with default
    ):
        """
        A sample tool for testing purposes.
        This is its short description.

        Args:
            param_str (str): A required string parameter.
            param_int (int): An optional integer parameter with a default value.
            param_bool (Optional[bool]): An optional boolean parameter.
            project_directory (Optional[str]): (Internal) The project directory.
            tool_call_id (Optional[str]): (Internal) The tool call ID.
        """
        return f"{param_str}, {param_int}, {param_bool}"

    # Check registration
    tool_def = global_tool_registry.get_tool("sample_tool_for_test")
    assert tool_def is not None
    assert tool_def.name == "sample_tool_for_test"
    assert tool_def.description == "A sample tool for testing purposes."
    assert tool_def.function == sample_tool_for_test
    assert hasattr(sample_tool_for_test, '_tool_definition')
    assert sample_tool_for_test._tool_definition == tool_def # type: ignore

    # Check parameters (LLM-facing)
    assert len(tool_def.parameters) == 3

    param_str_def = next(p for p in tool_def.parameters if p.name == "param_str")
    assert param_str_def.type == "string"
    assert param_str_def.description == "A required string parameter."
    assert param_str_def.required is True

    param_int_def = next(p for p in tool_def.parameters if p.name == "param_int")
    assert param_int_def.type == "integer"
    assert param_int_def.description == "An optional integer parameter with a default value."
    assert param_int_def.required is False # Has default

    param_bool_def = next(p for p in tool_def.parameters if p.name == "param_bool")
    assert param_bool_def.type == "boolean" # Optional[bool] unwraps to bool
    assert param_bool_def.description == "An optional boolean parameter."
    assert param_bool_def.required is False # Is Optional

    # Check internal parameters
    assert len(tool_def.internal_parameters) == 2
    assert "project_directory" in tool_def.internal_parameters
    assert "tool_call_id" in tool_def.internal_parameters


def test_openhands_tool_no_docstring():
    @openhands_tool
    def tool_no_docstring(param1: str):
        pass # pragma: no cover

    tool_def = global_tool_registry.get_tool("tool_no_docstring")
    assert tool_def is not None
    assert tool_def.description == "tool_no_docstring" # Defaults to function name
    assert tool_def.parameters[0].description == "No description."

def test_openhands_tool_pydantic_param_description():
    @openhands_tool
    def tool_with_pydantic(data: SamplePydanticParam):
        """
        Tool that takes a Pydantic model as input.
        Args:
            data (SamplePydanticParam): The structured data input.
        """
        pass # pragma: no cover

    tool_def = global_tool_registry.get_tool("tool_with_pydantic")
    assert tool_def is not None
    assert len(tool_def.parameters) == 1
    param_data_def = tool_def.parameters[0]
    assert param_data_def.name == "data"
    assert param_data_def.type == "object"
    assert "Pydantic Model: SamplePydanticParam" in param_data_def.description
    assert param_data_def.required is True

# --- Tests for ToolParameter and ToolDefinition models ---

def test_tool_parameter_creation():
    param = ToolParameter(name="test_param", type="string", description="A test param.", required=True)
    assert param.name == "test_param"
    assert param.type == "string"

def test_tool_definition_creation():
    def dummy_func(): pass # pragma: no cover
    param = ToolParameter(name="p1", type="string", description="d1", required=True)
    tool_def = ToolDefinition(
        name="my_tool",
        description="My test tool.",
        parameters=[param],
        function=dummy_func,
        internal_parameters=["project_dir"]
    )
    assert tool_def.name == "my_tool"
    assert len(tool_def.parameters) == 1
    assert tool_def.function == dummy_func
    assert "project_dir" in tool_def.internal_parameters

```
