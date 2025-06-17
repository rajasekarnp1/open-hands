import pytest
import json
from typing import Callable

from src.agents.tools.registry import ToolRegistry
from src.agents.tools.base import ToolDefinition, ToolParameter

# Sample tool functions for testing
def tool_func_1(param1: str, param2: int = 5) -> str:
    """Tool 1 description.
    Args:
        param1 (str): Description for param1.
        param2 (int): Description for param2.
    """
    return f"{param1}-{param2}"

def tool_func_2(project_directory: str, flag: bool) -> bool:
    """Tool 2 description.
    Args:
        project_directory (str): Internal project path.
        flag (bool): A boolean flag.
    """
    return not flag


@pytest.fixture
def sample_tool_def_1() -> ToolDefinition:
    return ToolDefinition(
        name="tool_func_1",
        description="Tool 1 description.",
        parameters=[
            ToolParameter(name="param1", type="string", description="Description for param1.", required=True),
            ToolParameter(name="param2", type="integer", description="Description for param2.", required=False),
        ],
        function=tool_func_1,
        internal_parameters=[]
    )

@pytest.fixture
def sample_tool_def_2() -> ToolDefinition:
    return ToolDefinition(
        name="tool_func_2",
        description="Tool 2 description.",
        parameters=[
            ToolParameter(name="flag", type="boolean", description="A boolean flag.", required=True),
        ],
        function=tool_func_2,
        internal_parameters=["project_directory"]
    )

@pytest.fixture
def registry() -> ToolRegistry:
    """Returns a fresh ToolRegistry instance for each test."""
    return ToolRegistry()

# --- Test Cases ---

def test_registry_register_tool(registry: ToolRegistry, sample_tool_def_1: ToolDefinition):
    registry.register(sample_tool_def_1)
    assert registry.get_tool("tool_func_1") == sample_tool_def_1

def test_registry_register_tool_conflict(registry: ToolRegistry, sample_tool_def_1: ToolDefinition):
    registry.register(sample_tool_def_1)
    with pytest.raises(ValueError, match="Tool with name 'tool_func_1' is already registered."):
        registry.register(sample_tool_def_1) # Registering the same tool again

def test_registry_get_tool_not_found(registry: ToolRegistry):
    assert registry.get_tool("non_existent_tool") is None

def test_registry_get_all_tools(registry: ToolRegistry, sample_tool_def_1: ToolDefinition, sample_tool_def_2: ToolDefinition):
    assert registry.get_all_tools() == []
    registry.register(sample_tool_def_1)
    registry.register(sample_tool_def_2)
    all_tools = registry.get_all_tools()
    assert len(all_tools) == 2
    assert sample_tool_def_1 in all_tools
    assert sample_tool_def_2 in all_tools

def test_registry_generate_llm_tool_descriptions_empty(registry: ToolRegistry):
    assert registry.generate_llm_tool_descriptions() == []
    assert registry.generate_llm_tool_descriptions(as_json_string=True) == "[]"

def test_registry_generate_llm_tool_descriptions_populated(registry: ToolRegistry, sample_tool_def_1: ToolDefinition, sample_tool_def_2: ToolDefinition):
    registry.register(sample_tool_def_1)
    registry.register(sample_tool_def_2)

    # Test as list of dicts
    descriptions_list = registry.generate_llm_tool_descriptions(as_json_string=False)
    assert isinstance(descriptions_list, list)
    assert len(descriptions_list) == 2

    desc_tool1 = next(d for d in descriptions_list if d["name"] == "tool_func_1")
    assert desc_tool1["description"] == "Tool 1 description."
    assert desc_tool1["parameters"]["type"] == "object"
    assert "param1" in desc_tool1["parameters"]["properties"]
    assert desc_tool1["parameters"]["properties"]["param1"]["type"] == "string"
    assert desc_tool1["parameters"]["properties"]["param1"]["description"] == "Description for param1."
    assert "param2" in desc_tool1["parameters"]["properties"]
    assert desc_tool1["parameters"]["properties"]["param2"]["type"] == "integer"
    assert "param1" in desc_tool1["parameters"]["required"]
    assert "param2" not in desc_tool1["parameters"]["required"] # param2 has a default

    desc_tool2 = next(d for d in descriptions_list if d["name"] == "tool_func_2")
    assert desc_tool2["description"] == "Tool 2 description."
    assert "flag" in desc_tool2["parameters"]["properties"]
    assert desc_tool2["parameters"]["properties"]["flag"]["type"] == "boolean"
    assert "flag" in desc_tool2["parameters"]["required"]
    # project_directory should not be in parameters for LLM
    assert "project_directory" not in desc_tool2["parameters"]["properties"]
    assert "usage_notes" not in desc_tool2 # sample_tool_def_2 has no usage_notes


    # Test as JSON string
    descriptions_json_str = registry.generate_llm_tool_descriptions(as_json_string=True)
    assert isinstance(descriptions_json_str, str)
    descriptions_loaded_from_json = json.loads(descriptions_json_str)
    assert len(descriptions_loaded_from_json) == 2

    # Verify one of the tools from JSON string for good measure
    json_desc_tool1 = next(d for d in descriptions_loaded_from_json if d["name"] == "tool_func_1")
    assert json_desc_tool1["parameters"]["properties"]["param1"]["type"] == "string"


def test_registry_tool_with_usage_notes_in_description(registry: ToolRegistry, sample_tool_def_1: ToolDefinition):
    sample_tool_def_1.usage_notes = "Important usage note here."
    registry.register(sample_tool_def_1)

    descriptions_list = registry.generate_llm_tool_descriptions(as_json_string=False)
    assert len(descriptions_list) == 1
    desc_tool1 = descriptions_list[0]
    assert desc_tool1["name"] == sample_tool_def_1.name
    assert "usage_notes" in desc_tool1
    assert desc_tool1["usage_notes"] == "Important usage note here."

    descriptions_json_str = registry.generate_llm_tool_descriptions(as_json_string=True)
    descriptions_loaded_from_json = json.loads(descriptions_json_str)
    json_desc_tool1 = descriptions_loaded_from_json[0]
    assert "usage_notes" in json_desc_tool1
    assert json_desc_tool1["usage_notes"] == "Important usage note here."


def test_registry_tool_with_no_llm_params(registry: ToolRegistry):
    def no_param_tool_func(project_directory: str): # Only internal param
        """A tool with no LLM-facing parameters."""
        pass # pragma: no cover

    no_param_tool_def = ToolDefinition(
        name="no_param_tool",
        description="A tool with no LLM-facing parameters.",
        parameters=[], # Empty list for LLM
        function=no_param_tool_func,
        internal_parameters=["project_directory"],
        usage_notes="This tool runs automatically based on context."
    )
    registry.register(no_param_tool_def)

    descriptions_list = registry.generate_llm_tool_descriptions(as_json_string=False)
    assert len(descriptions_list) == 1
    desc_no_param_tool = descriptions_list[0]
    assert desc_no_param_tool["name"] == "no_param_tool"
    assert desc_no_param_tool["parameters"]["type"] == "object"
    assert desc_no_param_tool["parameters"]["properties"] == {} # No properties for LLM
    assert not desc_no_param_tool["parameters"].get("required", []) # No required params
    assert desc_no_param_tool["usage_notes"] == "This tool runs automatically based on context."

```
