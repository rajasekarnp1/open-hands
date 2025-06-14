"""
Base definitions for tools that AI agents can use.
Includes ToolParameter, ToolDefinition models and the @openhands_tool decorator.
"""
from __future__ import annotations # For Python 3.7, 3.8 compatibility with Pydantic v2

import inspect
from typing import Callable, List, Optional, Dict, Any, Type
from pydantic import BaseModel, create_model, Field
import docstring_parser

class ToolParameter(BaseModel):
    """Describes a parameter for a tool."""
    name: str
    type: str  # JSON schema type (e.g., "string", "integer", "boolean", "object", "array")
    description: str
    required: bool
    properties: Optional[Dict[str, ToolParameter]] = None # For nested object types
    items: Optional[ToolParameter] = None # For array types if elements are uniform

class ToolDefinition(BaseModel):
    """Describes a tool available to an agent."""
    name: str
    description: str
    parameters: List[ToolParameter] # Parameters the LLM should provide
    function: Callable # The actual function to call, not exposed to LLM directly
    # Internal parameters are those injected by the agent runtime, not specified by LLM
    internal_parameters: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True # To allow Callable

# Placeholder for the decorator and registry logic which will be added next.
# The decorator will populate ToolDefinition objects.
# A global registry instance will likely live in a separate 'registry.py'.

# Example of how Pydantic model schemas can be extracted
def get_pydantic_model_schema(pydantic_model: Type[BaseModel]) -> List[ToolParameter]:
    """
    Converts a Pydantic model's JSON schema into a list of ToolParameter.
    This is a simplified version. Real schema conversion can be more complex.
    """
    if not hasattr(pydantic_model, "model_json_schema"):
        return [] # Not a Pydantic model or old Pydantic version

    schema = pydantic_model.model_json_schema()
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    tool_params = []
    for name, prop_schema in properties.items():
        param_type = prop_schema.get("type", "any")
        if "anyOf" in prop_schema: # Handle Optional fields or Unions
            # Simplification: take the first non-null type or just represent as 'any'
            # A more robust solution would detail the union or optional nature.
            non_null_type = next((t.get("type", "any") for t in prop_schema["anyOf"] if t.get("type") != "null"), "any")
            param_type = non_null_type

        description = prop_schema.get("description", "")
        # If description is not in schema, try to get from title (often field name for simple fields)
        if not description and "title" in prop_schema:
            description = f"Parameter: {prop_schema['title']}"

        param = ToolParameter(
            name=name,
            type=param_type,
            description=description,
            required=name in required_fields
        )
        # Basic handling for nested objects - this would need recursion for full support
        if param_type == "object" and "properties" in prop_schema:
            # This is a simplification; true nested schema parsing is more involved
            # For now, we'll just indicate it's an object.
            # A full implementation would recursively call something like this function.
             param.description += " (Object with nested fields)"
        tool_params.append(param)

    return tool_params

# Placeholder for the decorator itself
def openhands_tool(func: Callable) -> Callable:
    # Actual decorator logic will be implemented in the next step.
    # It will create a ToolDefinition and register it.
    # For now, just return the function itself.
from .registry import global_tool_registry # Import the actual global registry

def _parse_type_hint(hint: Type[Any]) -> str:
    """Converts Python type hints to JSON schema type strings."""
    origin = getattr(hint, "__origin__", None)
    args = getattr(hint, "__args__", [])

    if hint is str:
        return "string"
    if hint is int:
        return "integer"
    if hint is float:
        return "number"
    if hint is bool:
        return "boolean"
    if hint is list or origin is list:
        if args and args[0] is not Any:
            # For List[type], describe as array of that type.
            # Simplified: does not create full 'items' schema here yet.
            return f"array (items type: {_parse_type_hint(args[0])})"
        return "array"
    if hint is dict or origin is dict:
        return "object"
    if issubclass(hint, BaseModel):
        # For Pydantic models, we'd ideally return "object" and a nested schema.
        # For now, just identify as object. get_pydantic_model_schema can be used later.
        return "object" # In future, could add schema details here
    if hint is Any or hint is inspect.Parameter.empty:
        return "any"

    # Handle Optional[Type] by extracting the inner type
    if origin is Optional or (origin is Union and type(None) in args):
        if args and args[0] is not type(None): # Make sure there's another type apart from None
            return _parse_type_hint(args[0]) # Get the type of T in Optional[T]
        else: # pragma: no cover (e.g. Optional[NoneType] which is just NoneType)
            return "null"

    return str(hint) # Fallback, might not be a valid JSON schema type


def openhands_tool(func: Callable) -> Callable:
    """
    Decorator to register a function as an OpenHands tool.
    It inspects the function's signature and docstring to create a ToolDefinition.
    """
    parsed_docstring = docstring_parser.parse(func.__doc__ or "")
    tool_description = parsed_docstring.short_description or func.__name__

    sig = inspect.signature(func)
    tool_parameters: List[ToolParameter] = []
    internal_params: List[str] = []

    # Identify internal parameters (e.g., project_directory, tool_call_id for specific tools)
    # These are typically injected by the agent runtime.
    INTERNAL_PARAM_NAMES = ["project_directory", "tool_call_id"]

    for name, param in sig.parameters.items():
        if name in INTERNAL_PARAM_NAMES:
            internal_params.append(name)
            continue # Skip adding internal params to LLM-facing schema

        param_doc = next((p for p in parsed_docstring.params if p.arg_name == name), None)
        param_description = param_doc.description if param_doc else "No description."

        param_type_hint = param.annotation
        param_type_str = _parse_type_hint(param_type_hint)

        # Check if parameter is required (no default value)
        is_required = (param.default == inspect.Parameter.empty)

        # Basic handling for Pydantic models as parameters
        properties_schema = None
        items_schema = None
        if inspect.isclass(param_type_hint) and issubclass(param_type_hint, BaseModel):
            # For Pydantic models, generate a simplified nested schema
            # This is a placeholder for a more robust schema generation
            # properties_schema = get_pydantic_model_schema(param_type_hint) # TODO: Refine this
            param_type_str = "object" # Indicate it's an object, details can be in description
            param_description += f" (Pydantic Model: {param_type_hint.__name__})"


        tool_parameters.append(ToolParameter(
            name=name,
            type=param_type_str,
            description=param_description,
            required=is_required,
            properties=properties_schema, # For nested objects
            items=items_schema         # For arrays of objects
        ))

    tool_def = ToolDefinition(
        name=func.__name__,
        description=tool_description,
        parameters=tool_parameters,
        function=func,
        internal_parameters=internal_params
    )

    # Register with the global tool registry
    global_tool_registry.register(tool_def)

    # Store the definition on the function itself for potential direct access/inspection
    func._tool_definition = tool_def # type: ignore

    return func
