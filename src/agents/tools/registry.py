"""
Tool Registry for managing and accessing OpenHands tools.
"""
from __future__ import annotations
import json
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import ToolDefinition # Import only for type checking to avoid circular dependency at runtime if base imports registry

class ToolRegistry:
    """
    Manages the registration and retrieval of tools available to agents.
    """
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool_def: ToolDefinition):
        """
        Registers a tool definition.

        Args:
            tool_def: The ToolDefinition object to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool_def.name in self._tools:
            raise ValueError(f"Tool with name '{tool_def.name}' is already registered.")
        self._tools[tool_def.name] = tool_def
        print(f"Tool '{tool_def.name}' registered.") # Simple print for now, use logger in real app

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """
        Retrieves a tool definition by its name.

        Args:
            name: The name of the tool to retrieve.

        Returns:
            The ToolDefinition if found, else None.
        """
        return self._tools.get(name)

    def get_all_tools(self) -> List[ToolDefinition]:
        """
        Returns a list of all registered tool definitions.
        """
        return list(self._tools.values())

    def generate_llm_tool_descriptions(self, as_json_string: bool = False) -> str | List[Dict[str, Any]]:
        """
        Generates a description of all registered tools formatted for inclusion in an LLM prompt.
        This description informs the LLM about what tools it can call.

        Args:
            as_json_string: If True, returns a JSON string. Otherwise, returns a list of dicts.

        Returns:
            A string (plain text or JSON) or a list of dictionaries describing the tools.
        """
        tool_schemas = []
        for tool_def in self._tools.values():
            parameters_schema = {
                "type": "object",
                "properties": {},
                "required": []
            }
            for param in tool_def.parameters:
                param_schema = {
                    "type": param.type,
                    "description": param.description
                }
                # Handle nested properties for object types if defined (basic support)
                if param.type == "object" and param.properties:
                    # This part would need to recursively build the schema for nested objects
                    # For now, we'll just pass the simplified 'properties' if available
                    nested_props = {}
                    for p_name, p_def in param.properties.items():
                        nested_props[p_name] = {"type": p_def.type, "description": p_def.description}
                    param_schema["properties"] = nested_props

                # Handle items for array types if defined (basic support)
                if param.type.startswith("array") and param.items:
                     param_schema["items"] = {"type": param.items.type, "description": param.items.description}


                parameters_schema["properties"][param.name] = param_schema
                if param.required:
                    parameters_schema["required"].append(param.name)

            if not parameters_schema["properties"]: # No parameters for the tool
                 parameters_schema = {"type": "object", "properties": {}}


            tool_schema_entry = {
                "name": tool_def.name,
                "description": tool_def.description,
                "parameters": parameters_schema
            }
            if tool_def.usage_notes:
                tool_schema_entry["usage_notes"] = tool_def.usage_notes

            tool_schemas.append(tool_schema_entry)

        if as_json_string:
            try:
                return json.dumps(tool_schemas, indent=2)
            except TypeError as e:
                # Fallback for complex types not serializable by default json.dumps
                # (e.g. if a 'type' field contained a Python type object by mistake)
                print(f"Error serializing tool schemas to JSON: {e}. Returning non-indented string.")
                return str(tool_schemas) # Basic string representation as fallback
        else:
            return tool_schemas

# Create a global instance of the ToolRegistry for easy access
global_tool_registry = ToolRegistry()

# Now, modify base.py to use this global_tool_registry
# This part is tricky as it creates a potential circular import if base.py imports registry.py
# for global_tool_registry and registry.py imports base.py for ToolDefinition.
# This is often solved by:
# 1. Defining global_tool_registry in base.py itself (simplest for now).
# 2. Or, the decorator in base.py appends to a list, and registry explicitly pulls from that list.
# 3. Or, using runtime imports or a setter method on the registry.

# For this step, I will assume that base.py's decorator will register to this instance.
# I will modify base.py in the next step to correctly use this.
```
