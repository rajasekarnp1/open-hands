"""
Secure Filesystem Tools for AI Agents.

Provides basic, sandboxed file operations like reading, writing, and listing files
within a specified project directory.
"""

import os
import asyncio
from pathlib import Path
import logging

import aiofiles # Will be added to requirements.txt

logger = logging.getLogger(__name__)

def _resolve_safe_path(project_directory: str, relative_path: str) -> Path:
    """
    Resolves a relative path against a project directory and ensures it's safe.

    Args:
        project_directory: The absolute path to the root of the project.
        relative_path: The relative path from the user/agent.

    Returns:
        A resolved, absolute Path object if safe.

    Raises:
        PermissionError: If the resolved path attempts to traverse outside
                         the project_directory (path traversal).
        ValueError: If project_directory is not an absolute path.
    """
    if not os.path.isabs(project_directory):
        # Ensure project_directory is absolute for security checks.
        # This should ideally be enforced by the caller or system configuration.
        logger.warning(f"Project directory '{project_directory}' was not absolute. Resolving now.")
        project_root = Path(project_directory).resolve()
    else:
        project_root = Path(project_directory)

    # Normalize relative_path: remove leading slashes if any to ensure proper joining
    # os.path.normpath can also help, but Path objects handle this well.
    # If relative_path starts with '/', it's treated as absolute by Path's / operator,
    # but resolve() later still correctly checks against project_root.
    # For clarity, ensure it's treated as relative.
    norm_relative_path = relative_path.lstrip('./\\') # Remove leading relative components
    if os.path.isabs(norm_relative_path): # If it became absolute (e.g. was "/foo")
        # This case should ideally be blocked or handled carefully.
        # Forcing it to be relative to project_root:
        norm_relative_path = norm_relative_path.lstrip(os.path.sep)


    # Join the project root with the potentially untrusted relative path
    unsafe_path = project_root / norm_relative_path

    # Resolve the path (evaluates '..' etc.)
    resolved_path = unsafe_path.resolve()

    # Security Check: Verify that the resolved path is still within the project_root.
    # Path.is_relative_to() is available in Python 3.9+
    # For robust checking: project_root.resolve() and resolved_path.resolve()
    if not resolved_path.is_relative_to(project_root.resolve()):
        logger.error(f"Path traversal attempt: '{relative_path}' resolved to '{resolved_path}' which is outside project root '{project_root.resolve()}'.")
        raise PermissionError(f"Path traversal detected and blocked for path: '{relative_path}'")

    return resolved_path

async def read_file(project_directory: str, filepath: str) -> str:
    """
    Reads the content of a file securely within the project directory.

    Args:
        project_directory: The absolute path to the root of the project.
        filepath: The relative path to the file.

    Returns:
        The content of the file as a string, or an error message.
    """
    try:
        safe_path = _resolve_safe_path(project_directory, filepath)

        # Check if it's a file (synchronous, run in thread)
        is_file = await asyncio.to_thread(safe_path.is_file)
        if not is_file:
            # To be more precise, check if it exists at all first.
            exists = await asyncio.to_thread(safe_path.exists)
            if not exists:
                 return f"Error: File not found at path '{filepath}'."
            return f"Error: Path '{filepath}' is not a file."

        async with aiofiles.open(safe_path, mode='r', encoding='utf-8') as f:
            content = await f.read()
        return content
    except PermissionError as e:
        logger.warning(f"Access denied for {filepath} in {project_directory}: {e}")
        return f"Error: Access denied. {e}"
    except FileNotFoundError: # Should be caught by is_file/exists, but as fallback
        logger.warning(f"File not found for {filepath} in {project_directory}")
        return f"Error: File not found at path '{filepath}'."
    except Exception as e:
        logger.error(f"Could not read file '{filepath}' in {project_directory}: {e}", exc_info=True)
        return f"Error: Could not read file '{filepath}'. {str(e)}"

async def write_file(project_directory: str, filepath: str, content: str) -> str:
    """
    Writes content to a file securely within the project directory.
    Creates parent directories if they don't exist.

    Args:
        project_directory: The absolute path to the root of the project.
        filepath: The relative path to the file.
        content: The content to write to the file.

    Returns:
        A success message string, or an error message string.
    """
    try:
        safe_path = _resolve_safe_path(project_directory, filepath)

        # Ensure parent directories exist (synchronous, run in thread)
        parent_dir = safe_path.parent
        await asyncio.to_thread(lambda: parent_dir.mkdir(parents=True, exist_ok=True))

        async with aiofiles.open(safe_path, mode='w', encoding='utf-8') as f:
            await f.write(content)
        logger.info(f"File '{filepath}' written successfully in '{project_directory}'.")
        return f"File '{filepath}' written successfully."
    except PermissionError as e:
        logger.warning(f"Access denied for writing {filepath} in {project_directory}: {e}")
        return f"Error: Access denied. {e}"
    except Exception as e:
        logger.error(f"Could not write file '{filepath}' in {project_directory}: {e}", exc_info=True)
        return f"Error: Could not write file '{filepath}'. {str(e)}"

async def list_files(project_directory: str, path: str = "") -> list[str] | str:
    """
    Lists files and directories securely within a specified path in the project directory.
    Appends a '/' to directory names in the returned list.

    Args:
        project_directory: The absolute path to the root of the project.
        path: The relative path to the directory to list. Defaults to project root.

    Returns:
        A list of file/directory names, or an error message string.
    """
    try:
        safe_target_path = _resolve_safe_path(project_directory, path)

        # Check if the resolved path is a directory (synchronous, run in thread)
        is_dir = await asyncio.to_thread(safe_target_path.is_dir)
        if not is_dir:
            exists = await asyncio.to_thread(safe_target_path.exists)
            if not exists:
                return f"Error: Directory not found at path '{path}'."
            return f"Error: Path '{path}' is not a directory."

        # List directory contents (synchronous, run in thread)
        dir_contents = await asyncio.to_thread(lambda: os.listdir(safe_target_path))

        results = []
        for item_name in sorted(dir_contents): # Sort for consistent output
            item_path = safe_target_path / item_name
            # Check if item is a directory (synchronous, run in thread)
            is_item_dir = await asyncio.to_thread(item_path.is_dir)
            if is_item_dir:
                results.append(f"{item_name}/")
            else:
                results.append(item_name)
        return results
    except PermissionError as e:
        logger.warning(f"Access denied for listing {path} in {project_directory}: {e}")
        return f"Error: Access denied. {e}"
    except FileNotFoundError: # Should be caught by is_dir/exists, but as fallback
        logger.warning(f"Directory not found for listing {path} in {project_directory}")
        return f"Error: Directory not found at path '{path}'."
    except Exception as e:
        logger.error(f"Could not list files in '{path}' in {project_directory}: {e}", exc_info=True)
        return f"Error: Could not list files in '{path}'. {str(e)}"
