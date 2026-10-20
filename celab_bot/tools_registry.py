import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from celab_bot.users import get_user_home


class ToolsRegistry:
    """Gestiona el catálogo de herramientas built-in, generales y personales."""

    def __init__(self, shared_dir: Path):
        self.shared_dir = shared_dir
        self.shared_tools_dir = shared_dir / "tools"
        self.shared_recipes_dir = shared_dir / "recipes"

        # Asegurar directorios compartidos si existen permisos
        try:
            self.shared_tools_dir.mkdir(parents=True, exist_ok=True)
            self.shared_recipes_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def get_core_tools(self) -> List[Dict[str, Any]]:
        """Herramienta básica inicial para el LLM."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_tools",
                    "description": "Searches the agent's memory for tools and recipes. Write the query IN ENGLISH. Example: 'list files in directory'. Returns matching tools you can use in your next response.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "English description of the capability you need. Be specific."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    def get_builtin_tools(self) -> List[Dict[str, Any]]:
        """Herramientas del sistema siempre disponibles mediante search_tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_files_in_dir",
                    "description": "Lists all files and folders in a given directory (defaults to current directory).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {
                                "type": "string",
                                "description": "Directory to list (optional). Defaults to current."
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Reads and returns the content of a specific file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The exact path of the file to read."
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Edits an existing text file by replacing old text with new text, or creates a simple text file. Do NOT use this to create agent tools or recipes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path of the file to edit or create."
                            },
                            "prev_text": {
                                "type": "string",
                                "description": "Text to be replaced (can be empty for new files)."
                            },
                            "new_text": {
                                "type": "string",
                                "description": "New text to insert or replace."
                            }
                        },
                        "required": ["path", "new_text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Gets the current weather forecast for a city. Use ONLY if the user explicitly asks about the weather, temperature, or rain.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The name of the city."
                            }
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_tool",
                    "description": "Teaches the agent a new Python skill by writing code. Specify scope as 'personal' (default, saved to user's home) or 'general' (system-wide, requires admin).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Unique function name (snake_case)."},
                            "description": {"type": "string", "description": "Clear description in English of what the tool does."},
                            "parameters": {"type": "object", "description": "JSON schema of the function parameters."},
                            "code": {"type": "string", "description": "Python code. Must start with 'def function_name(...):'"},
                            "scope": {"type": "string", "enum": ["personal", "general"], "description": "Scope of the tool. 'general' requires admin."}
                        },
                        "required": ["name", "description", "parameters", "code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_recipe",
                    "description": "Saves a reusable multi-step workflow. Specify scope as 'personal' (default) or 'general' (requires admin).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Unique name for the recipe (snake_case)."},
                            "description": {"type": "string", "description": "Clear English description of what the recipe does."},
                            "parameters": {"type": "object", "description": "JSON schema of input parameters the recipe needs."},
                            "steps": {
                                "type": "array",
                                "description": "Ordered list of steps with tool and args.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "tool": {"type": "string"},
                                        "args": {"type": "object"}
                                    },
                                    "required": ["tool", "args"]
                                }
                            },
                            "scope": {"type": "string", "enum": ["personal", "general"], "description": "Scope of the recipe."}
                        },
                        "required": ["name", "description", "steps"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_recipe",
                    "description": "Executes a previously saved recipe by name, filling in the provided parameters.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "recipe_name": {"type": "string", "description": "Name of the saved recipe to run."},
                            "params": {"type": "object", "description": "Input values to fill into the recipe's parameter placeholders."}
                        },
                        "required": ["recipe_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_terminal_command",
                    "description": "Executes a command in the user's system shell and returns output. Runs as the requesting user.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command to execute in the terminal."
                            }
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Performs a web search to find information on the internet. Returns a list of relevant URLs.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    def get_user_dirs(self, linux_user: str) -> Tuple[Path, Path]:
        """Devuelve (user_tools_dir, user_recipes_dir) en el home del usuario."""
        home = get_user_home(linux_user)
        base = home / ".celab-bot"
        return base / "tools", base / "recipes"

    def load_dynamic_tools_and_recipes(self, linux_user: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Carga herramientas y recetas:
        1. Generales (/opt/celab-bot/shared)
        2. Personales (~/.celab-bot/)
        Si hay conflicto de nombre, la personal del usuario tiene prioridad.
        """
        tools_map = {}
        recipes_map = {}

        # 1. Herramientas y recetas generales
        if self.shared_tools_dir.is_dir():
            for f in self.shared_tools_dir.glob("*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        schema = json.load(fp)
                        tools_map[schema["function"]["name"]] = schema
                except Exception:
                    pass

        if self.shared_recipes_dir.is_dir():
            for f in self.shared_recipes_dir.glob("*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        recipe = json.load(fp)
                        recipes_map[recipe["name"]] = recipe
                except Exception:
                    pass

        # 2. Herramientas y recetas personales
        user_tools_dir, user_recipes_dir = self.get_user_dirs(linux_user)
        if user_tools_dir.is_dir():
            for f in user_tools_dir.glob("*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        schema = json.load(fp)
                        tools_map[schema["function"]["name"]] = schema
                except Exception:
                    pass

        if user_recipes_dir.is_dir():
            for f in user_recipes_dir.glob("*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        recipe = json.load(fp)
                        recipes_map[recipe["name"]] = recipe
                except Exception:
                    pass

        return list(tools_map.values()), recipes_map
