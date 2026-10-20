#!/usr/bin/env python3
"""
Módulo runner para celab-exec.
"""

import json
import os
import sys
from pathlib import Path

from celab_bot.config import load_config
from celab_bot.tools_registry import ToolsRegistry


def main():
    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            print(json.dumps({"error": "No se recibió payload en stdin"}))
            sys.exit(1)

        payload = json.loads(raw_input)
        tool_name = payload.get("tool")
        args = payload.get("args", {})
        linux_user = payload.get("user")
        is_admin = bool(payload.get("is_admin", False))

        if not tool_name or not linux_user:
            print(json.dumps({"error": "Falta 'tool' o 'user' en el payload"}))
            sys.exit(1)

        config = load_config()
        registry = ToolsRegistry(config.shared_dir)

        from celab_bot.core import AgentEngine
        engine = AgentEngine(
            client=None,
            config=config,
            tool_router=None,
            context_router=None,
            tools_registry=registry
        )

        result = engine._execute_tool_locally(tool_name, args, linux_user, is_admin)
        print(json.dumps(result, ensure_ascii=False))

    except Exception as e:
        print(json.dumps({"error": f"Fallo interno en celab-exec: {str(e)}"}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
