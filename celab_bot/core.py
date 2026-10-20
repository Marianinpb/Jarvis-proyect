import importlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from celab_bot.config import CelabConfig
from celab_bot.memory import UserMemoryStore
from celab_bot.security import requires_user_confirmation
from celab_bot.tools_registry import ToolsRegistry
from celab_bot.users import get_user_home


class AgentEngine:
    """Motor agéntico principal de CELab bot desacoplado de estado global."""

    def __init__(
        self,
        client: Any,
        config: CelabConfig,
        tool_router: Any,
        context_router: Any,
        tools_registry: Optional[ToolsRegistry] = None,
    ):
        self.client = client
        self.config = config
        self.tool_router = tool_router
        self.context_router = context_router
        self.tools_registry = tools_registry or ToolsRegistry(config.shared_dir)
        self._translation_cache: Dict[str, str] = {}

    # ─────────────────────────────────────────────────────────────────────
    # TRADUCCIÓN INCREMENTAL AL INGLÉS (CACHEADA PARA LAYA)
    # ─────────────────────────────────────────────────────────────────────
    def translate_to_english(self, text: str) -> str:
        """Traduce un mensaje individual a inglés usando el LLM de forma concisa."""
        if not text or not isinstance(text, str):
            return ""
        text = text.strip()
        if not text:
            return ""
        if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
            return text

        try:
            resp = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Translate the user message directly to concise English. Respond ONLY with the translation, no extra commentary or quotes."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                max_tokens=120,
                temperature=0.0
            )
            res = resp.choices[0].message.content.strip()
            if res.startswith('"') and res.endswith('"') and len(res) > 1:
                res = res[1:-1]
            return res if res else text
        except Exception as e:
            print(f"  ⚠️ [TRANSLATE] Error traduciendo a inglés: {e}")
            return text

    def get_text_en(self, text: str) -> str:
        """Obtiene y cachea la versión en inglés de un texto."""
        if not text or not isinstance(text, str):
            return ""
        text_clean = text.strip()
        if text_clean in self._translation_cache:
            return self._translation_cache[text_clean]
        en = self.translate_to_english(text_clean)
        self._translation_cache[text_clean] = en
        return en

    def get_message_en(self, msg: Any) -> str:
        """Obtiene la versión en inglés de un mensaje individual (1 sola vez)."""
        if isinstance(msg, dict):
            if "_content_en" in msg:
                return msg["_content_en"]
            content = msg.get("content", "")
            en = self.get_text_en(content)
            msg["_content_en"] = en
            return en
        else:
            if hasattr(msg, "_content_en"):
                return getattr(msg, "_content_en")
            content = getattr(msg, "content", "")
            en = self.get_text_en(content)
            setattr(msg, "_content_en", en)
            return en

    # ─────────────────────────────────────────────────────────────────────
    # FILTRADO DE CONTEXTO CON LAYA (ENVÍO EN INGLÉS)
    # ─────────────────────────────────────────────────────────────────────
    def get_filtered_messages(
        self,
        messages: List[Dict[str, Any]],
        current_task: str,
        memory_store: UserMemoryStore,
    ) -> List[Dict[str, Any]]:
        """
        Filtra el historial de conversación con Laya Context Router.
        A Laya se le envía todo en inglés (current_task y context_piece).
        Al LLM se le devuelven los mensajes originales en español.
        """
        if len(messages) <= 2:
            return messages

        last_user_idx = len(messages) - 1
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            role = msg.get("role") if isinstance(msg, dict) else msg.role
            if role == "user":
                last_user_idx = i
                break

        current_task_en = self.get_text_en(current_task)
        filtered = [messages[0]]  # System prompt

        memory_categories = {
            "user_preferences": "Identity memory: immutable rules, personal user preferences, language, formatting or style instructions",
            "project_context": "Working memory: technical details of the user's current task or project (files, configs, errors, decisions)",
            "general_knowledge": "Semantic memory: concepts, definitions, general facts or reusable code snippets",
            "conversational_context": "Episodic memory: recent conversational thread, greetings, small talk, acknowledgements"
        }

        i = 1
        while i < last_user_idx:
            msg = messages[i]
            role = msg.get("role") if isinstance(msg, dict) else msg.role
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)

            block = [msg]
            content_for_eval = str(content) if content else ""
            msg_en = self.get_message_en(msg)
            content_for_eval_en = f"Role: {role}, Content: {msg_en}"

            if tool_calls:
                tc_name = tool_calls[0].get("function", {}).get("name") if isinstance(tool_calls[0], dict) else tool_calls[0].function.name
                content_for_eval += f" [Llamada a tool: {tc_name}]"
                content_for_eval_en += f" [Tool call: {tc_name}]"

                if i + 1 < last_user_idx:
                    next_msg = messages[i + 1]
                    next_role = next_msg.get("role") if isinstance(next_msg, dict) else next_msg.role
                    if next_role == "tool":
                        block.append(next_msg)
                        next_content = next_msg.get("content") if isinstance(next_msg, dict) else getattr(next_msg, "content", "")
                        next_en = self.get_message_en(next_msg)
                        content_for_eval += f" [Resultado: {str(next_content)[:100]}]"
                        content_for_eval_en += f" [Result: {str(next_en)[:100]}]"
                        i += 1

            if role == "user" and i + 1 < last_user_idx:
                next_msg = messages[i + 1]
                next_role = next_msg.get("role") if isinstance(next_msg, dict) else next_msg.role
                if next_role == "assistant":
                    block.append(next_msg)
                    next_content = next_msg.get("content") if isinstance(next_msg, dict) else getattr(next_msg, "content", "")
                    asst_en = self.get_message_en(next_msg)
                    content_for_eval = f"User: {content_for_eval} | Assistant: {str(next_content)}"
                    content_for_eval_en = f"Role: user, Content: {msg_en}\nRole: assistant, Content: {asst_en}"
                    i += 1

            state = {
                "current_task": current_task_en,
                "context_piece": content_for_eval_en
            }
            questions = {
                "is_relevant": {
                    "type": "noul",
                    "instructions": "Is this exchange strictly necessary to solve the current task?"
                },
                "should_store": {
                    "type": "noul",
                    "instructions": "Does this exchange contain at least one piece of information worth saving in long-term memory for future conversations?"
                },
                "memory_type": {
                    "type": "choice",
                    "instructions": "Which type of memory does the most important information in this exchange belong to?",
                    "criteria": memory_categories
                }
            }

            try:
                res = self.context_router.predict(state, questions)
                ans = res.get("answers", {})
                prob_yes = ans.get("is_relevant", {}).get("noul", 0.0)
                prob_store = ans.get("should_store", {}).get("noul", 0.0)
                category = ans.get("memory_type", {}).get("choice")

                # RELEVANCIA
                if prob_yes > 0.5:
                    filtered.extend(block)

                # PERSISTENCIA EN MEMORIA DEL USUARIO
                if prob_store > 0.5 and category in memory_store.CATEGORIES:
                    memory_store.add(category, content_for_eval)

            except Exception as e:
                print(f"⚠️ [CONTEXT] Error evaluando memoria: {e}. Reteniendo.")
                filtered.extend(block)

            i += 1

        filtered.extend(messages[last_user_idx:])
        return filtered

    # ─────────────────────────────────────────────────────────────────────
    # BÚSQUEDA DE TOOLS CON LAYA TOOL ROUTER
    # ─────────────────────────────────────────────────────────────────────
    def search_tools(
        self,
        query: str,
        linux_user: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Busca herramientas y recetas usando Laya."""
        dynamic_tools, dynamic_recipes = self.tools_registry.load_dynamic_tools_and_recipes(linux_user)
        builtins = self.tools_registry.get_builtin_tools()

        searchable = []
        desc_to_item = {}

        for tool in builtins + dynamic_tools:
            desc = tool["function"]["description"]
            desc_to_item[desc] = {"type": "tool", "schema": tool}
            searchable.append(desc)

        for name, recipe in dynamic_recipes.items():
            desc = recipe.get("description", name)
            desc_to_item[desc] = {"type": "recipe", "name": name, "recipe": recipe}
            searchable.append(desc)

        if not searchable:
            return [], {}

        searchable.append("none of the above")
        state = {"question": query}
        questions = {
            "match": {
                "type": "choice",
                "instructions": "Which capability best matches what is needed?",
                "criteria": searchable
            }
        }

        try:
            result = self.tool_router.predict(state, questions)
            match_ans = result.get("answers", {}).get("match", {})
            probabilities = match_ans.get("probabilities", {})
        except Exception as e:
            print(f"⚠️ [LAYA TOOLS] Error en search_tools: {e}")
            return [], {}

        found_tools = []
        found_recipes = {}

        for desc, prob in probabilities.items():
            if desc == "none of the above":
                continue
            if prob > self.config.search_threshold:
                item = desc_to_item.get(desc)
                if item and item["type"] == "tool":
                    schema = item["schema"]
                    if schema not in found_tools:
                        found_tools.append(schema)
                elif item and item["type"] == "recipe":
                    name = item["name"]
                    found_recipes[name] = item["recipe"]

        return found_tools, found_recipes

    # ─────────────────────────────────────────────────────────────────────
    # EJECUCIÓN AISLADA (celab-exec O LOCAL)
    # ─────────────────────────────────────────────────────────────────────
    def execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        linux_user: str,
        is_admin: bool,
    ) -> Dict[str, Any]:
        """Ejecuta una herramienta como el usuario Linux correspondiente."""
        # Modo multiusuario en Linux: usar celab-exec mediante sudo -u
        if not self.config.single_user_mode and os.name == "posix":
            cmd = ["sudo", "-u", linux_user, str(self.config.exec_binary)]
            payload = json.dumps({
                "tool": tool_name,
                "args": args,
                "user": linux_user,
                "is_admin": is_admin
            })
            try:
                res = subprocess.run(
                    cmd,
                    input=payload,
                    text=True,
                    capture_output=True,
                    timeout=30.0,
                    cwd=str(get_user_home(linux_user))
                )
                if res.returncode != 0:
                    return {"error": f"Error ejecutando como {linux_user}: {res.stderr}"}
                return json.loads(res.stdout)
            except Exception as e:
                return {"error": f"Error llamando celab-exec: {e}"}

        # Modo local / Dev fallback: ejecución directa
        return self._execute_tool_locally(tool_name, args, linux_user, is_admin)

    def _execute_tool_locally(
        self,
        tool_name: str,
        args: Dict[str, Any],
        linux_user: str,
        is_admin: bool,
    ) -> Dict[str, Any]:
        """Ejecución directa en el mismo proceso (fallback dev)."""
        home = get_user_home(linux_user)

        if tool_name == "list_files_in_dir":
            directory = args.get("directory", ".")
            target = (home / directory).resolve() if not os.path.isabs(directory) else Path(directory)
            try:
                return {"files": os.listdir(target)}
            except Exception as e:
                return {"error": str(e)}

        elif tool_name == "read_file":
            path = Path(args.get("path", ""))
            target = (home / path).resolve() if not path.is_absolute() else path
            try:
                with open(target, "r", encoding="utf-8") as f:
                    return {"content": f.read()}
            except Exception as e:
                return {"error": str(e)}

        elif tool_name == "edit_file":
            path = Path(args.get("path", ""))
            target = (home / path).resolve() if not path.is_absolute() else path
            prev = args.get("prev_text", "")
            new = args.get("new_text", "")
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and prev:
                    with open(target, "r", encoding="utf-8") as f:
                        data = f.read()
                    if prev not in data:
                        return {"error": f"Texto previo no encontrado en {path}"}
                    data = data.replace(prev, new)
                else:
                    data = new

                with open(target, "w", encoding="utf-8") as f:
                    f.write(data)
                return {"status": f"Archivo {path} guardado exitosamente."}
            except Exception as e:
                return {"error": str(e)}

        elif tool_name == "get_weather":
            loc = args.get("location", "Desconocida")
            return {"clima": "Soleado", "temperatura": "24°C", "ubicacion": loc}

        elif tool_name == "create_tool":
            scope = args.get("scope", "personal")
            if scope == "general" and not is_admin:
                return {"error": "Solo administradores pueden crear herramientas con scope 'general'."}

            name = args.get("name")
            desc = args.get("description")
            params = args.get("parameters")
            code = args.get("code")

            target_dir = self.config.shared_dir / "tools" if scope == "general" else home / ".celab-bot" / "tools"
            target_dir.mkdir(parents=True, exist_ok=True)

            schema = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": params if isinstance(params, dict) else json.loads(params)
                },
                "scope": scope,
                "author": linux_user
            }

            with open(target_dir / f"{name}.json", "w", encoding="utf-8") as f:
                json.dump(schema, f, indent=4)

            with open(target_dir / f"{name}.py", "w", encoding="utf-8") as f:
                f.write(code)

            return {"status": f"Herramienta '{name}' creada con scope {scope}."}

        elif tool_name == "create_recipe":
            scope = args.get("scope", "personal")
            if scope == "general" and not is_admin:
                return {"error": "Solo administradores pueden crear recetas con scope 'general'."}

            name = args.get("name")
            target_dir = self.config.shared_dir / "recipes" if scope == "general" else home / ".celab-bot" / "recipes"
            target_dir.mkdir(parents=True, exist_ok=True)

            recipe_data = {
                "name": name,
                "description": args.get("description", ""),
                "parameters": args.get("parameters", {}),
                "steps": args.get("steps", []),
                "scope": scope,
                "author": linux_user
            }
            with open(target_dir / f"{name}.json", "w", encoding="utf-8") as f:
                json.dump(recipe_data, f, indent=4)

            return {"status": f"Receta '{name}' guardada con scope {scope}."}

        elif tool_name == "run_recipe":
            recipe_name = args.get("recipe_name")
            params = args.get("params", {})
            _, recipes = self.tools_registry.load_dynamic_tools_and_recipes(linux_user)
            if recipe_name not in recipes:
                return {"error": f"Receta '{recipe_name}' no encontrada."}

            recipe = recipes[recipe_name]
            results = []
            for i, step in enumerate(recipe.get("steps", [])):
                t_name = step["tool"]
                raw_args = step.get("args", {})
                resolved = {}
                for k, v in raw_args.items():
                    if isinstance(v, str):
                        for pk, pv in params.items():
                            v = v.replace(f"{{{pk}}}", str(pv))
                        for j, prev_r in enumerate(results):
                            v = v.replace(f"{{step_{j}}}", str(prev_r))
                    resolved[k] = v

                r = self.execute_tool(t_name, resolved, linux_user, is_admin)
                results.append(r)
                if "error" in r:
                    break

            return {"status": f"Receta '{recipe_name}' ejecutada.", "results": results}

        elif tool_name == "run_terminal_command":
            cmd = args.get("command", "")
            try:
                res = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=20,
                    cwd=str(home)
                )
                return {
                    "stdout": res.stdout,
                    "stderr": res.stderr,
                    "returncode": res.returncode
                }
            except Exception as e:
                return {"error": str(e)}

        elif tool_name == "web_search":
            query = args.get("query", "")
            try:
                from googlesearch import search
                results = list(search(query, num_results=5, lang="es"))
                return {"results": results}
            except Exception as e:
                return {"error": f"Error en web_search: {e}"}

        # Búsqueda en tools dinámicas de usuario
        user_tools_dir, _ = self.tools_registry.get_user_dirs(linux_user)
        py_candidates = [
            user_tools_dir / f"{tool_name}.py",
            self.config.shared_dir / "tools" / f"{tool_name}.py"
        ]
        for py_path in py_candidates:
            if py_path.is_file():
                try:
                    spec = importlib.util.spec_from_file_location(tool_name, str(py_path))
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    if hasattr(mod, tool_name):
                        fn = getattr(mod, tool_name)
                        return fn(**args)
                except Exception as e:
                    return {"error": f"Error ejecutando {tool_name} desde {py_path.name}: {e}"}

        return {"error": f"Herramienta '{tool_name}' no encontrada."}
