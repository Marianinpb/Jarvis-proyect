import asyncio
import importlib
import importlib.util
import inspect
import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from celab_bot.config import CelabConfig
from celab_bot.memory import UserMemoryStore
from celab_bot.tools_registry import ToolsRegistry
from celab_bot.users import get_user_home, get_user_data_dir


_SPANISH_HINTS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
    "en", "por", "para", "que", "con", "sin", "haz", "dime", "lista",
    "listar", "lee", "leer", "crea", "crear", "archivo", "archivos",
    "carpeta", "carpetas", "directorio", "directorios", "hola", "si", "sí",
    "no", "quiero", "necesito", "puedes", "como", "cómo", "cual", "cuál",
    "donde", "dónde", "mi", "mis", "tu", "tus", "su", "sus", "este", "esta",
    "esto", "hacer", "ejecutar", "comando", "buscar", "busca", "guarda",
    "guardar", "borra", "borrar", "elimina", "eliminar", "muestra", "mostrar",
    "texto", "espacio", "trabajo", "servidor", "usuario", "memoria", "porfa",
    "favor", "gracias", "cambia", "edita", "editar", "ayuda", "clima",
}


def looks_like_english(text: str) -> bool:
    """Heurística rápida para evitar traducir textos que ya están en inglés."""
    if not text or not isinstance(text, str):
        return True
    s = text.strip()
    if not s:
        return True
    if any(ch in s for ch in "áéíóúüñÁÉÍÓÚÜÑ¿¡"):
        return False
    tokens = set(re.findall(r"[a-zA-Z]+", s.lower()))
    if not tokens:
        return True
    if tokens & _SPANISH_HINTS:
        return False
    return True


def detect_repetition_loop(text: str) -> bool:
    """Detecta bucles degenerados de repetición típicos de modelos cuantizados a 1-bit."""
    if not text or len(text) < 220:
        return False
    tail = text[-900:]
    return bool(re.search(r"(.{20,120}?)\1{4,}", tail, flags=re.DOTALL))


def extract_fallback_tool_calls(
    content: str,
    available_tools: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Parser de respaldo para cuando llama.cpp / Bonsai-27b-Q1_0 emite una llamada a tool
    en texto plano (<tool_call>{...}</tool_call> o bloque JSON) en lugar de msg.tool_calls.
    """
    if not content or not available_tools:
        return []

    valid_names = {
        t.get("function", {}).get("name")
        for t in available_tools
        if isinstance(t, dict) and t.get("function", {}).get("name")
    }
    if not valid_names:
        return []

    candidates: List[str] = []
    # 1. Etiquetas <tool_call>...</tool_call>
    for m in re.finditer(r"<tool_call>([\s\S]*?)</tool_call>", content, flags=re.IGNORECASE):
        inner = m.group(1).strip()
        if inner.startswith("{") and inner.endswith("}"):
            candidates.append(inner)

    # 2. Bloques ```json ... ```
    if not candidates:
        for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", content, flags=re.IGNORECASE):
            inner = m.group(1).strip()
            if inner.startswith("{") and inner.endswith("}"):
                candidates.append(inner)

    # 3. JSON crudo si toda la respuesta es un objeto JSON
    stripped = content.strip()
    if not candidates and stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)

    parsed_calls: List[Dict[str, Any]] = []
    for raw in candidates:
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        fn_name = obj.get("name") or obj.get("function") or obj.get("tool")
        if isinstance(fn_name, dict):
            fn_name = fn_name.get("name")
        if fn_name not in valid_names:
            continue

        raw_args = obj.get("arguments")
        if raw_args is None:
            raw_args = obj.get("parameters")
        if raw_args is None:
            raw_args = obj.get("args", {})
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except Exception:
                raw_args = {}
        if not isinstance(raw_args, dict):
            raw_args = {}

        parsed_calls.append({
            "id": f"call_fb_{uuid.uuid4().hex[:10]}",
            "name": fn_name,
            "arguments": raw_args,
        })

    return parsed_calls


class AgentEngine:
    """Motor agéntico principal de CELab bot desacoplado de estado global."""

    MEMORY_CATEGORIES = {
        "user_preferences": "Identity memory: immutable rules, personal user preferences, language, formatting or style instructions",
        "project_context": "Working memory: technical details of the user's current task or project (files, configs, errors, decisions)",
        "general_knowledge": "Semantic memory: concepts, definitions, general facts or reusable code snippets",
        "conversational_context": "Episodic memory: recent conversational thread, greetings, small talk, acknowledgements",
    }

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
    # TRADUCCIÓN INCREMENTAL AL INGLÉS (CON CACHÉ EN RAM + SQLITE)
    # ─────────────────────────────────────────────────────────────────────
    async def atranslate_to_english(self, text: str) -> str:
        """Traduce un texto a inglés de forma asíncrona con timeout acotado."""
        if not text or not isinstance(text, str):
            return ""
        text = text.strip()
        if not text:
            return ""
        if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
            return text
        if looks_like_english(text):
            return text
        if self.client is None:
            return text

        messages = [
            {
                "role": "system",
                "content": "Translate the user message directly to concise English. Respond ONLY with the translation, no extra commentary or quotes.",
            },
            {"role": "user", "content": text[:800]},
        ]
        try:
            create_fn = self.client.chat.completions.create
            kwargs = {
                "model": self.config.llm_model,
                "messages": messages,
                "max_tokens": 120,
                "temperature": 0.0,
                "timeout": 15.0,
            }
            if inspect.iscoroutinefunction(create_fn):
                resp = await create_fn(**kwargs)
            else:
                resp = await asyncio.to_thread(create_fn, **kwargs)
            res = (resp.choices[0].message.content or "").strip()
            if res.startswith('"') and res.endswith('"') and len(res) > 1:
                res = res[1:-1]
            return res if res else text
        except Exception as e:
            print(f"  ⚠️ [TRANSLATE] Error traduciendo a inglés: {e}")
            return text

    async def aget_text_en(
        self,
        text: str,
        memory_store: Optional[UserMemoryStore] = None,
    ) -> str:
        """Obtiene y cachea la versión en inglés (RAM -> SQLite -> LLM)."""
        if not text or not isinstance(text, str):
            return ""
        text_clean = text.strip()
        if not text_clean:
            return ""
        if text_clean in self._translation_cache:
            return self._translation_cache[text_clean]
        if looks_like_english(text_clean):
            self._translation_cache[text_clean] = text_clean
            return text_clean
        if memory_store is not None:
            cached = memory_store.get_translation(text_clean)
            if cached:
                self._translation_cache[text_clean] = cached
                return cached

        en = await self.atranslate_to_english(text_clean)
        self._translation_cache[text_clean] = en
        if memory_store is not None and en:
            memory_store.save_translation(text_clean, en)
        return en

    async def aget_message_en(
        self,
        msg: Any,
        memory_store: Optional[UserMemoryStore] = None,
    ) -> str:
        """Obtiene la versión en inglés de un mensaje individual sin retraducir."""
        if isinstance(msg, dict):
            if "_content_en" in msg:
                return msg["_content_en"]
            content = msg.get("content", "")
            en = await self.aget_text_en(content, memory_store)
            msg["_content_en"] = en
            return en
        if hasattr(msg, "_content_en"):
            return getattr(msg, "_content_en")
        content = getattr(msg, "content", "")
        en = await self.aget_text_en(content, memory_store)
        setattr(msg, "_content_en", en)
        return en

    # ─────────────────────────────────────────────────────────────────────
    # FILTRADO DE CONTEXTO Y CLASIFICACIÓN EN SQLITE CON LAYA
    # ─────────────────────────────────────────────────────────────────────
    async def aget_filtered_messages(
        self,
        messages: List[Dict[str, Any]],
        current_task: str,
        memory_store: UserMemoryStore,
    ) -> List[Dict[str, Any]]:
        """
        Filtra el historial de conversación con Laya Context Router (sin bloquear el event loop),
        reutiliza traducciones/clasificaciones de SQLite y almacena recuerdos relevantes.
        """
        if len(messages) <= 2 or self.context_router is None:
            return messages

        last_user_idx = len(messages) - 1
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            role = msg.get("role") if isinstance(msg, dict) else msg.role
            if role == "user":
                last_user_idx = i
                break

        current_task_en = await self.aget_text_en(current_task, memory_store)
        filtered = [messages[0]]  # System prompt

        # Acotar evaluación según el tamaño de contexto configurado (evita latencia en chats largos)
        history_window = 32 if self.config.llm_ctx_size >= 32768 else 16
        start_idx = max(1, last_user_idx - history_window)
        i = start_idx
        while i < last_user_idx:
            msg = messages[i]
            role = msg.get("role") if isinstance(msg, dict) else msg.role
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)

            block = [msg]
            content_for_eval = str(content) if content else ""
            msg_en = await self.aget_message_en(msg, memory_store)
            content_for_eval_en = f"Role: {role}, Content: {msg_en}"

            if tool_calls:
                tc_name = (
                    tool_calls[0].get("function", {}).get("name")
                    if isinstance(tool_calls[0], dict)
                    else tool_calls[0].function.name
                )
                content_for_eval += f" [Llamada a tool: {tc_name}]"
                content_for_eval_en += f" [Tool call: {tc_name}]"

                if i + 1 < last_user_idx:
                    next_msg = messages[i + 1]
                    next_role = next_msg.get("role") if isinstance(next_msg, dict) else next_msg.role
                    if next_role == "tool":
                        block.append(next_msg)
                        next_content = (
                            next_msg.get("content")
                            if isinstance(next_msg, dict)
                            else getattr(next_msg, "content", "")
                        )
                        next_en = await self.aget_message_en(next_msg, memory_store)
                        content_for_eval += f" [Resultado: {str(next_content)[:100]}]"
                        content_for_eval_en += f" [Result: {str(next_en)[:100]}]"
                        i += 1

            if role == "user" and i + 1 < last_user_idx:
                next_msg = messages[i + 1]
                next_role = next_msg.get("role") if isinstance(next_msg, dict) else next_msg.role
                if next_role == "assistant":
                    block.append(next_msg)
                    next_content = (
                        next_msg.get("content")
                        if isinstance(next_msg, dict)
                        else getattr(next_msg, "content", "")
                    )
                    asst_en = await self.aget_message_en(next_msg, memory_store)
                    content_for_eval = f"User: {content_for_eval} | Assistant: {str(next_content)}"
                    content_for_eval_en = f"Role: user, Content: {msg_en}\nRole: assistant, Content: {asst_en}"
                    i += 1

            cached_cls = memory_store.get_cached_message_classification(content_for_eval)
            state = {
                "current_task": current_task_en,
                "context_piece": content_for_eval_en,
            }
            questions = {
                "is_relevant": {
                    "type": "noul",
                    "instructions": "Is this exchange strictly necessary to solve the current task?",
                },
                "should_store": {
                    "type": "noul",
                    "instructions": "Does this exchange contain at least one piece of information worth saving in long-term memory for future conversations?",
                },
                "memory_type": {
                    "type": "choice",
                    "instructions": "Which type of memory does the most important information in this exchange belong to?",
                    "criteria": self.MEMORY_CATEGORIES,
                },
            }

            try:
                res = await asyncio.to_thread(self.context_router.predict, state, questions)
                ans = res.get("answers", {})
                prob_yes = float(ans.get("is_relevant", {}).get("noul", 0.0))
                prob_store = float(ans.get("should_store", {}).get("noul", 0.0))
                category = ans.get("memory_type", {}).get("choice")

                if not cached_cls:
                    memory_store.log_message(
                        role=role,
                        content=content_for_eval,
                        content_en=content_for_eval_en,
                        category=category,
                        is_relevant_score=prob_yes,
                        should_store_score=prob_store,
                    )

                if prob_yes > 0.5:
                    filtered.extend(block)

                if prob_store > 0.5 and category in memory_store.CATEGORIES:
                    memory_store.add(
                        category=category,
                        content=content_for_eval,
                        content_en=content_for_eval_en,
                        importance=prob_store,
                    )

            except Exception as e:
                print(f"⚠️ [CONTEXT] Error evaluando memoria: {e}. Reteniendo.")
                filtered.extend(block)

            i += 1

        filtered.extend(messages[last_user_idx:])
        return filtered

    async def aclassify_and_store_turn(
        self,
        user_text: str,
        assistant_text: str,
        memory_store: UserMemoryStore,
    ) -> None:
        """Clasifica el turno recién completado en segundo plano y lo guarda en SQLite."""
        if self.context_router is None:
            return
        user_clean = (user_text or "").strip()
        asst_clean = (assistant_text or "").strip()
        if not user_clean or not asst_clean:
            return

        content_es = f"User: {user_clean} | Assistant: {asst_clean[:600]}"
        if memory_store.get_cached_message_classification(content_es):
            return

        try:
            user_en = await self.aget_text_en(user_clean, memory_store)
            asst_en = await self.aget_text_en(asst_clean[:600], memory_store)
            content_en = f"Role: user, Content: {user_en}\nRole: assistant, Content: {asst_en}"

            state = {
                "current_task": user_en,
                "context_piece": content_en,
            }
            questions = {
                "should_store": {
                    "type": "noul",
                    "instructions": "Does this exchange contain at least one piece of information worth saving in long-term memory for future conversations?",
                },
                "memory_type": {
                    "type": "choice",
                    "instructions": "Which type of memory does the most important information in this exchange belong to?",
                    "criteria": self.MEMORY_CATEGORIES,
                },
            }
            res = await asyncio.to_thread(self.context_router.predict, state, questions)
            ans = res.get("answers", {})
            prob_store = float(ans.get("should_store", {}).get("noul", 0.0))
            category = ans.get("memory_type", {}).get("choice")

            memory_store.log_message(
                role="exchange",
                content=content_es,
                content_en=content_en,
                category=category,
                is_relevant_score=1.0,
                should_store_score=prob_store,
            )

            if prob_store > 0.5 and category in memory_store.CATEGORIES:
                added = memory_store.add(
                    category=category,
                    content=content_es,
                    content_en=content_en,
                    importance=prob_store,
                )
                if added:
                    print(f"🧠 [MEMORY] Nuevo recuerdo guardado ({category}, score={prob_store:.2f}) para {memory_store.linux_user}")
        except Exception as e:
            print(f"⚠️ [MEMORY BG] Error clasificando turno en segundo plano: {e}")

    async def abuild_memory_context(
        self,
        current_query: str,
        memory_store: UserMemoryStore,
    ) -> str:
        """Recupera recuerdos relevantes de SQLite formateados para el System Prompt."""
        try:
            query_en = await self.aget_text_en(current_query, memory_store)
            return memory_store.format_prompt_context(
                query=current_query,
                query_en=query_en,
                max_items=self.config.max_memory_inject,
            )
        except Exception as e:
            print(f"⚠️ [MEMORY] Error recuperando contexto de memoria: {e}")
            return ""

    # ─────────────────────────────────────────────────────────────────────
    # BÚSQUEDA DE TOOLS CON LAYA TOOL ROUTER (ASYNC + TRADUCCIÓN + TOP-K)
    # ─────────────────────────────────────────────────────────────────────
    async def asearch_tools(
        self,
        query: str,
        linux_user: str,
        memory_store: Optional[UserMemoryStore] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Busca herramientas y recetas usando Laya asegurando query en inglés y fallback Top-K."""
        if self.tool_router is None:
            print("⚠️ [LAYA TOOLS] tool_router no está cargado.")
            return [], {}

        query_en = await self.aget_text_en(query, memory_store)
        excluded = set()

        dynamic_tools, dynamic_recipes = self.tools_registry.load_dynamic_tools_and_recipes(linux_user)
        builtins = self.tools_registry.get_builtin_tools(exclude_names=excluded)

        searchable = []
        desc_to_item = {}

        for tool in builtins + dynamic_tools:
            if tool["function"]["name"] in excluded:
                continue
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
        state = {"question": query_en}
        questions = {
            "match": {
                "type": "choice",
                "instructions": "Which capability best matches what is needed?",
                "criteria": searchable,
            }
        }

        try:
            result = await asyncio.to_thread(self.tool_router.predict, state, questions)
            match_ans = result.get("answers", {}).get("match", {})
            probabilities = match_ans.get("probabilities", {})
        except Exception as e:
            print(f"⚠️ [LAYA TOOLS] Error en search_tools: {e}")
            return [], {}

        none_prob = float(probabilities.get("none of the above", 0.0))
        scored_items = sorted(
            [(desc, float(prob)) for desc, prob in probabilities.items() if desc != "none of the above"],
            key=lambda x: x[1],
            reverse=True,
        )

        top_debug = [
            f"{desc_to_item[d]['schema']['function']['name'] if desc_to_item[d]['type'] == 'tool' else desc_to_item[d]['name']}={p:.2f}"
            for d, p in scored_items[:3]
            if d in desc_to_item
        ]
        print(f"🔎 [LAYA TOOLS] query='{query}' (en='{query_en}') none={none_prob:.2f} top={top_debug}")

        found_tools: List[Dict[str, Any]] = []
        found_recipes: Dict[str, Any] = {}

        for desc, prob in scored_items:
            if prob > self.config.search_threshold:
                item = desc_to_item.get(desc)
                if item and item["type"] == "tool":
                    schema = item["schema"]
                    if schema not in found_tools:
                        found_tools.append(schema)
                elif item and item["type"] == "recipe":
                    found_recipes[item["name"]] = item["recipe"]

        # Red de seguridad Top-K: si nada pasó el threshold pero "none of the above" no domina
        if not found_tools and not found_recipes and none_prob < 0.85:
            for desc, prob in scored_items[:2]:
                if prob >= 0.01:
                    item = desc_to_item.get(desc)
                    if item and item["type"] == "tool":
                        found_tools.append(item["schema"])
                    elif item and item["type"] == "recipe":
                        found_recipes[item["name"]] = item["recipe"]

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
        if not self.config.single_user_mode and os.name == "posix":
            cmd = ["sudo", "-u", linux_user, str(self.config.exec_binary)]
            payload = json.dumps({
                "tool": tool_name,
                "args": args,
                "user": linux_user,
                "is_admin": is_admin,
            })
            try:
                res = subprocess.run(
                    cmd,
                    input=payload,
                    text=True,
                    capture_output=True,
                    timeout=30.0,
                    cwd="/tmp",
                )
                if res.returncode != 0:
                    return {"error": f"Error ejecutando como {linux_user}: {res.stderr}"}
                return json.loads(res.stdout)
            except Exception as e:
                return {"error": f"Error llamando celab-exec: {e}"}

        return self._execute_tool_locally(tool_name, args, linux_user, is_admin)

    def _execute_tool_locally(
        self,
        tool_name: str,
        args: Dict[str, Any],
        linux_user: str,
        is_admin: bool,
    ) -> Dict[str, Any]:
        """Ejecución directa en el mismo proceso (fallback dev o dentro de celab-exec)."""
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
                    data = f.read()
                # Truncar lecturas enormes de forma proporcional al ctx_size configurado
                max_read_chars = min(60000, max(12000, self.config.llm_ctx_size))
                if len(data) > max_read_chars:
                    return {
                        "content": data[:max_read_chars],
                        "truncated": True,
                        "total_chars": len(data),
                    }
                return {"content": data}
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

            target_dir = (
                self.config.shared_dir / "tools"
                if scope == "general"
                else get_user_data_dir(linux_user) / "tools"
            )
            target_dir.mkdir(parents=True, exist_ok=True)

            schema = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": params if isinstance(params, dict) else json.loads(params),
                },
                "scope": scope,
                "author": linux_user,
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
            target_dir = (
                self.config.shared_dir / "recipes"
                if scope == "general"
                else get_user_data_dir(linux_user) / "recipes"
            )
            target_dir.mkdir(parents=True, exist_ok=True)

            recipe_data = {
                "name": name,
                "description": args.get("description", ""),
                "parameters": args.get("parameters", {}),
                "steps": args.get("steps", []),
                "scope": scope,
                "author": linux_user,
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
            for step in recipe.get("steps", []):
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
                    cwd=str(home),
                )
                stdout = res.stdout or ""
                stderr = res.stderr or ""
                max_stdout_chars = min(32000, max(8000, self.config.llm_ctx_size // 2))
                max_stderr_chars = min(12000, max(4000, self.config.llm_ctx_size // 6))
                if len(stdout) > max_stdout_chars:
                    stdout = stdout[:max_stdout_chars] + "\n... [salida truncada]"
                if len(stderr) > max_stderr_chars:
                    stderr = stderr[:max_stderr_chars] + "\n... [stderr truncado]"
                return {
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": res.returncode,
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
            self.config.shared_dir / "tools" / f"{tool_name}.py",
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
