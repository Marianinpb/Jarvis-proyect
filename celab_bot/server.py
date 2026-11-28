import asyncio
import json
import re
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from fastapi import FastAPI, Header, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from openai import AsyncOpenAI

from celab_bot.config import CelabConfig, load_config
from celab_bot.core import (
    AgentEngine,
    detect_repetition_loop,
    extract_fallback_tool_calls,
)
from celab_bot.memory import UserMemoryStore
from celab_bot.planner import (
    agenerate_plan,
    build_step_messages,
    build_synthesizer_messages,
    format_plan_markdown,
    should_skip_planner,
    summarize_step_outcome,
)
from celab_bot.security import requires_user_confirmation, format_confirmation_request
from celab_bot.users import get_linux_user, is_admin_user

try:
    from laya import Router
except ImportError:
    Router = None

AFFIRMATIVE_WORDS = {
    "si", "sí", "yes", "y", "s", "adelante", "procede", "proceder",
    "dale", "ok", "afirmativo", "confirmo", "confirmar", "hazlo",
    "ejecutar", "ejecútalo", "ejecutalo", "autorizo", "autorizado",
    "claro", "por supuesto", "de acuerdo", "va",
}

NEGATIVE_WORDS = {
    "no", "nop", "n", "cancelar", "cancela", "detener", "detén",
    "deten", "rechazar", "rechazo", "abortar", "abort", "para", "alto",
    "negativo", "denegar",
}

INTERNAL_TASK_MARKERS = (
    "### Task:",
    "Generate a concise",
    "Suggest 3-5 relevant follow-up",
    "Generate 1-3 broad tags",
    "### Chat History:",
)


def is_openwebui_background_task(text: str) -> bool:
    """Detecta si la petición es una tarea automática en segundo plano de OpenWebUI (títulos, tags, sugerencias)."""
    if not text or not isinstance(text, str):
        return False
    return any(marker in text for marker in INTERNAL_TASK_MARKERS)


def strip_ui_blocks(text: str) -> str:
    """Limpia bloques <details> y alertas de seguridad del historial antes de enviarlo al LLM."""
    if not text or not isinstance(text, str):
        return ""
    cleaned = re.sub(r"<details[\s\S]*?</details>", "", text)
    cleaned = re.sub(
        r"⚠️ \*\*Confirmación de Seguridad Requerida\*\*[\s\S]*?buscar una alternativa\.",
        "",
        cleaned,
    )
    cleaned = re.sub(r"(?:>\s*)?🚫\s*\*{0,2}Acción cancelada:\*{0,2}[^\n]*", "", cleaned)
    return cleaned.strip()


def check_confirmation_intent(text: str) -> Optional[bool]:
    """Retorna True si la respuesta es afirmativa, False si es negativa, o None si no corresponde."""
    clean = text.strip().lower()
    if not clean:
        return None
    tokens = clean.split()
    first_word = tokens[0].rstrip(".,!?:;")
    if clean in AFFIRMATIVE_WORDS or first_word in AFFIRMATIVE_WORDS:
        return True
    if clean in NEGATIVE_WORDS or first_word in NEGATIVE_WORDS:
        return False
    return None


def _looks_like_raw_tool_prefix(buf: str) -> bool:
    """Detecta si el texto inicial podría ser un <tool_call> o bloque JSON emitido en texto."""
    s = buf.lstrip()
    if not s:
        return True
    return s.startswith("<") or s.startswith("{") or s.startswith("```")


def _summarize_tool_signatures(tools: List[Dict[str, Any]]) -> List[str]:
    sigs = []
    for t in tools:
        fn = t.get("function", {})
        name = fn.get("name", "")
        props = list(fn.get("parameters", {}).get("properties", {}).keys())
        sigs.append(f"{name}({', '.join(props)})")
    return sigs


def _estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estima de forma conservadora el total de tokens para respetar ctx_size en llama-server."""
    total_chars = 0
    for m in messages:
        if isinstance(m, dict):
            total_chars += len(json.dumps(m, ensure_ascii=False))
        else:
            total_chars += len(str(m))
    return total_chars // 3 + len(messages) * 8 + 64


def _fit_context_budget(
    messages: List[Dict[str, Any]],
    ctx_size: int,
    desired_max_tokens: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Recorta historial intermedio más antiguo si excede ctx_size y ajusta max_tokens efectivos."""
    safety_margin = 512
    if not messages:
        effective = min(desired_max_tokens, max(512, ctx_size - safety_margin))
        return [], effective

    last_user_idx = len(messages) - 1
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role == "user":
            last_user_idx = i
            break

    system_head = [messages[0]] if last_user_idx > 0 else []
    middle = list(messages[1:last_user_idx]) if last_user_idx > 1 else []
    active_tail = list(messages[last_user_idx:])

    min_reply_reserve = min(desired_max_tokens, 2048)
    trimmed = system_head + middle + active_tail
    while middle and (_estimate_messages_tokens(trimmed) + min_reply_reserve + safety_margin > ctx_size):
        middle.pop(0)
        trimmed = system_head + middle + active_tail

    est_in = _estimate_messages_tokens(trimmed)
    effective_max_tokens = min(desired_max_tokens, max(512, ctx_size - est_in - safety_margin))
    return trimmed, effective_max_tokens


def _make_assistant_tool_msg(content_buf: str, normalized_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Construye un mensaje assistant limpio para llama.cpp usando JSON válido incluso si hubo truncado."""
    return {
        "role": "assistant",
        "content": content_buf or "",
        "tool_calls": [
            {
                "id": c["id"],
                "type": "function",
                "function": {
                    "name": c["name"],
                    "arguments": (
                        json.dumps(
                            {"_truncated_or_invalid": True, "raw_chars": len(c.get("raw_arguments") or "")},
                            ensure_ascii=False,
                        )
                        if c.get("parse_error")
                        else json.dumps(c["arguments"], ensure_ascii=False)
                    ),
                },
            }
            for c in normalized_calls
        ],
    }


def _tool_result_succeeded(tool_res: Any) -> bool:
    """Determina si el resultado de una herramienta fue exitoso."""
    if not isinstance(tool_res, dict):
        return True
    if "error" in tool_res:
        return False
    if "returncode" in tool_res and int(tool_res.get("returncode", 0)) != 0:
        return False
    return True


def create_app(config: Optional[CelabConfig] = None) -> FastAPI:

    config = config or load_config()

    app = FastAPI(
        title="CELab Bot API",
        version="2.1.0",
        description="OpenAI-compatible async server for OpenWebUI with Laya Routing, SQLite Memory and Multi-user Isolation",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    security = HTTPBearer(auto_error=False)

    # Cliente asíncrono para no bloquear el event loop de FastAPI
    llm_client = AsyncOpenAI(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        timeout=config.llm_timeout,
    )

    # Cargar routers Laya una sola vez en memoria
    print("⏳ [LAYA] Cargando modelos de Laya en memoria del servidor...")
    tool_router = None
    context_router = None

    if Router is not None:
        try:
            tool_router = Router(
                models={"english": config.tool_model_path},
                default="english",
                preload=True,
                device=config.laya_device,
            )
            context_router = Router(
                models={"english": config.context_model_path},
                default="english",
                preload=True,
                device=config.laya_device,
            )
            print("✅ [LAYA] Routers precargados exitosamente.")
        except Exception as e:
            print(f"⚠️ [LAYA] Error al precargar routers: {e}. Se cargará bajo demanda.")
    else:
        print("⚠️ [LAYA] Módulo laya no disponible en este entorno.")

    engine = AgentEngine(
        client=llm_client,
        config=config,
        tool_router=tool_router,
        context_router=context_router,
    )

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "model": config.llm_model,
            "backend": config.llm_base_url,
            "ctx_size": config.llm_ctx_size,
            "max_tokens": config.llm_max_tokens,
            "laya_loaded": tool_router is not None and context_router is not None,
        }

    @app.get("/v1/models")
    async def list_models():
        """Expone 'celab-bot' como modelo compatible con OpenWebUI."""
        return {
            "object": "list",
            "data": [
                {
                    "id": "celab-bot",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "celab",
                    "permission": [],
                    "root": "celab-bot",
                    "parent": None,
                    "context_length": config.llm_ctx_size,
                }
            ],
        }

    def sse_chunk(content: str, model: str = "celab-bot", finish_reason: Optional[str] = None) -> str:
        data = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content} if content else {},
                    "finish_reason": finish_reason,
                }
            ],
        }
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        auth: Optional[HTTPAuthorizationCredentials] = Security(security),
        x_openwebui_user_name: Optional[str] = Header(None, alias="X-OpenWebUI-User-Name"),
        x_openwebui_user_email: Optional[str] = Header(None, alias="X-OpenWebUI-User-Email"),
        x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
    ):
        # Validar API Key si está configurada
        if config.api_key and config.api_key != "no-key-needed":
            token = auth.credentials if auth else None
            if token != config.api_key:
                raise HTTPException(status_code=401, detail="API Key no válida o ausente.")

        body = await request.json()
        messages = body.get("messages", [])
        stream = body.get("stream", False)

        # Extraer última consulta del usuario
        current_query = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                raw_c = m.get("content", "")
                if isinstance(raw_c, list):
                    current_query = " ".join(
                        p.get("text", "") for p in raw_c if isinstance(p, dict) and p.get("type") == "text"
                    )
                else:
                    current_query = str(raw_c)
                break

        # 0. Peticiones automáticas en segundo plano de OpenWebUI (títulos, tags, sugerencias):
        # Responder directo, asíncrono y con límite corto sin tocar memoria ni confirmaciones.
        if is_openwebui_background_task(current_query):
            try:
                bg_resp = await llm_client.chat.completions.create(
                    model=config.llm_model,
                    messages=messages,
                    max_tokens=128,
                    temperature=0.2,
                    timeout=25.0,
                )
                bg_text = bg_resp.choices[0].message.content or ""
            except Exception:
                bg_text = "Chat"

            if stream:
                async def bg_stream():
                    yield sse_chunk(bg_text)
                    yield sse_chunk("", finish_reason="stop")
                    yield "data: [DONE]\n\n"
                return StreamingResponse(bg_stream(), media_type="text/event-stream")
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "celab-bot",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": bg_text},
                        "finish_reason": "stop",
                    }
                ],
            }

        # Detectar identidad de OpenWebUI probando todos los headers candidatos
        candidates = [
            c.strip()
            for c in (x_openwebui_user_name, x_openwebui_user_email, x_openwebui_user_id, body.get("user"))
            if isinstance(c, str) and c.strip()
        ]
        user_identity = candidates[0] if candidates else "default_user"
        linux_user = None
        for cand in candidates:
            linux_user = get_linux_user(cand, config.users_file)
            if linux_user:
                user_identity = cand
                break
        if not linux_user and not candidates:
            linux_user = get_linux_user("default_user", config.users_file)

        # Si el usuario NO está vinculado
        if not linux_user:
            unlinked_msg = (
                f"👋 **Hola.** Tu usuario de OpenWebUI (`{user_identity}`) aún no está vinculado "
                f"a ninguna cuenta Linux en el servidor del laboratorio.\n\n"
                f"Para activarte, pide a un administrador que ejecute en la terminal:\n"
                f"```bash\nsudo celab-bot link \"{user_identity}\" <tu_usuario_linux>\n```"
            )
            if stream:
                async def unlinked_stream():
                    yield sse_chunk(unlinked_msg)
                    yield sse_chunk("", finish_reason="stop")
                    yield "data: [DONE]\n\n"
                return StreamingResponse(unlinked_stream(), media_type="text/event-stream")
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "celab-bot",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": unlinked_msg},
                        "finish_reason": "stop",
                    }
                ],
            }

        is_admin = is_admin_user(linux_user)
        memory_store = UserMemoryStore(linux_user)

        # Gestión del ciclo de confirmación persistente en SQLite
        pending = memory_store.get_pending_confirmation(ttl_seconds=900)
        action_executed_info = None
        action_denied_info = None
        resumed_plan_state = None

        if pending:
            intent = check_confirmation_intent(current_query)
            if intent is True:
                memory_store.clear_pending_confirmation()
                tool_res = await asyncio.to_thread(
                    engine.execute_tool,
                    pending["tool_name"],
                    pending["args"],
                    linux_user,
                    is_admin,
                )
                resumed_plan_state = pending.get("plan_state")
                action_executed_info = {
                    "tool_name": pending["tool_name"],
                    "args": pending["args"],
                    "description": pending["description"],
                    "quest": pending["quest"],
                    "result": tool_res,
                }
            elif intent is False:
                memory_store.clear_pending_confirmation()
                action_denied_info = {
                    "tool_name": pending["tool_name"],
                    "description": pending["description"],
                    "quest": pending["quest"],
                    "plan_state": pending.get("plan_state"),
                }
            else:
                memory_store.clear_pending_confirmation()

        # Recuperar recuerdos persistentes relevantes de SQLite para inyectar al System Prompt
        memory_context = await engine.abuild_memory_context(current_query, memory_store)

        # Único System Prompt al inicio (regla estricta de plantilla Jinja en Bonsai-27b)
        system_content = (
            f"Eres CELab bot, el asistente inteligente del laboratorio. "
            f"Estás interactuando con el usuario Linux '{linux_user}'{' (ADMINISTRADOR)' if is_admin else ''}.\n"
            f"REGLAS DE HERRAMIENTAS:\n"
            f"1. Usa `search_tools` (con `query` EN INGLÉS) siempre que necesites interactuar con archivos, terminal, web o recetas.\n"
            f"2. Para crear o modificar archivos (`.py`, `.html`, `.css`, `.txt`, configs), usa SIEMPRE `edit_file` (nunca `cat << EOF` en `run_terminal_command`), escribiendo código limpio y conciso sin comentarios decorativos extensos. Usa `run_terminal_command` solo para comandos shell (`ps`, `df`, `git`, ejecutar scripts).\n"
            f"3. Una vez que `search_tools` desbloquee las herramientas, INVOCA la herramienta adecuada directamente mediante tool call en lugar de solo mostrar el código en texto.\n"
            f"4. Para conversaciones generales, saludos o dudas teóricas que no requieran el sistema, responde directamente en español."
            f"{memory_context}"
        )

        # Limpiar bloques de interfaz (<details> y carteles de confirmación) y normalizar roles
        cleaned_messages: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                continue  # El único system prompt lo colocamos en posición 0
            m_copy: Dict[str, Any] = {"role": role, "content": m.get("content", "")}
            if isinstance(m_copy["content"], list):
                m_copy["content"] = " ".join(
                    p.get("text", "") for p in m_copy["content"] if isinstance(p, dict) and p.get("type") == "text"
                )
            if role == "assistant" and isinstance(m_copy["content"], str):
                c_clean = strip_ui_blocks(m_copy["content"])
                m_copy["content"] = c_clean if c_clean else "(Esperando respuesta del usuario)"
            cleaned_messages.append(m_copy)

        prepared_messages: List[Dict[str, Any]] = [{"role": "system", "content": system_content}] + cleaned_messages

        # Inyectar contexto de confirmación dentro del último mensaje user (respetando Jinja single-system)
        if action_executed_info:
            exec_note = (
                f"\n\n[ACCIÓN AUTORIZADA Y EJECUTADA DIRECTAMENTE POR EL HARNESS]:\n"
                f"El usuario autorizó la acción '{action_executed_info['tool_name']}' ({action_executed_info['description']}) "
                f"para la tarea original: '{action_executed_info['quest']}'.\n"
                f"Resultado de la ejecución:\n{json.dumps(action_executed_info['result'], ensure_ascii=False)}\n\n"
                f"INSTRUCCIÓN: La acción YA FUE EJECUTADA CON ÉXITO. No llames a search_tools ni vuelvas a ejecutar el comando. "
                f"Simplemente confirma al usuario en español que la acción se completó y muéstrale el resultado."
            )
            if prepared_messages and prepared_messages[-1].get("role") == "user":
                prepared_messages[-1] = dict(prepared_messages[-1])
                prepared_messages[-1]["content"] = f"{prepared_messages[-1].get('content', '')}{exec_note}"
            else:
                prepared_messages.append({"role": "user", "content": exec_note})
        elif action_denied_info:
            plan_ctx_note = ""
            denied_plan = action_denied_info.get("plan_state")
            if isinstance(denied_plan, dict):
                steps = denied_plan.get("steps", [])
                s_idx = int(denied_plan.get("current_step_idx", 0))
                rem = [f"{s.get('id')}. {s.get('goal')}" for s in steps[s_idx + 1 :]]
                rem_txt = "\n".join(rem) if rem else "(ninguno)"
                plan_ctx_note = (
                    f"\nEsta acción correspondía al Paso {s_idx + 1}/{len(steps)} del plan.\n"
                    f"Pasos restantes sin ejecutar:\n{rem_txt}\n"
                )
            denied_note = (
                f"\n\n[PERMISO DENEGADO POR EL USUARIO]:\n"
                f"El usuario DENEGÓ el permiso para ejecutar '{action_denied_info['tool_name']}' ({action_denied_info['description']}).\n"
                f"La tarea original que intentabas resolver era: '{action_denied_info['quest']}'.\n"
                f"{plan_ctx_note}\n"
                f"INSTRUCCIÓN: Con esta negativa, NO intentes ejecutar esta misma acción otra vez. "
                f"Evalúa si puedes resolver la tarea con una alternativa viable que no requiera esa acción destructiva, "
                f"o explica al usuario con amabilidad qué se requería y pregúntale cómo desea proceder."
            )
            if prepared_messages and prepared_messages[-1].get("role") == "user":
                prepared_messages[-1] = dict(prepared_messages[-1])
                prepared_messages[-1]["content"] = f"{prepared_messages[-1].get('content', '')}{denied_note}"
            else:
                prepared_messages.append({"role": "user", "content": denied_note})

        # Filtrar contexto con Laya (asíncrono, usando caché de SQLite)
        if action_executed_info or action_denied_info:
            filtered_msgs = prepared_messages
        else:
            filtered_msgs = await engine.aget_filtered_messages(prepared_messages, current_query, memory_store)

        # Limpiar claves internas (_content_en) antes de mandar al LLM
        def sanitize_for_llm(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            clean = []
            for item in msgs:
                if isinstance(item, dict):
                    clean.append({k: v for k, v in item.items() if not k.startswith("_")})
                else:
                    clean.append(item)
            return clean

        async def _stream_llm_turn(
            conv_msgs: List[Dict[str, Any]],
            tools: Optional[List[Dict[str, Any]]],
            stream_text_live: bool = True,
            step_label: str = "direct",
        ) -> AsyncGenerator[Tuple[str, Any], None]:
            content_buf = ""
            streamed_live = False
            aborted_loop = False
            finish_reason = None
            tool_calls_acc: Dict[int, Dict[str, Any]] = {}

            budgeted_messages, effective_max_tokens = _fit_context_budget(
                conv_msgs,
                config.llm_ctx_size,
                config.llm_max_tokens,
            )

            llm_kwargs: Dict[str, Any] = {
                "model": config.llm_model,
                "messages": budgeted_messages,
                "max_tokens": effective_max_tokens,
                "temperature": config.llm_temperature,
                "stream": True,
                "extra_body": {"repeat_penalty": config.llm_repeat_penalty},
            }
            if tools:
                llm_kwargs["tools"] = tools
                llm_kwargs["tool_choice"] = "auto"

            try:
                llm_stream = await llm_client.chat.completions.create(**llm_kwargs)
                async for chunk in llm_stream:
                    if await request.is_disconnected():
                        await llm_stream.close()
                        yield ("disconnected", None)
                        return

                    if not chunk.choices:
                        continue
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason
                    delta = chunk.choices[0].delta

                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index if tc_delta.index is not None else 0
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {
                                    "id": tc_delta.id or f"call_{uuid.uuid4().hex[:10]}",
                                    "name": "",
                                    "arguments": "",
                                }
                            if tc_delta.id:
                                tool_calls_acc[idx]["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    tool_calls_acc[idx]["name"] += tc_delta.function.name
                                if tc_delta.function.arguments:
                                    tool_calls_acc[idx]["arguments"] += tc_delta.function.arguments
                                    if detect_repetition_loop(tool_calls_acc[idx]["arguments"]):
                                        aborted_loop = True
                                        print(f"⚠️ [BONSAI GUARD] Bucle de repetición en argumentos ({step_label}). Abortando stream.")
                                        await llm_stream.close()
                                        break
                        if aborted_loop:
                            break

                    if delta.content:
                        content_buf += delta.content
                        if detect_repetition_loop(content_buf):
                            aborted_loop = True
                            print(f"⚠️ [BONSAI GUARD] Bucle de repetición en texto ({step_label}). Abortando stream.")
                            await llm_stream.close()
                            break

                        if stream_text_live and not tool_calls_acc:
                            if streamed_live:
                                yield ("sse", sse_chunk(delta.content))
                            elif len(content_buf) >= 24 and not _looks_like_raw_tool_prefix(content_buf):
                                streamed_live = True
                                yield ("sse", sse_chunk(content_buf))

            except Exception as err:
                yield ("error", str(err))
                return

            normalized_calls: List[Dict[str, Any]] = []
            for idx in sorted(tool_calls_acc.keys()):
                tc_item = tool_calls_acc[idx]
                fn_name = tc_item["name"].strip()
                raw_args = tc_item["arguments"]
                parse_error: Optional[str] = None
                parsed_args: Dict[str, Any] = {}

                if not raw_args or not raw_args.strip():
                    parse_error = (
                        f"Missing or empty JSON arguments for '{fn_name}' "
                        f"(finish_reason={finish_reason})."
                    )
                else:
                    try:
                        loaded = json.loads(raw_args)
                        if isinstance(loaded, dict):
                            parsed_args = loaded
                        else:
                            parse_error = f"Tool arguments for '{fn_name}' must be a JSON object."
                    except Exception as je:
                        trunc_note = " (truncated at max_tokens limit)" if finish_reason == "length" else ""
                        parse_error = (
                            f"Malformed/truncated JSON arguments for '{fn_name}'{trunc_note}: {je}."
                        )

                if not parse_error:
                    if fn_name == "run_terminal_command" and not (parsed_args.get("command") or "").strip():
                        parse_error = (
                            "Missing or empty 'command' argument (likely truncated). "
                            "If creating a file, use edit_file instead of run_terminal_command."
                        )
                    elif fn_name == "edit_file" and not (parsed_args.get("path") or "").strip():
                        parse_error = "Missing or empty 'path' argument (likely truncated)."

                if fn_name:
                    normalized_calls.append({
                        "id": tc_item["id"],
                        "name": fn_name,
                        "arguments": parsed_args,
                        "raw_arguments": raw_args,
                        "parse_error": parse_error,
                    })

            if not normalized_calls and not streamed_live and tools:
                fallback_calls = extract_fallback_tool_calls(content_buf, tools)
                if fallback_calls:
                    print(f"🩹 [FALLBACK PARSER] Rescatadas {len(fallback_calls)} tool calls desde texto plano ({step_label}).")
                    normalized_calls = fallback_calls
                    content_buf = ""

            yield ("done", {
                "content": content_buf,
                "tool_calls": normalized_calls,
                "streamed_live": streamed_live,
                "aborted_loop": aborted_loop,
                "finish_reason": finish_reason,
            })

        async def _discover_tools(
            query: str,
            core_tools: List[Dict[str, Any]],
            discovered_tools: List[Dict[str, Any]],
        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
            found_t, found_r = await engine.asearch_tools(query, linux_user, memory_store)
            existing_names = {t["function"]["name"] for t in core_tools + discovered_tools}
            for ft in found_t:
                t_name = ft["function"]["name"]
                if t_name not in existing_names:
                    discovered_tools.append(ft)
                    existing_names.add(t_name)
            if config.always_include_terminal_tool and "run_terminal_command" not in existing_names:
                discovered_tools.append(engine.tools_registry.get_terminal_tool())
                existing_names.add("run_terminal_command")
            sigs = _summarize_tool_signatures(discovered_tools)
            return list(core_tools) + discovered_tools, found_r, sigs

        async def response_generator() -> AsyncGenerator[str, None]:
            core_tools = engine.tools_registry.get_core_tools(include_terminal=False)
            active_plan_state: Optional[Dict[str, Any]] = None
            step_recovery_note = ""

            if action_executed_info:
                t_name = action_executed_info["tool_name"]
                t_desc = action_executed_info["description"]
                t_res = action_executed_info["result"]
                res_str = json.dumps(t_res, ensure_ascii=False, indent=2)
                preview = (res_str[:400] + "...") if len(res_str) > 400 else res_str
                yield sse_chunk(
                    f"\n<details open><summary>✅ <b>Acción autorizada y ejecutada:</b> <code>{t_name}</code></summary>\n\n"
                    f"**Operación:** {t_desc}\n\n"
                    f"```json\n{preview}\n```\n"
                    f"</details>\n\n"
                )
                if isinstance(resumed_plan_state, dict):
                    active_plan_state = resumed_plan_state
                    s_idx = int(active_plan_state.get("current_step_idx", 0))
                    steps = active_plan_state.get("steps", [])
                    if 0 <= s_idx < len(steps):
                        cur_step = steps[s_idx]
                        if _tool_result_succeeded(t_res):
                            summary = summarize_step_outcome(
                                cur_step.get("goal", ""),
                                "",
                                [{"tool": t_name, "result": t_res}],
                            )
                            active_plan_state.setdefault("completed_steps", []).append({
                                "id": cur_step.get("id", s_idx + 1),
                                "goal": cur_step.get("goal", ""),
                                "summary": summary,
                            })
                            active_plan_state["current_step_idx"] = s_idx + 1
                            yield sse_chunk(
                                f"\n<details><summary>✅ <b>Paso {s_idx + 1}/{len(steps)} completado:</b> {cur_step.get('goal', '')}</summary>\n\n"
                                f"{summary}\n</details>\n\n"
                            )
                        else:
                            step_recovery_note = (
                                f"La herramienta autorizada '{t_name}' devolvió un error o código distinto de 0:\n"
                                f"{json.dumps(t_res, ensure_ascii=False)[:1000]}\n"
                                f"Diagnostica o corrige este paso antes de continuar."
                            )
            elif action_denied_info:
                t_desc = action_denied_info["description"]
                yield sse_chunk(f"\n> 🚫 **Acción cancelada:** No se ejecutó {t_desc}.\n\n")
            elif config.enable_planner and not should_skip_planner(current_query):
                plan_result = await agenerate_plan(
                    llm_client,
                    config,
                    current_query,
                    cleaned_messages,
                    memory_context,
                )
                if plan_result.get("mode") == "plan" and len(plan_result.get("steps", [])) >= 2:
                    active_plan_state = {
                        "original_quest": current_query,
                        "steps": plan_result["steps"],
                        "current_step_idx": 0,
                        "completed_steps": [],
                    }
                    yield sse_chunk(format_plan_markdown(active_plan_state))

            # ─────────────────────────────────────────────────────────────────
            # MODO PLAN-AND-EXECUTE (CONTEXTO MÍNIMO AISLADO POR PASO)
            # ─────────────────────────────────────────────────────────────────
            if active_plan_state is not None:
                steps = active_plan_state.get("steps", [])
                total_steps = len(steps)

                while active_plan_state["current_step_idx"] < total_steps:
                    s_idx = active_plan_state["current_step_idx"]
                    cur_step = steps[s_idx]
                    step_goal = cur_step.get("goal", "")
                    step_label = f"paso {s_idx + 1}/{total_steps}"

                    step_messages = build_step_messages(
                        linux_user,
                        is_admin,
                        memory_context,
                        active_plan_state,
                        s_idx,
                        recovery_note=step_recovery_note,
                    )
                    step_recovery_note = ""

                    step_discovered: List[Dict[str, Any]] = []
                    step_active_tools: Optional[List[Dict[str, Any]]] = None
                    if cur_step.get("needs_tools", True):
                        t_query = cur_step.get("tool_query", "")
                        if t_query:
                            step_active_tools, _, _ = await _discover_tools(
                                t_query, core_tools, step_discovered
                            )
                        else:
                            if config.always_include_terminal_tool:
                                step_discovered.append(engine.tools_registry.get_terminal_tool())
                            step_active_tools = list(core_tools) + step_discovered

                    step_tool_results: List[Dict[str, Any]] = []
                    step_final_text = ""

                    for _sub_idx in range(config.max_substeps_per_plan_step):
                        turn_done = None
                        disconnected = False
                        turn_err = None
                        async for ev_type, ev_val in _stream_llm_turn(
                            step_messages,
                            step_active_tools,
                            stream_text_live=False,
                            step_label=step_label,
                        ):
                            if ev_type == "disconnected":
                                disconnected = True
                                break
                            elif ev_type == "error":
                                turn_err = ev_val
                                break
                            elif ev_type == "done":
                                turn_done = ev_val

                        if disconnected:
                            return
                        if turn_err is not None:
                            yield sse_chunk(f"\n❌ Error al comunicarse con el modelo local ({step_label}): {turn_err}\n")
                            yield sse_chunk("", finish_reason="stop")
                            yield "data: [DONE]\n\n"
                            return

                        normalized_calls = turn_done["tool_calls"]
                        if not normalized_calls:
                            step_final_text = turn_done["content"]
                            break

                        step_messages.append(_make_assistant_tool_msg(turn_done["content"], normalized_calls))

                        for call in normalized_calls:
                            fn_name = call["name"]
                            args = call["arguments"]
                            tc_id = call["id"]

                            yield sse_chunk(f"\n<details><summary>🛠️ <i>[{step_label}] Herramienta: <code>{fn_name}</code></i></summary>\n\n")

                            if call.get("parse_error"):
                                tool_res = {
                                    "error": call["parse_error"],
                                    "hint": "Your previous tool call JSON was too long and got cut off. Retry edit_file NOW with a much more concise version of the file (avoid long comments and redundant CSS), or create a compact skeleton first.",
                                }
                            elif fn_name == "search_tools":
                                query = args.get("query", "")
                                step_active_tools, found_r, sigs = await _discover_tools(
                                    query, core_tools, step_discovered
                                )
                                guidance = (
                                    f"Tools unlocked and ready to call now: {', '.join(sigs)}. CALL the appropriate tool now."
                                    if (step_discovered or found_r)
                                    else "No specialized tool matched. Answer directly or refine your English search query."
                                )
                                tool_res = {
                                    "found_tools": [t["function"]["name"] for t in step_discovered],
                                    "signatures": sigs,
                                    "found_recipes": list(found_r.keys()),
                                    "message": guidance,
                                }
                            else:
                                req_confirm, desc = requires_user_confirmation(fn_name, args)
                                if req_confirm and config.require_confirmation:
                                    memory_store.set_pending_confirmation(
                                        tool_name=fn_name,
                                        args=args,
                                        description=desc,
                                        quest=f"Paso {s_idx + 1}/{total_steps}: {step_goal}",
                                        plan_state=active_plan_state,
                                    )
                                    confirm_text = format_confirmation_request(
                                        tc_id,
                                        fn_name,
                                        desc,
                                        f"Paso {s_idx + 1}/{total_steps}: {step_goal}",
                                    )
                                    yield sse_chunk(f"⏳ <i>Esperando autorización del usuario...</i>\n</details>\n\n{confirm_text}\n\n")
                                    yield sse_chunk("", finish_reason="stop")
                                    yield "data: [DONE]\n\n"
                                    return

                                tool_res = await asyncio.to_thread(
                                    engine.execute_tool, fn_name, args, linux_user, is_admin
                                )
                                step_tool_results.append({"tool": fn_name, "result": tool_res})

                            res_str = json.dumps(tool_res, ensure_ascii=False, indent=2)
                            preview = (res_str[:400] + "...") if len(res_str) > 400 else res_str
                            yield sse_chunk(f"```json\n{preview}\n```\n</details>\n\n")

                            step_messages.append({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": json.dumps(tool_res, ensure_ascii=False),
                            })

                    summary = summarize_step_outcome(step_goal, step_final_text, step_tool_results)
                    active_plan_state.setdefault("completed_steps", []).append({
                        "id": cur_step.get("id", s_idx + 1),
                        "goal": step_goal,
                        "summary": summary,
                    })
                    active_plan_state["current_step_idx"] = s_idx + 1
                    yield sse_chunk(
                        f"\n<details><summary>✅ <b>Paso {s_idx + 1}/{total_steps} completado:</b> {step_goal}</summary>\n\n"
                        f"{summary}\n</details>\n\n"
                    )

                # Fase 3 — Synthesizer
                synth_msgs = build_synthesizer_messages(
                    linux_user, is_admin, memory_context, active_plan_state
                )
                synth_done = None
                async for ev_type, ev_val in _stream_llm_turn(
                    synth_msgs,
                    tools=None,
                    stream_text_live=True,
                    step_label="synthesizer",
                ):
                    if ev_type == "sse":
                        yield ev_val
                    elif ev_type == "disconnected":
                        return
                    elif ev_type == "error":
                        yield sse_chunk(f"\n❌ Error al sintetizar respuesta final: {ev_val}\n")
                        yield sse_chunk("", finish_reason="stop")
                        yield "data: [DONE]\n\n"
                        return
                    elif ev_type == "done":
                        synth_done = ev_val

                if synth_done:
                    if not synth_done["streamed_live"] and synth_done["content"]:
                        yield sse_chunk(synth_done["content"])
                    if synth_done["aborted_loop"]:
                        yield sse_chunk("\n\n*(Respuesta truncada automáticamente al detectar repetición del modelo local.)*")
                    elif synth_done["finish_reason"] == "length":
                        yield sse_chunk("\n\n*(Salida alcanzó el límite de tokens por turno; pídeme continuar si faltó una parte.)*")
                    final_synth_text = synth_done["content"]
                    yield sse_chunk("", finish_reason="stop")
                    yield "data: [DONE]\n\n"

                    if final_synth_text.strip():
                        asyncio.create_task(
                            engine.aclassify_and_store_turn(
                                active_plan_state.get("original_quest", current_query),
                                final_synth_text,
                                memory_store,
                            )
                        )
                return

            # ─────────────────────────────────────────────────────────────────
            # MODO DIRECTO (SIN PLAN MULTI-PASO)
            # ─────────────────────────────────────────────────────────────────
            discovered_tools: List[Dict[str, Any]] = []
            active_tools: Optional[List[Dict[str, Any]]] = None if action_executed_info else list(core_tools)
            conversation_history: List[Dict[str, Any]] = sanitize_for_llm(list(filtered_msgs))
            final_assistant_text = ""

            for step_idx in range(config.max_agent_steps):
                turn_done = None
                disconnected = False
                turn_err = None
                async for ev_type, ev_val in _stream_llm_turn(
                    conversation_history,
                    active_tools,
                    stream_text_live=True,
                    step_label=f"direct-{step_idx + 1}",
                ):
                    if ev_type == "sse":
                        yield ev_val
                    elif ev_type == "disconnected":
                        disconnected = True
                        break
                    elif ev_type == "error":
                        turn_err = ev_val
                        break
                    elif ev_type == "done":
                        turn_done = ev_val

                if disconnected:
                    return
                if turn_err is not None:
                    yield sse_chunk(f"\n❌ Error al comunicarse con el modelo local: {turn_err}\n")
                    yield sse_chunk("", finish_reason="stop")
                    yield "data: [DONE]\n\n"
                    return

                normalized_calls = turn_done["tool_calls"]
                content_buf = turn_done["content"]
                streamed_live = turn_done["streamed_live"]
                aborted_loop = turn_done["aborted_loop"]
                finish_reason = turn_done["finish_reason"]

                if not normalized_calls:
                    if not streamed_live and content_buf:
                        yield sse_chunk(content_buf)
                    if aborted_loop:
                        yield sse_chunk("\n\n*(Respuesta truncada automáticamente al detectar repetición del modelo local.)*")
                    elif finish_reason == "length":
                        yield sse_chunk("\n\n*(Salida alcanzó el límite de tokens por turno; pídeme continuar si faltó una parte.)*")
                    final_assistant_text = content_buf
                    yield sse_chunk("", finish_reason="stop")
                    yield "data: [DONE]\n\n"

                    if not action_executed_info and not action_denied_info and final_assistant_text.strip():
                        asyncio.create_task(
                            engine.aclassify_and_store_turn(current_query, final_assistant_text, memory_store)
                        )
                    return

                conversation_history.append(_make_assistant_tool_msg(content_buf, normalized_calls))

                for call in normalized_calls:
                    fn_name = call["name"]
                    args = call["arguments"]
                    tc_id = call["id"]

                    yield sse_chunk(f"\n<details><summary>🛠️ <i>Llamando herramienta: <code>{fn_name}</code></i></summary>\n\n")

                    if call.get("parse_error"):
                        tool_res = {
                            "error": call["parse_error"],
                            "hint": "Your previous tool call JSON was too long and got cut off. Retry edit_file NOW with a much more concise version of the file (avoid long comments and redundant CSS), or create a compact skeleton first.",
                        }
                    elif fn_name == "search_tools":
                        query = args.get("query", "")
                        active_tools, found_r, sigs = await _discover_tools(
                            query, core_tools, discovered_tools
                        )
                        if discovered_tools or found_r:
                            guidance = (
                                f"Tools unlocked and ready to call now: {', '.join(sigs)}. "
                                f"CALL the appropriate tool now via a tool call to complete the user request."
                            )
                        else:
                            guidance = (
                                "No specialized tool matched. Answer directly or refine your English search query."
                            )

                        tool_res = {
                            "found_tools": [t["function"]["name"] for t in discovered_tools],
                            "signatures": sigs,
                            "found_recipes": list(found_r.keys()),
                            "message": guidance,
                        }
                    else:
                        req_confirm, desc = requires_user_confirmation(fn_name, args)
                        if req_confirm and config.require_confirmation:
                            memory_store.set_pending_confirmation(
                                tool_name=fn_name,
                                args=args,
                                description=desc,
                                quest=current_query,
                            )
                            confirm_text = format_confirmation_request(tc_id, fn_name, desc, current_query)
                            yield sse_chunk(f"⏳ <i>Esperando autorización del usuario...</i>\n</details>\n\n{confirm_text}\n\n")
                            yield sse_chunk("", finish_reason="stop")
                            yield "data: [DONE]\n\n"
                            return

                        tool_res = await asyncio.to_thread(
                            engine.execute_tool, fn_name, args, linux_user, is_admin
                        )

                    res_str = json.dumps(tool_res, ensure_ascii=False, indent=2)
                    preview = (res_str[:400] + "...") if len(res_str) > 400 else res_str
                    yield sse_chunk(f"```json\n{preview}\n```\n</details>\n\n")

                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps(tool_res, ensure_ascii=False),
                    })

            yield sse_chunk(
                f"\n⚠️ Se alcanzó el límite máximo de pasos ({config.max_agent_steps}) para esta consulta. "
                f"Indícame si deseas que continúe con un paso específico."
            )
            yield sse_chunk("", finish_reason="stop")
            yield "data: [DONE]\n\n"

        if stream:
            return StreamingResponse(response_generator(), media_type="text/event-stream")

        full_content = ""
        async for chunk in response_generator():
            if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                try:
                    parsed = json.loads(chunk[6:].strip())
                    c = parsed["choices"][0]["delta"].get("content", "")
                    full_content += c
                except Exception:
                    pass

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "celab-bot",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_content},
                    "finish_reason": "stop",
                }
            ],
        }

    return app
