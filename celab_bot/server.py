import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request, Response, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from openai import OpenAI

from celab_bot.config import CelabConfig, load_config
from celab_bot.core import AgentEngine
from celab_bot.memory import UserMemoryStore
from celab_bot.security import requires_user_confirmation, format_confirmation_request
from celab_bot.users import get_linux_user, is_admin_user

# Mapeo en memoria de confirmaciones pendientes de usuario: { linux_user: dict }
PENDING_CONFIRMATIONS: Dict[str, Dict[str, Any]] = {}

AFFIRMATIVE_WORDS = {
    "si", "sí", "yes", "y", "s", "adelante", "procede", "proceder",
    "dale", "ok", "afirmativo", "confirmo", "confirmar", "hazlo",
    "ejecutar", "ejecútalo", "ejecutalo", "autorizo", "autorizado",
    "claro", "por supuesto", "de acuerdo", "va"
}

NEGATIVE_WORDS = {
    "no", "nop", "n", "cancelar", "cancela", "detener", "detén",
    "deten", "rechazar", "rechazo", "abortar", "abort", "para", "alto",
    "negativo", "denegar"
}


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


def create_app(config: Optional[CelabConfig] = None) -> FastAPI:

    config = config or load_config()

    app = FastAPI(
        title="CELab Bot API",
        version="2.0.0",
        description="OpenAI-compatible server for OpenWebUI with Laya Routing and Multi-user Isolation",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    security = HTTPBearer(auto_error=False)

    # Inicializar OpenAI client para el LLM local
    llm_client = OpenAI(
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

        # Detectar identidad de OpenWebUI
        user_identity = x_openwebui_user_name or x_openwebui_user_email or x_openwebui_user_id
        if not user_identity and messages:
            # Fallback en caso de prueba local o sin headers
            user_identity = "default_user"

        linux_user = get_linux_user(user_identity, config.users_file)

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
            else:
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

        # Extraer última consulta del usuario
        current_query = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                current_query = m.get("content", "")
                break

        # Gestión del ciclo de confirmación en el harness
        pending = PENDING_CONFIRMATIONS.get(linux_user)
        action_executed_info = None
        action_denied_info = None

        if pending:
            # Expirar confirmaciones pendientes tras 15 minutos (900 seg)
            if time.time() - pending.get("timestamp", 0) > 900:
                PENDING_CONFIRMATIONS.pop(linux_user, None)
                pending = None

        if pending:
            intent = check_confirmation_intent(current_query)
            if intent is True:
                # El usuario AUTORIZÓ la acción: el harness la ejecuta de inmediato
                PENDING_CONFIRMATIONS.pop(linux_user, None)
                tool_res = engine.execute_tool(
                    pending["tool_name"],
                    pending["args"],
                    linux_user,
                    is_admin,
                )
                action_executed_info = {
                    "tool_name": pending["tool_name"],
                    "args": pending["args"],
                    "description": pending["description"],
                    "quest": pending["quest"],
                    "result": tool_res,
                }
            elif intent is False:
                # El usuario DENEGÓ la acción
                PENDING_CONFIRMATIONS.pop(linux_user, None)
                action_denied_info = {
                    "tool_name": pending["tool_name"],
                    "description": pending["description"],
                    "quest": pending["quest"],
                }
            else:
                # Mensaje no relacionado con la confirmación: descartar acción pendiente previa
                PENDING_CONFIRMATIONS.pop(linux_user, None)

        # Iniciar System Prompt personalizado para CELab bot
        system_content = (
            f"Eres CELab bot, el asistente inteligente del laboratorio. "
            f"Estás interactuando con el usuario Linux '{linux_user}'{' (ADMINISTRADOR)' if is_admin else ''}. "
            f"Tienes acceso a la herramienta search_tools para descubrir habilidades cuando sea necesario. "
            f"Usa search_tools con una descripción EN INGLÉS ÚNICAMENTE cuando la tarea requiera "
            f"interactuar con el sistema (archivos, terminal, recetas, clima o herramientas externas). "
            f"Para conversaciones generales, saludos, dudas de física o preguntas personales, responde directamente."
        )

        # Asegurar mensaje de sistema al inicio
        prepared_messages = list(messages)
        if not prepared_messages or prepared_messages[0].get("role") != "system":
            prepared_messages.insert(0, {"role": "system", "content": system_content})
        else:
            prepared_messages[0]["content"] = system_content

        # Inyectar contexto al modelo según la decisión del usuario
        if action_executed_info:
            prepared_messages.append({
                "role": "system",
                "content": (
                    f"[ACCIÓN AUTORIZADA Y EJECUTADA DIRECTAMENTE POR EL HARNESS]:\n"
                    f"El usuario autorizó la acción '{action_executed_info['tool_name']}' ({action_executed_info['description']}) "
                    f"para la tarea original: '{action_executed_info['quest']}'.\n"
                    f"Resultado de la ejecución:\n{json.dumps(action_executed_info['result'], ensure_ascii=False)}\n\n"
                    f"INSTRUCCIÓN: La acción ya se ejecutó con éxito. NO vuelvas a solicitar confirmación ni a llamar a esta misma herramienta. "
                    f"Informa al usuario sobre el resultado obtenido y continúa con los pasos pendientes de la tarea si los hay."
                )
            })
        elif action_denied_info:
            prepared_messages.append({
                "role": "system",
                "content": (
                    f"[PERMISO DENEGADO POR EL USUARIO]:\n"
                    f"El usuario DENEGÓ el permiso para ejecutar '{action_denied_info['tool_name']}' ({action_denied_info['description']}).\n"
                    f"La tarea original que intentabas resolver era: '{action_denied_info['quest']}'.\n\n"
                    f"INSTRUCCIÓN: Con esta negativa, NO intentes ejecutar esta misma acción otra vez. "
                    f"Evalúa si puedes resolver la tarea con una alternativa viable que no requiera esa acción destructiva, "
                    f"o explica al usuario con amabilidad qué se requería y pregúntale cómo desea proceder."
                )
            })

        # Filtrar contexto con Laya (si hubo autorización/denegación, mantener prepared_messages intacto)
        if action_executed_info or action_denied_info:
            filtered_msgs = prepared_messages
        else:
            filtered_msgs = engine.get_filtered_messages(prepared_messages, current_query, memory_store)

        # Iniciar ciclo de respuesta agéntica con streaming
        async def response_generator() -> AsyncGenerator[str, None]:
            discovered_tools = []
            active_tools = engine.tools_registry.get_core_tools()
            conversation_history = list(filtered_msgs)

            # Si acabamos de ejecutar una acción autorizada, mostrar su bloque colapsable de inmediato
            if action_executed_info:
                t_name = action_executed_info["tool_name"]
                t_desc = action_executed_info["description"]
                t_res = action_executed_info["result"]
                res_str = json.dumps(t_res, ensure_ascii=False, indent=2)
                preview = (res_str[:300] + "...") if len(res_str) > 300 else res_str
                yield sse_chunk(
                    f"\n<details open><summary>✅ <b>Acción autorizada por el usuario:</b> <code>{t_name}</code></summary>\n\n"
                    f"**Operación ejecutada:** `{t_desc}`\n\n"
                    f"```json\n{preview}\n```\n"
                    f"</details>\n\n"
                )
            elif action_denied_info:
                t_desc = action_denied_info["description"]
                yield sse_chunk(
                    f"\n> 🚫 **Acción cancelada:** No se ejecutó `{t_desc}`.\n\n"
                )

            while True:
                # 1. Llamar al LLM
                try:
                    chat_resp = llm_client.chat.completions.create(
                        model=config.llm_model,
                        messages=conversation_history,
                        tools=active_tools if active_tools else None,
                    )
                except Exception as err:
                    yield sse_chunk(f"\n❌ Error al comunicarse con el modelo local: {err}\n")
                    yield sse_chunk("", finish_reason="stop")
                    yield "data: [DONE]\n\n"
                    return

                msg = chat_resp.choices[0].message
                content = msg.content or ""
                tool_calls = msg.tool_calls or []

                conversation_history.append(msg)

                # Si NO hay llamadas a tools, emitir respuesta final
                if not tool_calls:
                    yield sse_chunk(content)
                    yield sse_chunk("", finish_reason="stop")
                    yield "data: [DONE]\n\n"
                    break

                # Si HAY tool calls, procesarlas
                for tc in tool_calls:
                    fn_name = tc.function.name
                    raw_args = tc.function.arguments
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        args = {}

                    # Emitir bloque colapsable en Markdown para OpenWebUI
                    yield sse_chunk(f"\n<details><summary>🛠️ <i>Llamando herramienta: <code>{fn_name}</code></i></summary>\n\n")

                    # Caso especial: search_tools
                    if fn_name == "search_tools":
                        query = args.get("query", "")
                        found_t, found_r = engine.search_tools(query, linux_user)
                        for ft in found_t:
                            if ft not in discovered_tools:
                                discovered_tools.append(ft)
                        active_tools = engine.tools_registry.get_core_tools() + discovered_tools

                        tool_res = {
                            "found_tools": [t["function"]["name"] for t in found_t],
                            "found_recipes": list(found_r.keys()),
                            "message": f"Se descubrieron {len(found_t)} herramientas y {len(found_r)} recetas."
                        }
                    else:
                        # Verificar confirmación para herramientas destructivas
                        req_confirm, desc = requires_user_confirmation(fn_name, args)
                        if req_confirm and config.require_confirmation:
                            PENDING_CONFIRMATIONS[linux_user] = {
                                "tool_name": fn_name,
                                "args": args,
                                "description": desc,
                                "quest": current_query,
                                "timestamp": time.time(),
                            }
                            confirm_text = format_confirmation_request(tc.id, fn_name, desc, current_query)
                            # Cerrar el bloque de detalles para que la solicitud quede visible afuera
                            yield sse_chunk(f"⏳ <i>Esperando autorización del usuario...</i>\n</details>\n\n{confirm_text}\n\n")
                            yield sse_chunk("", finish_reason="stop")
                            yield "data: [DONE]\n\n"
                            return

                        # Ejecutar herramienta bajo la identidad de linux_user
                        tool_res = engine.execute_tool(fn_name, args, linux_user, is_admin)

                    # Mostrar resultado dentro del bloque details
                    res_str = json.dumps(tool_res, ensure_ascii=False, indent=2)
                    preview = (res_str[:300] + "...") if len(res_str) > 300 else res_str
                    yield sse_chunk(f"```json\n{preview}\n```\n</details>\n\n")

                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_res, ensure_ascii=False)
                    })


        if stream:
            return StreamingResponse(response_generator(), media_type="text/event-stream")
        else:
            # Versión no-streaming consolidada
            # Consumir generador
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
