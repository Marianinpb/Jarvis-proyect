import json
import re
from typing import Any, Dict, List, Optional


_SKIP_EXACT = {
    "hola", "buenas", "buenos dias", "buenos días", "buenas tardes", "buenas noches",
    "hey", "hi", "hello", "que tal", "qué tal", "como estas", "cómo estás",
    "gracias", "muchas gracias", "ok", "okay", "va", "vale", "listo",
    "si", "sí", "no", "adios", "adiós", "bye", "hasta luego", "perfecto",
    "genial", "entendido", "claro", "de acuerdo",
}

_MULTI_ACTION_HINTS = (
    " y ", " luego ", " después ", " despues ", " primero ", " además ",
    " ademas ", " también ", " tambien ", " entonces ", " finalmente ",
    ", ", "; ", "\n",
)


def should_skip_planner(query: str) -> bool:
    """Heurística rápida para evitar llamadas al planner en saludos o respuestas triviales."""
    if not query or not isinstance(query, str):
        return True
    clean = re.sub(r"[¿?¡!.,;:]+", " ", query.strip().lower()).strip()
    if not clean or clean in _SKIP_EXACT:
        return True
    has_connector = any(h in query.lower() for h in _MULTI_ACTION_HINTS)
    if len(clean) < 18 and not has_connector:
        return True
    return False


def parse_plan_json(raw_text: str, max_steps: int = 5) -> Dict[str, Any]:
    """Extrae y valida el JSON de planificación. Cae a modo 'direct' si hay < 2 pasos válidos."""
    fallback = {"mode": "direct", "reason": "direct_execution", "steps": []}
    if not raw_text or not isinstance(raw_text, str):
        return fallback

    candidates: List[str] = []
    for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", raw_text, flags=re.IGNORECASE):
        inner = m.group(1).strip()
        if inner.startswith("{") and inner.endswith("}"):
            candidates.append(inner)

    if not candidates:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end > start:
            candidates.append(raw_text[start : end + 1])

    parsed = None
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                parsed = obj
                break
        except Exception:
            continue

    if not parsed:
        return fallback

    mode = str(parsed.get("mode", "direct")).strip().lower()
    reason = str(parsed.get("reason", "")).strip()
    raw_steps = parsed.get("steps")

    if mode != "plan" or not isinstance(raw_steps, list):
        return {"mode": "direct", "reason": reason or "single_step_or_chat", "steps": []}

    normalized_steps: List[Dict[str, Any]] = []
    for step in raw_steps:
        if not isinstance(step, dict):
            continue
        goal = str(step.get("goal", "")).strip()
        if not goal:
            continue
        needs_tools = bool(step.get("needs_tools", True))
        tool_query = str(step.get("tool_query", "")).strip() if needs_tools else ""
        normalized_steps.append({
            "id": len(normalized_steps) + 1,
            "goal": goal,
            "tool_query": tool_query,
            "needs_tools": needs_tools,
        })
        if len(normalized_steps) >= max_steps:
            break

    if len(normalized_steps) < 2:
        return {"mode": "direct", "reason": reason or "fewer_than_two_steps", "steps": []}

    return {
        "mode": "plan",
        "reason": reason or "multi_step_task",
        "steps": normalized_steps,
    }


async def agenerate_plan(
    llm_client: Any,
    config: Any,
    current_query: str,
    recent_history: List[Dict[str, Any]],
    memory_context: str,
) -> Dict[str, Any]:
    """
    Fase 1 — Router / Planner: decide si la petición se resuelve directo o en un plan de pasos atómicos.
    Mantiene exactamente un mensaje system al inicio (compatible con Jinja de Bonsai-27b).
    """
    if should_skip_planner(current_query):
        return {"mode": "direct", "reason": "skipped_trivial", "steps": []}

    non_sys = [m for m in recent_history if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
    # Excluir el último turno si coincide con current_query para no duplicar en el prompt
    if non_sys and non_sys[-1].get("role") == "user" and str(non_sys[-1].get("content", "")).strip() == current_query.strip():
        non_sys = non_sys[:-1]
    snippet_msgs = non_sys[-4:]
    history_lines = []
    for m in snippet_msgs:
        c = str(m.get("content", "")).replace("\n", " ").strip()
        if c:
            history_lines.append(f"- {m.get('role')}: {c[:250]}")
    history_block = "\n".join(history_lines) if history_lines else "(sin historial previo relevante)"

    sys_prompt = (
        "Eres el planificador (Router/Planner) de CELab bot. Tu trabajo es decidir si una petición "
        "debe ejecutarse de forma directa (`direct`) o dividirse en un plan (`plan`) de pasos pequeños y secuenciales.\n"
        "REGLAS ESTRICTAS:\n"
        f"- Usa `\"mode\": \"direct\"` para saludos, charla, preguntas teóricas/conceptuales, o acciones simples de 1 solo paso.\n"
        f"- Usa `\"mode\": \"plan\"` ÚNICAMENTE cuando la petición requiera claramente 2 o más acciones secuenciales distintas "
        f"(máximo {config.max_plan_steps} pasos), por ejemplo: crear varios archivos, inspeccionar y luego editar/ejecutar, o construir y luego verificar.\n"
        "- Cada paso debe ser atómico (máximo 1 archivo editado o 1 comando/búsqueda).\n"
        "- En cada paso incluye: `goal` (en español, breve y concreto), `needs_tools` (true/false) y `tool_query` (en INGLÉS para buscar la tool en Laya, ej. 'create or edit python file', 'run terminal command to check syntax', 'list files in directory'; o '' si needs_tools es false).\n"
        "- Responde ÚNICAMENTE con JSON válido sin texto extra:\n"
        '{"mode": "direct" | "plan", "reason": "...", "steps": [{"id": 1, "goal": "...", "tool_query": "...", "needs_tools": true}]}'
        f"{memory_context}"
    )

    user_prompt = (
        f"Historial reciente:\n{history_block}\n\n"
        f"Petición actual del usuario:\n{current_query}"
    )

    try:
        resp = await llm_client.chat.completions.create(
            model=config.llm_model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=512,
            temperature=0.1,
            timeout=35.0,
            extra_body={"repeat_penalty": config.llm_repeat_penalty},
        )
        raw = (resp.choices[0].message.content or "").strip()
        plan = parse_plan_json(raw, max_steps=config.max_plan_steps)
        print(f"🗺️ [PLANNER] mode={plan['mode']} steps={len(plan['steps'])} reason={plan['reason']}")
        return plan
    except Exception as e:
        print(f"⚠️ [PLANNER] Error generando plan ({e}). Usando modo directo.")
        return {"mode": "direct", "reason": "planner_error", "steps": []}


def format_plan_markdown(plan_state: Dict[str, Any], denied_step_idx: Optional[int] = None) -> str:
    """Renderiza el plan de ejecución como un bloque colapsable compacto para OpenWebUI."""
    steps = plan_state.get("steps", [])
    current_idx = int(plan_state.get("current_step_idx", 0))
    completed_ids = {c.get("id") for c in plan_state.get("completed_steps", [])}

    lines = [
        f"\n<details open><summary>🗺️ <b>Plan de ejecución ({len(steps)} pasos)</b></summary>\n"
    ]
    for idx, step in enumerate(steps):
        sid = step.get("id", idx + 1)
        goal = step.get("goal", "")
        if sid in completed_ids or idx < current_idx:
            icon = "✅"
        elif denied_step_idx is not None and idx == denied_step_idx:
            icon = "🚫"
        elif idx == current_idx:
            icon = "🔄"
        else:
            icon = "⏳"
        lines.append(f"{idx + 1}. {icon} {goal}")
    lines.append("\n</details>\n\n")
    return "\n".join(lines)


def summarize_step_outcome(
    step_goal: str,
    assistant_text: str,
    tool_results: List[Dict[str, Any]],
) -> str:
    """Resume de forma determinista el resultado de un paso en <= 500 caracteres."""
    parts: List[str] = []
    for item in tool_results:
        t_name = item.get("tool", "tool")
        res = item.get("result", {})
        if isinstance(res, dict):
            if "error" in res:
                parts.append(f"{t_name}: error={str(res['error'])[:120]}")
            elif "status" in res:
                parts.append(f"{t_name}: {str(res['status'])[:140]}")
            elif "returncode" in res:
                out = (res.get("stdout") or res.get("stderr") or "").strip().replace("\n", " ")
                parts.append(f"{t_name}(rc={res.get('returncode')}): {out[:120]}")
            elif "files" in res:
                files = res.get("files", [])
                parts.append(f"{t_name}: {len(files)} elementos")
            else:
                parts.append(f"{t_name}: ok")
        else:
            parts.append(f"{t_name}: {str(res)[:100]}")

    note = (assistant_text or "").strip().replace("\n", " ")
    summary = " | ".join(parts)
    if note:
        summary = f"{summary} — {note}" if summary else note
    if not summary:
        summary = f"Completado: {step_goal}"
    if len(summary) > 500:
        summary = summary[:497] + "..."
    return summary


def build_step_messages(
    linux_user: str,
    is_admin: bool,
    memory_context: str,
    plan_state: Dict[str, Any],
    step_idx: int,
    recovery_note: str = "",
) -> List[Dict[str, Any]]:
    """Construye el contexto mínimo aislado [system, user] para ejecutar un paso específico."""
    steps = plan_state.get("steps", [])
    total = len(steps)
    step = steps[step_idx]
    completed = plan_state.get("completed_steps", [])

    sys_content = (
        f"Eres CELab bot, el asistente inteligente del laboratorio ejecutando un plan por pasos.\n"
        f"Usuario Linux: '{linux_user}'{' (ADMINISTRADOR)' if is_admin else ''}.\n"
        f"REGLAS DEL PASO ACTUAL ({step_idx + 1}/{total}):\n"
        f"1. Ejecuta ÚNICAMENTE este paso concreto usando las herramientas disponibles.\n"
        f"2. Para crear o modificar archivos (`.py`, `.html`, `.css`, `.txt`, configs), usa SIEMPRE `edit_file` (nunca `cat << EOF` en `run_terminal_command`), con código limpio y conciso.\n"
        f"3. Usa `run_terminal_command` solo para comandos shell (`ps`, `df`, `git`, `python -m py_compile`, ejecutar scripts).\n"
        f"4. Si falta alguna herramienta específica, llama a `search_tools` con query EN INGLÉS.\n"
        f"5. Al terminar este paso, resume de forma breve y factual lo que hiciste u obtuviste."
        f"{memory_context}"
    )

    user_lines = [f"Objetivo global del usuario: {plan_state.get('original_quest', '')}"]
    if completed:
        user_lines.append("\nPasos completados anteriormente:")
        for c in completed:
            user_lines.append(f"- Paso {c.get('id')}: {c.get('goal')} -> {c.get('summary')}")

    user_lines.append(f"\nPaso actual a ejecutar ({step_idx + 1}/{total}): {step.get('goal', '')}")
    if recovery_note:
        user_lines.append(f"\n[NOTA DE RECUPERACIÓN]:\n{recovery_note}")

    return [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def build_synthesizer_messages(
    linux_user: str,
    is_admin: bool,
    memory_context: str,
    plan_state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Construye el contexto compacto [system, user] para redactar la respuesta final del plan."""
    completed = plan_state.get("completed_steps", [])
    sys_content = (
        f"Eres CELab bot, el asistente inteligente del laboratorio.\n"
        f"Usuario Linux: '{linux_user}'{' (ADMINISTRADOR)' if is_admin else ''}.\n"
        f"INSTRUCCIÓN: Redacta la respuesta final al usuario en español de forma clara y concisa, "
        f"resumiendo qué se logró en cada paso del plan, rutas de archivos creados o salidas relevantes, "
        f"y cualquier advertencia o siguiente paso sugerido. No inventes resultados que no estén en el reporte."
        f"{memory_context}"
    )

    lines = [
        f"Petición original del usuario: {plan_state.get('original_quest', '')}",
        "\nResultados de los pasos ejecutados:",
    ]
    for c in completed:
        lines.append(f"- Paso {c.get('id')} ({c.get('goal')}): {c.get('summary')}")

    return [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": "\n".join(lines)},
    ]
