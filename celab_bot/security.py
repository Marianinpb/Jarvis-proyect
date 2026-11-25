from typing import Dict, Any, Tuple

DESTRUCTIVE_TOOLS = {
    "run_terminal_command",
    "edit_file",
}


def requires_user_confirmation(tool_name: str, args: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Determina si una llamada a herramienta altera el sistema y requiere confirmación explícita.
    Devuelve (requiere_confirmacion, mensaje_descriptivo).
    """
    if tool_name == "run_terminal_command":
        cmd = (args.get("command") or "").strip()
        # Comandos completamente benignos de solo lectura pueden exentarse si se desea
        safe_read_commands = {"whoami", "pwd", "date", "uptime"}
        if cmd in safe_read_commands:
            return False, ""
        if "\n" in cmd or "`" in cmd or len(cmd) > 100:
            return True, f"Ejecutar comando en terminal:\n```bash\n{cmd}\n```"
        return True, f"Ejecutar comando en terminal: `{cmd}`"

    if tool_name == "edit_file":
        path = (args.get("path") or "").strip()
        prev_text = args.get("prev_text", "")
        new_text = args.get("new_text", "")
        action = "Modificar" if prev_text else "Crear/Sobrescribir"
        return True, f"{action} el archivo `{path}` ({len(new_text)} caracteres)"

    return False, ""


def format_confirmation_request(tool_call_id: str, tool_name: str, description: str, quest: str = "") -> str:
    """Formatea la petición de confirmación para ser renderizada en OpenWebUI."""
    quest_section = f"\n🎯 **Objetivo:** *{quest}*\n" if quest else ""
    if "\n" in description:
        desc_block = description
    else:
        desc_block = f"> **{description}**"
    return (
        f"⚠️ **Confirmación de Seguridad Requerida**\n\n"
        f"CELab bot solicita tu autorización para:\n"
        f"{desc_block}\n"
        f"{quest_section}\n"
        f"¿Deseas autorizar esta acción?\n"
        f"- Responde **\"sí\"** (o *adelante / confirmar*) para ejecutarla inmediatamente.\n"
        f"- Responde **\"no\"** (o *cancelar*) para descartarla y buscar una alternativa."
    )
