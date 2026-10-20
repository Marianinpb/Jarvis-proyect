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
        cmd = args.get("command", "")
        # Comandos completamente benignos de solo lectura pueden exentarse si se desea
        safe_read_commands = {"whoami", "pwd", "date", "uptime"}
        if cmd.strip() in safe_read_commands:
            return False, ""
        return True, f"Ejecutar comando en terminal: `{cmd}`"

    if tool_name == "edit_file":
        path = args.get("path", "")
        prev_text = args.get("prev_text", "")
        action = "Modificar" if prev_text else "Crear/Sobrescribir"
        return True, f"{action} el archivo: `{path}`"

    return False, ""


def format_confirmation_request(tool_call_id: str, tool_name: str, description: str) -> str:
    """Formatea la petición de confirmación para ser renderizada en OpenWebUI."""
    return (
        f"⚠️ **Confirmación de Seguridad Requerida**\n\n"
        f"CELab bot solicita autorización para:\n"
        f"> **{description}**\n\n"
        f"¿Deseas autorizar esta acción? (Responde *'sí'* para confirmar o *'no'* para cancelar)."
    )
