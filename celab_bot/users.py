import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


def get_default_users_path(configured_path: Optional[Path] = None) -> Path:
    if configured_path and (configured_path.exists() or configured_path.parent.exists()):
        return configured_path
    if Path("/etc/celab-bot/users.toml").exists():
        return Path("/etc/celab-bot/users.toml")
    return Path("./users.toml").resolve()


def load_users(path: Optional[Path] = None) -> Dict[str, str]:
    """Carga el mapa {openwebui_id: linux_user} desde el archivo TOML."""
    file_path = get_default_users_path(path)
    if not file_path.is_file():
        return {}

    try:
        if tomllib:
            with open(file_path, "rb") as f:
                data = tomllib.load(f)
                return data.get("users", {})
        else:
            # Fallback parser elemental para cuando no hay módulo toml
            users = {}
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#") and not line.startswith("["):
                        k, v = line.split("=", 1)
                        users[k.strip().strip('"').strip("'")] = v.strip().strip('"').strip("'")
            return users
    except Exception as e:
        print(f"⚠️ [USERS] Error cargando usuarios desde {file_path}: {e}")
        return {}


def save_users(users: Dict[str, str], path: Optional[Path] = None) -> bool:
    """Guarda el diccionario de usuarios en formato TOML."""
    file_path = get_default_users_path(path)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# Mapeo explícito de usuarios: OpenWebUI -> Linux User\n")
            f.write("# Administrado con: sudo celab-bot link <openwebui_user> <linux_user>\n\n")
            f.write("[users]\n")
            for ow_user, lx_user in sorted(users.items()):
                f.write(f'"{ow_user}" = "{lx_user}"\n')
        return True
    except Exception as e:
        print(f"❌ [USERS] Error guardando {file_path}: {e}")
        return False


def link_user(openwebui_id: str, linux_user: str, path: Optional[Path] = None) -> Tuple[bool, str]:
    """Vincula un usuario de OpenWebUI con un usuario de Linux en el sistema."""
    openwebui_id = openwebui_id.strip()
    linux_user = linux_user.strip()

    if not openwebui_id or not linux_user:
        return False, "Nombre de OpenWebUI o usuario Linux vacío."

    # Verificar existencia del usuario en Linux si estamos en entorno POSIX
    if os.name == "posix":
        try:
            import pwd
            pwd.getpwnam(linux_user)
        except KeyError:
            return False, f"El usuario Linux '{linux_user}' no existe en este servidor."

    users = load_users(path)
    users[openwebui_id] = linux_user
    ok = save_users(users, path)
    if ok:
        return True, f"Vinculado '{openwebui_id}' -> '{linux_user}' con éxito."
    return False, "Error al escribir el archivo de usuarios."


def unlink_user(openwebui_id: str, path: Optional[Path] = None) -> Tuple[bool, str]:
    """Elimina la vinculación de un usuario."""
    openwebui_id = openwebui_id.strip()
    users = load_users(path)
    if openwebui_id not in users:
        return False, f"No existe vinculación para '{openwebui_id}'."

    del users[openwebui_id]
    ok = save_users(users, path)
    if ok:
        return True, f"Desvinculado '{openwebui_id}' con éxito."
    return False, "Error al escribir el archivo de usuarios."


def get_linux_user(openwebui_id: str, path: Optional[Path] = None) -> Optional[str]:
    """Obtiene el usuario Linux asociado al ID/nombre de OpenWebUI."""
    if not openwebui_id:
        return None
    users = load_users(path)
    return users.get(openwebui_id.strip())


def is_admin_user(linux_user: str) -> bool:
    """Verifica si el usuario Linux pertenece a los grupos celab-admins o sudo."""
    if not linux_user:
        return False

    # En Linux, verificar membresía real en el sistema
    if os.name == "posix":
        try:
            import grp
            admin_groups = {"celab-admins", "sudo", "wheel", "admin"}
            user_groups = set()

            # Obtener grupos primarios y secundarios del usuario
            import pwd
            pw = pwd.getpwnam(linux_user)
            # Grupo primario
            user_groups.add(grp.getgrgid(pw.pw_gid).gr_name)

            # Grupos secundarios
            for g in grp.getgrall():
                if linux_user in g.gr_mem:
                    user_groups.add(g.gr_name)

            return bool(user_groups & admin_groups)
        except Exception:
            # Fallback seguro con subprocess id -nG
            try:
                import subprocess
                res = subprocess.run(["id", "-nG", linux_user], capture_output=True, text=True, timeout=2)
                groups = set(res.stdout.strip().split())
                return bool(groups & {"celab-admins", "sudo", "wheel", "admin"})
            except Exception:
                return False

    # Entorno Windows / Dev fallback: si es mtico o admin
    return linux_user.lower() in ("mtico", "admin", "root", "mariano")


def get_user_home(linux_user: str) -> Path:
    """Obtiene el HOME del usuario Linux de forma segura."""
    if os.name == "posix":
        try:
            import pwd
            return Path(pwd.getpwnam(linux_user).pw_dir)
        except Exception:
            return Path(f"/home/{linux_user}")
    
    # En desarrollo local Windows, simular bajo su directorio actual o usuario actual
    return Path.home()
