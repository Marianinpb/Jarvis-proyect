#!/usr/bin/env python3
"""
Módulo CLI de administración y servicio para celab-bot.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from celab_bot.config import load_config
from celab_bot.memory import UserMemoryStore
from celab_bot.users import is_admin_user, link_user, load_users, unlink_user


def cmd_link(args):
    ok, msg = link_user(args.openwebui_user, args.linux_user)
    if ok:
        print(f"✅ {msg}")
        admin_tag = " (ADMIN)" if is_admin_user(args.linux_user) else ""
        print(f"   Identidad asignada: OpenWebUI '{args.openwebui_user}' ➔ Linux '{args.linux_user}'{admin_tag}")
    else:
        print(f"❌ {msg}")
        sys.exit(1)


def cmd_unlink(args):
    ok, msg = unlink_user(args.openwebui_user)
    if ok:
        print(f"✅ {msg}")
    else:
        print(f"❌ {msg}")
        sys.exit(1)


def cmd_users(args):
    users = load_users()
    if not users:
        print("ℹ️ No hay usuarios vinculados todavía.")
        print("   Usa: sudo celab-bot link <openwebui_user> <linux_user>")
        return

    print("📋 Vinculaciones OpenWebUI ➔ Linux:")
    print("-" * 55)
    print(f"{'OpenWebUI User / Email':<30} {'Linux User':<15} {'Rol':<10}")
    print("-" * 55)
    for ow, lx in sorted(users.items()):
        role = "ADMIN" if is_admin_user(lx) else "user"
        print(f"{ow:<30} {lx:<15} {role:<10}")
    print("-" * 55)


def cmd_memory(args):
    store = UserMemoryStore(args.linux_user)
    stats = store.get_stats()
    print(f"🧠 Base de Datos de Memoria SQLite — Usuario: {args.linux_user}")
    print(f"   Ruta DB: {stats['db_path']} (FTS5: {'ON' if stats['fts_enabled'] else 'OFF'})")
    print(f"   Mensajes clasificados: {stats['total_messages']} | Traducciones cacheadas: {stats['total_translations']}")
    print("   Recuerdos activos por categoría:")
    for cat, count in stats["memories_by_category"].items():
        print(f"     • {cat}: {count}")

    if args.show_all:
        all_mems = store.get_all()
        print("\n📚 Recuerdos guardados:")
        for cat, items in all_mems.items():
            if items:
                print(f"\n  [{cat}]")
                for idx, item in enumerate(items, 1):
                    print(f"    {idx}. {item[:180]}")

    if args.recent:
        recent = store.get_recent_messages(limit=args.recent)
        print(f"\n🕒 Últimos {len(recent)} mensajes clasificados por Laya:")
        for m in recent:
            rel = f"{m['is_relevant_score']:.2f}" if m["is_relevant_score"] is not None else "-"
            st = f"{m['should_store_score']:.2f}" if m["should_store_score"] is not None else "-"
            print(f"  #{m['id']} [{m['category'] or 'n/a'} | rel={rel} store={st}] {m['content'][:120]}")


def cmd_status(args):
    subprocess.run(["systemctl", "status", "celab-bot.service"])


def cmd_logs(args):
    subprocess.run(["journalctl", "-u", "celab-bot.service", "-f", "-n", "50"])


def cmd_restart(args):
    print("🔄 Reiniciando celab-bot.service...")
    subprocess.run(["systemctl", "restart", "celab-bot.service"])


def cmd_update(args):
    root_dir = Path(__file__).resolve().parent.parent
    print("🚀 Actualizando CELab bot desde GitHub...")
    subprocess.run(["git", "config", "--global", "--add", "safe.directory", str(root_dir)], cwd=str(root_dir))
    subprocess.run(["git", "fetch", "origin"], cwd=str(root_dir))
    res = subprocess.run(["git", "reset", "--hard", "origin/master"], cwd=str(root_dir))
    if res.returncode != 0:
        subprocess.run(["git", "reset", "--hard", "origin/main"], cwd=str(root_dir))

    venv_dir = root_dir / "venv"
    import shutil
    uv_bin = None
    for cand in ["uv", str(Path.home() / ".local" / "bin" / "uv"), "/root/.local/bin/uv"]:
        if shutil.which(cand) or Path(cand).is_file():
            uv_bin = cand
            break

    if uv_bin and venv_dir.is_dir():
        subprocess.run([uv_bin, "pip", "install", "--python", str(venv_dir), "-e", "."], cwd=str(root_dir))
    elif venv_dir.is_dir() and (venv_dir / "bin" / "pip").is_file():
        subprocess.run([str(venv_dir / "bin" / "pip"), "install", "-e", "."], cwd=str(root_dir))

    if os.name == "posix":
        subprocess.run(["chown", "-R", "celab-bot:celab-users", str(root_dir)])

    print("🔄 Reiniciando servicio...")
    subprocess.run(["systemctl", "restart", "celab-bot.service"])
    print("✅ Actualización completada con éxito.")


def cmd_serve(args):
    import uvicorn
    from celab_bot.server import create_app
    config = load_config(args.config)
    app = create_app(config)
    print(f"🚀 Iniciando CELab Bot API en {config.server_host}:{config.server_port}...")
    uvicorn.run(app, host=config.server_host, port=config.server_port, log_level="info")


def main():
    parser = argparse.ArgumentParser(description="CELab Bot CLI de administración y servicio")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Link
    p_link = subparsers.add_parser("link", help="Vincular usuario de OpenWebUI con usuario Linux")
    p_link.add_argument("openwebui_user", help="Nombre o email que aparece en OpenWebUI")
    p_link.add_argument("linux_user", help="Nombre de usuario real en Linux")
    p_link.set_defaults(func=cmd_link)

    # Unlink
    p_unlink = subparsers.add_parser("unlink", help="Eliminar vinculación de usuario")
    p_unlink.add_argument("openwebui_user", help="Nombre o email en OpenWebUI a desvincular")
    p_unlink.set_defaults(func=cmd_unlink)

    # Users
    p_users = subparsers.add_parser("users", help="Listar usuarios vinculados")
    p_users.set_defaults(func=cmd_users)

    # Memory DB Inspector
    p_mem = subparsers.add_parser("memory", help="Inspeccionar base de datos SQLite de memoria de un usuario")
    p_mem.add_argument("linux_user", help="Usuario Linux cuya memoria SQLite deseas inspeccionar")
    p_mem.add_argument("--all", dest="show_all", action="store_true", help="Mostrar todos los recuerdos guardados")
    p_mem.add_argument("--recent", type=int, default=0, help="Mostrar los últimos N mensajes clasificados por Laya")
    p_mem.set_defaults(func=cmd_memory)

    # Systemctl wrappers
    p_status = subparsers.add_parser("status", help="Ver estado del servicio systemd")
    p_status.set_defaults(func=cmd_status)

    p_logs = subparsers.add_parser("logs", help="Ver logs en tiempo real")
    p_logs.set_defaults(func=cmd_logs)

    p_restart = subparsers.add_parser("restart", help="Reiniciar el servicio")
    p_restart.set_defaults(func=cmd_restart)

    p_update = subparsers.add_parser("update", help="Actualizar código desde git y reiniciar")
    p_update.set_defaults(func=cmd_update)

    # Serve
    p_serve = subparsers.add_parser("serve", help="Ejecutar servidor FastAPI")
    p_serve.add_argument("--config", help="Ruta al archivo de configuración toml", default=None)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
