import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Fallback
    except ImportError:
        tomllib = None


@dataclass
class CelabConfig:
    # LLM Settings
    llm_base_url: str = "http://localhost:3000/v1"
    llm_model: str = "Bonsai-27b-Q1_0"
    llm_api_key: str = "no-key-needed"
    llm_timeout: float = 90.0
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.2
    llm_repeat_penalty: float = 1.1

    # Laya Routers
    tool_model_path: str = "Marianinpb/laya-toolrag-router"
    context_model_path: str = "Marianinpb/laya-context-router"
    laya_device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    search_threshold: float = 0.05

    # Agent & Memory Settings
    max_agent_steps: int = 8
    max_memory_inject: int = 5
    always_include_terminal_tool: bool = True

    # System Paths
    shared_dir: Path = Path("/opt/celab-bot/shared")
    users_file: Path = Path("/etc/celab-bot/users.toml")
    config_file: Path = Path("/etc/celab-bot/config.toml")
    exec_binary: Path = Path("/opt/celab-bot/bin/celab-exec")

    # API Server
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    api_key: str = "celab-secret-key"

    # Runtime Flags
    single_user_mode: bool = False  # Para desarrollo local sin sudoers
    require_confirmation: bool = True  # Confirmar antes de ejecutar tools destructivas


def load_config(config_path: Optional[str] = None) -> CelabConfig:
    """Carga configuración desde archivo TOML, entorno o valores por defecto."""
    cfg = CelabConfig()

    candidates = [
        Path(config_path) if config_path else None,
        Path(os.environ.get("CELAB_CONFIG", "")),
        Path("/etc/celab-bot/config.toml"),
        Path("./config.toml"),
    ]

    selected_file = None
    for c in candidates:
        if c and c.is_file():
            selected_file = c
            break

    if selected_file and tomllib:
        try:
            with open(selected_file, "rb") as f:
                data = tomllib.load(f)

            # Mapear secciones del TOML
            llm = data.get("llm", {})
            cfg.llm_base_url = llm.get("base_url", cfg.llm_base_url)
            cfg.llm_model = llm.get("model", cfg.llm_model)
            cfg.llm_api_key = llm.get("api_key", cfg.llm_api_key)
            cfg.llm_timeout = float(llm.get("timeout", cfg.llm_timeout))
            cfg.llm_max_tokens = int(llm.get("max_tokens", cfg.llm_max_tokens))
            cfg.llm_temperature = float(llm.get("temperature", cfg.llm_temperature))
            cfg.llm_repeat_penalty = float(llm.get("repeat_penalty", cfg.llm_repeat_penalty))

            laya = data.get("laya", {})
            cfg.tool_model_path = laya.get("tool_model", cfg.tool_model_path)
            cfg.context_model_path = laya.get("context_model", cfg.context_model_path)
            cfg.laya_device = laya.get("device", cfg.laya_device)
            cfg.search_threshold = float(laya.get("search_threshold", cfg.search_threshold))

            agent = data.get("agent", {})
            cfg.max_agent_steps = int(agent.get("max_agent_steps", cfg.max_agent_steps))
            cfg.max_memory_inject = int(agent.get("max_memory_inject", cfg.max_memory_inject))
            cfg.always_include_terminal_tool = bool(
                agent.get("always_include_terminal_tool", cfg.always_include_terminal_tool)
            )

            paths = data.get("paths", {})
            if "shared_dir" in paths:
                cfg.shared_dir = Path(paths["shared_dir"])
            if "users_file" in paths:
                cfg.users_file = Path(paths["users_file"])

            server = data.get("server", {})
            cfg.server_host = server.get("host", cfg.server_host)
            cfg.server_port = int(server.get("port", cfg.server_port))
            cfg.api_key = server.get("api_key", cfg.api_key)
            cfg.require_confirmation = bool(server.get("require_confirmation", cfg.require_confirmation))

        except Exception as e:
            print(f"⚠️ [CONFIG] Error leyendo {selected_file}: {e}. Usando defaults/env.")

    # Variables de entorno tienen prioridad final
    if os.environ.get("CELAB_LLM_URL"):
        cfg.llm_base_url = os.environ["CELAB_LLM_URL"]
    if os.environ.get("CELAB_LLM_MODEL"):
        cfg.llm_model = os.environ["CELAB_LLM_MODEL"]
    if os.environ.get("CELAB_API_KEY"):
        cfg.api_key = os.environ["CELAB_API_KEY"]
    if os.environ.get("CELAB_SINGLE_USER"):
        cfg.single_user_mode = os.environ["CELAB_SINGLE_USER"].lower() in ("1", "true", "yes")

    # Si /opt/celab-bot no existe y estamos en dev local, usar ./shared
    if not cfg.shared_dir.exists() and Path("./shared").exists():
        cfg.shared_dir = Path("./shared").resolve()
    if not cfg.users_file.parent.exists() and Path("./users.toml").exists():
        cfg.users_file = Path("./users.toml").resolve()

    return cfg
