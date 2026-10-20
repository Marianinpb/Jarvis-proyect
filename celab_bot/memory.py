import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from celab_bot.users import get_user_home


class UserMemoryStore:
    """Gestiona la memoria a largo plazo individual en el $HOME de cada usuario Linux."""

    CATEGORIES = [
        "user_preferences",
        "project_context",
        "general_knowledge",
        "conversational_context",
    ]

    def __init__(self, linux_user: str, base_dir: Optional[Path] = None):
        self.linux_user = linux_user
        if base_dir:
            self.user_dir = base_dir / ".celab-bot"
        else:
            self.user_dir = get_user_home(linux_user) / ".celab-bot"

        self.file_path = self.user_dir / "memory_store.json"
        self.store: Dict[str, List[str]] = {cat: [] for cat in self.CATEGORIES}
        self.load()

    def load(self) -> Dict[str, List[str]]:
        """Carga la memoria del usuario desde su archivo personal."""
        if not self.file_path.is_file():
            return self.store

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                for cat in self.CATEGORIES:
                    if cat in loaded and isinstance(loaded[cat], list):
                        self.store[cat] = loaded[cat]
        except Exception as e:
            print(f"⚠️ [MEMORY] Error leyendo memoria para {self.linux_user}: {e}")

        return self.store

    def save(self) -> bool:
        """Guarda la memoria en disco asegurando permisos de usuario."""
        try:
            self.user_dir.mkdir(parents=True, exist_ok=True)
            # Escribir temporalmente y renombrar para atomicidad
            tmp_path = self.file_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.store, f, indent=4, ensure_ascii=False)
            tmp_path.replace(self.file_path)
            return True
        except Exception as e:
            print(f"❌ [MEMORY] Error guardando memoria para {self.linux_user}: {e}")
            return False

    def add(self, category: str, content: str) -> bool:
        """Añade un elemento a la categoría especificada evitando duplicados exactos."""
        if category not in self.store:
            return False

        clean_content = content.strip()
        if not clean_content:
            return False

        if clean_content not in self.store[category]:
            self.store[category].append(clean_content)
            self.save()
            return True

        return False

    def get_preferences(self) -> List[str]:
        """Devuelve las reglas y preferencias del usuario."""
        return self.store.get("user_preferences", [])

    def get_all(self) -> Dict[str, List[str]]:
        """Devuelve todo el almacén de memoria."""
        return self.store
