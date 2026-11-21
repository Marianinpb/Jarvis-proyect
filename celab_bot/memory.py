from contextlib import contextmanager
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from celab_bot.users import get_user_data_dir


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _extract_fts_terms(text: str, max_terms: int = 12) -> List[str]:
    """Extrae palabras clave limpias para consultas FTS5 sin operadores rotos."""
    if not text:
        return []
    tokens = re.findall(r"[a-zA-Z0-9áéíóúÁÉÍÓÚñÑ_]{3,}", text.lower())
    stopwords = {
        "que", "con", "por", "para", "como", "una", "uno", "unos", "unas",
        "los", "las", "del", "este", "esta", "estos", "estas", "pero", "mas",
        "más", "muy", "sin", "sobre", "entre", "desde", "hasta", "donde",
        "cuando", "quien", "cual", "the", "and", "for", "with", "that", "this",
        "from", "have", "are", "was", "were", "you", "your", "can", "could",
        "would", "should", "what", "how", "why", "where", "when", "please",
        "porfa", "favor", "hola", "puedes", "quiero", "necesito", "dime",
    }
    seen = set()
    terms = []
    for tok in tokens:
        if tok in stopwords or tok in seen:
            continue
        seen.add(tok)
        terms.append(tok)
        if len(terms) >= max_terms:
            break
    return terms


class UserMemoryStore:
    """
    Gestiona la memoria a largo plazo y el historial clasificado por Laya en SQLite
    aislado por usuario en /opt/celab-bot/users_data/<usuario>/memory/memory.db.
    """

    CATEGORIES = [
        "user_preferences",
        "project_context",
        "general_knowledge",
        "conversational_context",
    ]

    def __init__(self, linux_user: str, base_dir: Optional[Path] = None):
        self.linux_user = linux_user
        if base_dir:
            self.user_dir = Path(base_dir)
        else:
            self.user_dir = get_user_data_dir(linux_user) / "memory"

        try:
            self.user_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        self.db_path = self.user_dir / "memory.db"
        self.legacy_json_path = self.user_dir / "memory_store.json"
        self._fts_enabled = False
        self._init_db()
        self._migrate_legacy_json()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        try:
            with self._connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content_hash TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        content_en TEXT,
                        category TEXT,
                        is_relevant_score REAL,
                        should_store_score REAL,
                        created_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_messages_hash ON messages(content_hash);
                    CREATE INDEX IF NOT EXISTS idx_messages_category ON messages(category);

                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content_hash TEXT UNIQUE NOT NULL,
                        category TEXT NOT NULL,
                        content TEXT NOT NULL,
                        content_en TEXT,
                        importance REAL DEFAULT 0.5,
                        access_count INTEGER DEFAULT 0,
                        created_at REAL NOT NULL,
                        last_accessed REAL NOT NULL,
                        active INTEGER DEFAULT 1
                    );
                    CREATE INDEX IF NOT EXISTS idx_memories_cat_active
                        ON memories(category, active, last_accessed DESC);

                    CREATE TABLE IF NOT EXISTS translations (
                        text_hash TEXT PRIMARY KEY,
                        source_text TEXT NOT NULL,
                        text_en TEXT NOT NULL,
                        created_at REAL NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS pending_confirmations (
                        linux_user TEXT PRIMARY KEY,
                        tool_name TEXT NOT NULL,
                        args_json TEXT NOT NULL,
                        description TEXT NOT NULL,
                        quest TEXT NOT NULL,
                        plan_state_json TEXT,
                        created_at REAL NOT NULL
                    );
                    """
                )

                # Índice FTS5 para búsqueda rápida de recuerdos (con fallback si no está compilado)
                try:
                    conn.executescript(
                        """
                        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                            content,
                            content_en,
                            category,
                            content='memories',
                            content_rowid='id'
                        );

                        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                            INSERT INTO memories_fts(rowid, content, content_en, category)
                            VALUES (new.id, new.content, COALESCE(new.content_en, ''), new.category);
                        END;

                        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                            INSERT INTO memories_fts(memories_fts, rowid, content, content_en, category)
                            VALUES ('delete', old.id, old.content, COALESCE(old.content_en, ''), old.category);
                        END;

                        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                            INSERT INTO memories_fts(memories_fts, rowid, content, content_en, category)
                            VALUES ('delete', old.id, old.content, COALESCE(old.content_en, ''), old.category);
                            INSERT INTO memories_fts(rowid, content, content_en, category)
                            VALUES (new.id, new.content, COALESCE(new.content_en, ''), new.category);
                        END;
                        """
                    )
                    self._fts_enabled = True
                except sqlite3.OperationalError:
                    self._fts_enabled = False
        except Exception as e:
            print(f"⚠️ [MEMORY] Error inicializando SQLite para {self.linux_user}: {e}")

    def _migrate_legacy_json(self) -> None:
        """Migra automáticamente el antiguo memory_store.json si existe."""
        if not self.legacy_json_path.is_file():
            return
        try:
            with open(self.legacy_json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            migrated = 0
            for cat in self.CATEGORIES:
                items = loaded.get(cat, [])
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, str) and item.strip():
                            if self.add(cat, item.strip()):
                                migrated += 1
            backup_path = self.legacy_json_path.with_suffix(".json.bak")
            self.legacy_json_path.replace(backup_path)
            if migrated:
                print(f"✅ [MEMORY] Migrados {migrated} recuerdos de JSON a SQLite para {self.linux_user}.")
        except Exception as e:
            print(f"⚠️ [MEMORY] Error migrando memory_store.json para {self.linux_user}: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # CACHÉ PERSISTENTE DE TRADUCCIONES
    # ─────────────────────────────────────────────────────────────────────
    def get_translation(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return None
        th = _hash_text(text)
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT text_en FROM translations WHERE text_hash = ?",
                    (th,),
                ).fetchone()
                return row["text_en"] if row else None
        except Exception:
            return None

    def save_translation(self, source_text: str, text_en: str) -> None:
        if not source_text or not text_en:
            return
        th = _hash_text(source_text)
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO translations (text_hash, source_text, text_en, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (th, source_text.strip(), text_en.strip(), time.time()),
                )
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────
    # REGISTRO DE MENSAJES CLASIFICADOS POR LAYA
    # ─────────────────────────────────────────────────────────────────────
    def get_cached_message_classification(self, content: str) -> Optional[Dict[str, Any]]:
        """Recupera clasificación previa de un mensaje idéntico si ya fue evaluado."""
        if not content or not content.strip():
            return None
        ch = _hash_text(content)
        try:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT content_en, category, is_relevant_score, should_store_score
                    FROM messages
                    WHERE content_hash = ?
                    ORDER BY id DESC LIMIT 1
                    """,
                    (ch,),
                ).fetchone()
                return dict(row) if row else None
        except Exception:
            return None

    def log_message(
        self,
        role: str,
        content: str,
        content_en: Optional[str] = None,
        category: Optional[str] = None,
        is_relevant_score: Optional[float] = None,
        should_store_score: Optional[float] = None,
    ) -> None:
        clean = (content or "").strip()
        if not clean:
            return
        ch = _hash_text(clean)
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO messages (
                        content_hash, role, content, content_en,
                        category, is_relevant_score, should_store_score, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ch,
                        role,
                        clean,
                        content_en,
                        category,
                        is_relevant_score,
                        should_store_score,
                        time.time(),
                    ),
                )
        except Exception as e:
            print(f"⚠️ [MEMORY] Error registrando mensaje para {self.linux_user}: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # MEMORIAS A LARGO PLAZO (ESCRITURA Y RECUPERACIÓN)
    # ─────────────────────────────────────────────────────────────────────
    def add(
        self,
        category: str,
        content: str,
        content_en: Optional[str] = None,
        importance: float = 0.5,
    ) -> bool:
        """Añade un recuerdo a la categoría especificada evitando duplicados exactos."""
        if category not in self.CATEGORIES:
            return False

        clean_content = (content or "").strip()
        if not clean_content:
            return False

        ch = _hash_text(clean_content)
        now = time.time()
        try:
            with self._connect() as conn:
                existing = conn.execute(
                    "SELECT id, importance FROM memories WHERE content_hash = ?",
                    (ch,),
                ).fetchone()
                if existing:
                    # Reforzar importancia y fecha de acceso si reaparece
                    new_imp = max(float(existing["importance"] or 0.5), float(importance))
                    conn.execute(
                        "UPDATE memories SET last_accessed = ?, importance = ?, active = 1 WHERE id = ?",
                        (now, new_imp, existing["id"]),
                    )
                    return False

                conn.execute(
                    """
                    INSERT INTO memories (
                        content_hash, category, content, content_en,
                        importance, access_count, created_at, last_accessed, active
                    ) VALUES (?, ?, ?, ?, ?, 0, ?, ?, 1)
                    """,
                    (ch, category, clean_content, content_en, float(importance), now, now),
                )
                return True
        except Exception as e:
            print(f"❌ [MEMORY] Error guardando recuerdo para {self.linux_user}: {e}")
            return False

    def get_preferences(self, limit: int = 5) -> List[str]:
        """Devuelve las reglas y preferencias activas del usuario."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT content FROM memories
                    WHERE category = 'user_preferences' AND active = 1
                    ORDER BY importance DESC, last_accessed DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                return [r["content"] for r in rows]
        except Exception:
            return []

    def search_memories(
        self,
        query: str,
        query_en: Optional[str] = None,
        categories: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Busca recuerdos relevantes usando FTS5 (o LIKE fallback) y actualiza métricas de acceso."""
        target_cats = categories or ["project_context", "general_knowledge"]
        terms = _extract_fts_terms(f"{query or ''} {query_en or ''}")
        results: List[Dict[str, Any]] = []
        seen_ids = set()

        try:
            with self._connect() as conn:
                placeholders = ",".join("?" for _ in target_cats)

                if terms and self._fts_enabled:
                    fts_query = " OR ".join(f'"{t}"' for t in terms)
                    try:
                        rows = conn.execute(
                            f"""
                            SELECT m.id, m.category, m.content, m.content_en, m.importance, m.created_at
                            FROM memories_fts f
                            JOIN memories m ON m.id = f.rowid
                            WHERE memories_fts MATCH ?
                              AND m.active = 1
                              AND m.category IN ({placeholders})
                            ORDER BY bm25(memories_fts), m.importance DESC, m.last_accessed DESC
                            LIMIT ?
                            """,
                            (fts_query, *target_cats, limit),
                        ).fetchall()
                        for r in rows:
                            d = dict(r)
                            results.append(d)
                            seen_ids.add(d["id"])
                    except sqlite3.OperationalError:
                        pass

                elif terms:
                    like_clauses = " OR ".join("(content LIKE ? OR content_en LIKE ?)" for _ in terms)
                    like_params: List[Any] = []
                    for t in terms:
                        pat = f"%{t}%"
                        like_params.extend([pat, pat])
                    rows = conn.execute(
                        f"""
                        SELECT id, category, content, content_en, importance, created_at
                        FROM memories
                        WHERE active = 1
                          AND category IN ({placeholders})
                          AND ({like_clauses})
                        ORDER BY importance DESC, last_accessed DESC
                        LIMIT ?
                        """,
                        (*target_cats, *like_params, limit),
                    ).fetchall()
                    for r in rows:
                        d = dict(r)
                        results.append(d)
                        seen_ids.add(d["id"])

                # Complementar con los recuerdos recientes de project_context si hay pocos resultados
                if len(results) < min(2, limit):
                    rem = min(2, limit) - len(results)
                    rows = conn.execute(
                        """
                        SELECT id, category, content, content_en, importance, created_at
                        FROM memories
                        WHERE active = 1 AND category = 'project_context'
                        ORDER BY last_accessed DESC, importance DESC
                        LIMIT ?
                        """,
                        (rem + len(seen_ids),),
                    ).fetchall()
                    for r in rows:
                        if r["id"] not in seen_ids:
                            results.append(dict(r))
                            seen_ids.add(r["id"])
                            if len(results) >= limit:
                                break

                if results:
                    now = time.time()
                    ids = [r["id"] for r in results]
                    id_placeholders = ",".join("?" for _ in ids)
                    conn.execute(
                        f"""
                        UPDATE memories
                        SET access_count = access_count + 1, last_accessed = ?
                        WHERE id IN ({id_placeholders})
                        """,
                        (now, *ids),
                    )
        except Exception as e:
            print(f"⚠️ [MEMORY] Error buscando recuerdos para {self.linux_user}: {e}")

        return results[:limit]

    def format_prompt_context(
        self,
        query: str,
        query_en: Optional[str] = None,
        max_items: int = 5,
    ) -> str:
        """Construye un bloque compacto de memoria del usuario listo para inyectar al System Prompt."""
        prefs = self.get_preferences(limit=4)
        relevant = self.search_memories(
            query=query,
            query_en=query_en,
            categories=["project_context", "general_knowledge"],
            limit=max_items,
        )

        if not prefs and not relevant:
            return ""

        lines = ["\n[MEMORIA PERSISTENTE DEL USUARIO]:"]
        if prefs:
            lines.append("Preferencias y reglas del usuario:")
            for p in prefs:
                lines.append(f"- {p[:240]}")
        if relevant:
            lines.append("Contexto relevante recuperado:")
            for r in relevant:
                lines.append(f"- ({r['category']}) {r['content'][:260]}")
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────
    # CONFIRMACIONES PENDIENTES PERSISTENTES (SQLite)
    # ─────────────────────────────────────────────────────────────────────
    def get_pending_confirmation(self, ttl_seconds: int = 900) -> Optional[Dict[str, Any]]:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT tool_name, args_json, description, quest, plan_state_json, created_at
                    FROM pending_confirmations
                    WHERE linux_user = ?
                    """,
                    (self.linux_user,),
                ).fetchone()
                if not row:
                    return None
                if time.time() - float(row["created_at"]) > ttl_seconds:
                    self.clear_pending_confirmation()
                    return None
                return {
                    "tool_name": row["tool_name"],
                    "args": json.loads(row["args_json"]),
                    "description": row["description"],
                    "quest": row["quest"],
                    "plan_state": json.loads(row["plan_state_json"]) if row["plan_state_json"] else None,
                    "timestamp": float(row["created_at"]),
                }
        except Exception as e:
            print(f"⚠️ [MEMORY] Error leyendo confirmación pendiente: {e}")
            return None

    def set_pending_confirmation(
        self,
        tool_name: str,
        args: Dict[str, Any],
        description: str,
        quest: str,
        plan_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO pending_confirmations (
                        linux_user, tool_name, args_json, description, quest, plan_state_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.linux_user,
                        tool_name,
                        json.dumps(args, ensure_ascii=False),
                        description,
                        quest,
                        json.dumps(plan_state, ensure_ascii=False) if plan_state is not None else None,
                        time.time(),
                    ),
                )
        except Exception as e:
            print(f"⚠️ [MEMORY] Error guardando confirmación pendiente: {e}")

    def clear_pending_confirmation(self) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM pending_confirmations WHERE linux_user = ?",
                    (self.linux_user,),
                )
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────
    # INSPECCIÓN Y UTILIDADES (CLI / DEBUG)
    # ─────────────────────────────────────────────────────────────────────
    def get_all(self) -> Dict[str, List[str]]:
        """Devuelve todo el almacén de recuerdos activos agrupado por categoría."""
        out: Dict[str, List[str]] = {cat: [] for cat in self.CATEGORIES}
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT category, content FROM memories
                    WHERE active = 1
                    ORDER BY id ASC
                    """
                ).fetchall()
                for r in rows:
                    if r["category"] in out:
                        out[r["category"]].append(r["content"])
        except Exception:
            pass
        return out

    def get_stats(self) -> Dict[str, Any]:
        """Devuelve estadísticas de la base de datos SQLite del usuario."""
        stats: Dict[str, Any] = {
            "db_path": str(self.db_path),
            "fts_enabled": self._fts_enabled,
            "total_messages": 0,
            "total_translations": 0,
            "memories_by_category": {cat: 0 for cat in self.CATEGORIES},
        }
        try:
            with self._connect() as conn:
                stats["total_messages"] = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
                stats["total_translations"] = conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0]
                rows = conn.execute(
                    "SELECT category, COUNT(*) as cnt FROM memories WHERE active = 1 GROUP BY category"
                ).fetchall()
                for r in rows:
                    stats["memories_by_category"][r["category"]] = r["cnt"]
        except Exception:
            pass
        return stats

    def get_recent_messages(self, limit: int = 15) -> List[Dict[str, Any]]:
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT id, role, content, category, is_relevant_score, should_store_score, created_at
                    FROM messages
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []
