"""Persistent memory for the assistant, backed by SQLite."""
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

# Split a query into lowercase alphanumeric word tokens
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class MemoryStore:
    def __init__(self, db_path: str):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    key         TEXT NOT NULL UNIQUE,
                    value       TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def store(self, key: str, value: str) -> str:
        """Insert a fact, or overwrite the value if the key already exists."""
        key = (key or "").strip()
        value = (value or "").strip()
        if not key or not value:
            return "Both a key and a value are required to store a memory."
        now = self._now()
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO memories (key, value, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, now, now),
            )
        return f"Remembered '{key}': {value}"

    def recall(self, query: str, limit: int = 5) -> str:
        """Find facts relevant to the query.

        Ranks rows by how many words hit, and returns
        the best few. On a miss it surfaces what's stored, so the model can
        recover within this same tool call instead of having to chain another.
        """
        tokens = _tokenize(query)
        needles = [t for t in tokens if len(t) >= 2] or tokens
        needles = list(dict.fromkeys(needles))  # dedupe, preserve order
        if not needles:
            return self._available_hint()
        clause = " OR ".join(["LOWER(key) LIKE ? OR LOWER(value) LIKE ?"] * len(needles))
        params = [p for n in needles for p in (f"%{n}%", f"%{n}%")]
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT key, value, updated_at FROM memories WHERE {clause}", params
            ).fetchall()
        if not rows:
            return self._available_hint()
        q = query.lower().strip()
        rows.sort(key=lambda r: (self._relevance(r[0], r[1], needles, q), r[2]), reverse=True)
        return "\n".join(f"- {key}: {value}" for key, value, _ in rows[:limit])

    @staticmethod
    def _relevance(key: str, value: str, needles: list[str], q: str) -> int:
        key_l, value_l = key.lower(), value.lower()
        score = sum(2 for n in needles if n in key_l) + sum(1 for n in needles if n in value_l)
        if q and (q in key_l or q in value_l):
            score += 5  # an exact-phrase hit beats scattered single-word hits
        return score

    def _available_hint(self, cap: int = 50) -> str:
        """Show what is stored so the model can answer without needing a second tool turn."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT key, value FROM memories ORDER BY updated_at DESC"
            ).fetchall()
        if not rows:
            return "No memories found."
        if len(rows) <= cap:
            listed = "\n".join(f"- {key}: {value}" for key, value in rows)
            return f"No memory matched that search. Everything currently stored:\n{listed}"
        keys = ", ".join(key for key, _ in rows[:cap])
        return (
            "No memory matched that search. Stored memory keys include: "
            f"{keys}. Try searching again with one of these keywords."
        )

    def list_all(self) -> str:
        """Return every stored fact as readable text, newest first."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT key, value FROM memories ORDER BY updated_at DESC"
            ).fetchall()
        if not rows:
            return "No memories found."
        return "\n".join(f"- {key}: {value}" for key, value in rows)

    def delete(self, key: str) -> str:
        """Delete the fact stored under an exact key."""
        key = (key or "").strip()
        with closing(self._connect()) as conn, conn:
            deleted = conn.execute(
                "DELETE FROM memories WHERE key = ?", (key,)
            ).rowcount
        if deleted:
            return f"Forgot '{key}'."
        return f"No memory found with key '{key}'."
