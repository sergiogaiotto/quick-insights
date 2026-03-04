import hashlib
import secrets
from datetime import datetime, timedelta
from app.core.config import settings
from app.core.database import get_sync_connection


# ---------------------------------------------------------------------------
# API Key Management (existing)
# ---------------------------------------------------------------------------

def hash_api_key(raw_key: str) -> str:
    salted = f"{settings.api_salt}{raw_key}"
    return hashlib.sha256(salted.encode("utf-8")).hexdigest()


def generate_api_key() -> str:
    return secrets.token_urlsafe(32)


def create_api_key(label: str) -> dict:
    raw_key = generate_api_key()
    key_hash = hash_api_key(raw_key)
    conn = get_sync_connection()
    try:
        conn.execute(
            "INSERT INTO api_keys (key_hash, label) VALUES (?, ?)",
            (key_hash, label),
        )
        conn.commit()
        return {"key": raw_key, "label": label}
    finally:
        conn.close()


def validate_api_key(raw_key: str) -> bool:
    key_hash = hash_api_key(raw_key)
    conn = get_sync_connection()
    try:
        cursor = conn.execute(
            "SELECT id FROM api_keys WHERE key_hash = ? AND is_active = 1",
            (key_hash,),
        )
        return cursor.fetchone() is not None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Password Hashing
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    salted = f"{settings.api_salt}:{password}"
    return hashlib.sha256(salted.encode("utf-8")).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    return hash_password(password) == password_hash


# ---------------------------------------------------------------------------
# User Management
# ---------------------------------------------------------------------------

def get_user_count() -> int:
    conn = get_sync_connection()
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM users")
        return cursor.fetchone()[0]
    except Exception:
        return 0
    finally:
        conn.close()


def create_user(login: str, password: str, user_type: str = "user",
                display_name: str = "", profile_description: str = "") -> dict:
    conn = get_sync_connection()
    try:
        conn.execute(
            """INSERT INTO users (login, password_hash, user_type, display_name, profile_description)
               VALUES (?, ?, ?, ?, ?)""",
            (login, hash_password(password), user_type, display_name, profile_description),
        )
        conn.commit()
        user_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        return {"id": user_id, "login": login, "user_type": user_type, "display_name": display_name}
    finally:
        conn.close()


def authenticate_user(login: str, password: str) -> dict | None:
    """Authenticate and return user dict, or None. Auto-creates admin if DB is empty."""
    conn = get_sync_connection()
    try:
        # Check if any users exist
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count == 0:
            # First login ever → create admin
            pw_hash = hash_password(password)
            conn.execute(
                """INSERT INTO users (login, password_hash, user_type, display_name, profile_description)
                   VALUES (?, ?, 'admin', 'Super Usuário', 'Conta administrador criada automaticamente no primeiro acesso.')""",
                (login, pw_hash),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM users WHERE login = ? COLLATE NOCASE", (login,)).fetchone()
            return dict(row) if row else None

        # Normal login
        row = conn.execute("SELECT * FROM users WHERE login = ? COLLATE NOCASE AND is_active = 1", (login,)).fetchone()
        if row is None:
            return None
        user = dict(row)
        if not verify_password(password, user["password_hash"]):
            return None
        return user
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> dict | None:
    conn = get_sync_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE id = ? AND is_active = 1", (user_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_users() -> list[dict]:
    conn = get_sync_connection()
    try:
        cursor = conn.execute(
            "SELECT id, login, user_type, display_name, profile_description, is_active, created_at, updated_at "
            "FROM users ORDER BY created_at"
        )
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


def update_user(user_id: int, **kwargs) -> bool:
    allowed = {"login", "user_type", "display_name", "profile_description", "is_active"}
    fields = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    if not fields:
        return False
    fields["updated_at"] = datetime.utcnow().isoformat()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    conn = get_sync_connection()
    try:
        conn.execute(f"UPDATE users SET {set_clause} WHERE id = ?", (*fields.values(), user_id))
        conn.commit()
        return True
    finally:
        conn.close()


def change_password(user_id: int, new_password: str) -> bool:
    conn = get_sync_connection()
    try:
        conn.execute(
            "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
            (hash_password(new_password), datetime.utcnow().isoformat(), user_id),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def delete_user(user_id: int) -> bool:
    conn = get_sync_connection()
    try:
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        return True
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Session Management
# ---------------------------------------------------------------------------

SESSION_DURATION_HOURS = 24


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(48)
    expires = datetime.utcnow() + timedelta(hours=SESSION_DURATION_HOURS)
    conn = get_sync_connection()
    try:
        # Clean expired sessions
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (datetime.utcnow().isoformat(),))
        conn.execute(
            "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token, user_id, expires.isoformat()),
        )
        conn.commit()
        return token
    finally:
        conn.close()


def validate_session(token: str) -> dict | None:
    """Return user dict if session is valid, else None."""
    if not token:
        return None
    conn = get_sync_connection()
    try:
        row = conn.execute(
            """SELECT s.user_id, s.expires_at, u.id, u.login, u.user_type, u.display_name, u.profile_description, u.is_active
               FROM sessions s JOIN users u ON s.user_id = u.id
               WHERE s.token = ? AND u.is_active = 1""",
            (token,),
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        if datetime.fromisoformat(data["expires_at"]) < datetime.utcnow():
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
            return None
        return data
    finally:
        conn.close()


def destroy_session(token: str):
    conn = get_sync_connection()
    try:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
    finally:
        conn.close()