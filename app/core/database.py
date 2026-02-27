import sqlite3
from sqlalchemy import create_engine, text, inspect
from app.core.config import settings

# SQLAlchemy engine (used by SQL toolkit)
engine = create_engine(
    f"sqlite:///{settings.db_path}",
    echo=False,
    connect_args={"check_same_thread": False},
)


def get_sync_connection() -> sqlite3.Connection:
    """Raw sqlite3 connection for metadata operations."""
    conn = sqlite3.connect(str(settings.db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_metadata_tables():
    """Create internal metadata tables."""
    conn = get_sync_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS analysis_types (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                system_prompt TEXT NOT NULL DEFAULT '',
                guardrails_input TEXT NOT NULL DEFAULT '',
                guardrails_output TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                sql_generated TEXT,
                result_summary TEXT,
                analysis_type_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (analysis_type_id) REFERENCES analysis_types(id)
            );

            CREATE TABLE IF NOT EXISTS analysis_gallery (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                query_data TEXT NOT NULL,
                chart_config TEXT NOT NULL DEFAULT '',
                page_html TEXT NOT NULL DEFAULT '',
                share_token TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()

        # Migration: add page_html column if missing (existing DBs)
        try:
            conn.execute("SELECT page_html FROM analysis_gallery LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute("ALTER TABLE analysis_gallery ADD COLUMN page_html TEXT NOT NULL DEFAULT ''")
            conn.commit()

        cursor = conn.execute("SELECT COUNT(*) FROM analysis_types")
        if cursor.fetchone()[0] == 0:
            conn.execute("""
                INSERT INTO analysis_types (name, system_prompt, guardrails_input, guardrails_output)
                VALUES (
                    'Análise Geral',
                    'Você é um analista de dados especialista. Responda em português do Brasil. Gere SQL ANSI compatível com SQLite. Sempre explique os resultados de forma clara e objetiva.',
                    'A consulta deve ser relacionada aos dados disponíveis nas tabelas. Não permita comandos destrutivos (DROP, DELETE, UPDATE, INSERT).',
                    'Apresente os resultados de forma organizada. Inclua observações e insights quando relevante. Formate números com separadores de milhar.'
                )
            """)
            conn.commit()
    finally:
        conn.close()


def get_all_tables() -> list[dict]:
    """List all user data tables (excluding internal metadata)."""
    internal = {"analysis_types", "api_keys", "query_history", "analysis_gallery", "sqlite_sequence"}
    conn = get_sync_connection()
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = []
        for row in cursor.fetchall():
            name = row[0]
            if name in internal:
                continue
            col_cursor = conn.execute(f'PRAGMA table_info("{name}")')
            columns = [
                {"name": c[1], "type": c[2], "notnull": bool(c[3]), "pk": bool(c[5])}
                for c in col_cursor.fetchall()
            ]
            count_cursor = conn.execute(f'SELECT COUNT(*) FROM "{name}"')
            count = count_cursor.fetchone()[0]
            tables.append({"name": name, "columns": columns, "row_count": count})
        return tables
    finally:
        conn.close()


def get_table_schema_text() -> str:
    """Return a textual description of all user tables for context."""
    tables = get_all_tables()
    if not tables:
        return "Nenhuma tabela de dados encontrada no banco."
    parts = []
    for t in tables:
        cols = ", ".join(f'{c["name"]} ({c["type"]})' for c in t["columns"])
        parts.append(f'Tabela "{t["name"]}" ({t["row_count"]} registros): {cols}')
    return "\n".join(parts)


def execute_readonly_sql(sql: str) -> dict:
    """Execute a read-only SQL statement and return results."""
    forbidden = {
        "DROP", "DELETE", "UPDATE", "INSERT", "ALTER",
        "CREATE", "REPLACE", "TRUNCATE",
    }
    tokens = sql.upper().split()
    for token in tokens:
        if token in forbidden:
            return {"error": f"Comando '{token}' não é permitido em consultas de leitura."}

    conn = get_sync_connection()
    try:
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return {"columns": columns, "rows": rows, "row_count": len(rows)}
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()