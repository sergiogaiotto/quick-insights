import pandas as pd
import sqlite3
import re
from pathlib import Path
from app.core.database import get_sync_connection


def sanitize_table_name(name: str) -> str:
    name = re.sub(r"[^\w]", "_", name.strip())
    name = re.sub(r"_+", "_", name).strip("_").lower()
    if name and name[0].isdigit():
        name = f"t_{name}"
    return name or "unnamed_table"


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def import_excel(file_path: Path) -> list[dict]:
    """Import all sheets from an Excel file into SQLite tables."""
    xls = pd.ExcelFile(file_path, engine="openpyxl")
    conn = get_sync_connection()
    report = []

    try:
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            if df.empty:
                report.append({
                    "sheet": sheet_name,
                    "table": None,
                    "action": "skipped",
                    "reason": "Aba vazia",
                    "rows": 0,
                })
                continue

            df.columns = [
                re.sub(r"[^\w]", "_", str(c).strip()).strip("_").lower()
                for c in df.columns
            ]

            tbl = sanitize_table_name(sheet_name)
            exists = table_exists(conn, tbl)

            if exists:
                df.to_sql(tbl, conn, if_exists="append", index=False)
                action = "appended"
            else:
                df.to_sql(tbl, conn, if_exists="replace", index=False)
                action = "created"

            conn.commit()
            report.append({
                "sheet": sheet_name,
                "table": tbl,
                "action": action,
                "rows": len(df),
                "columns": list(df.columns),
            })

        return report
    finally:
        conn.close()
