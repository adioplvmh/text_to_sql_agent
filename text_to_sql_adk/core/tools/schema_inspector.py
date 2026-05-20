"""Schema inspector tool — reads DDL and sample values from the H&M PostgreSQL DB."""
from __future__ import annotations

import json
import logging
import textwrap

from sqlalchemy import create_engine, inspect, text

from text_to_sql_adk.core.config import DATABASE_URL

logger = logging.getLogger(__name__)

_ENGINE = None


def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _ENGINE


def inspect_schema(tables: list[str] | None = None) -> dict:
    """
    Inspect the H&M PostgreSQL database schema.

    Args:
        tables: list of table names to inspect; None/empty = all tables.

    Returns:
        dict with keys ``schemas`` (list of table dicts) and ``ddl_summary`` (str).
    """
    engine = _get_engine()
    inspector = inspect(engine)
    all_tables = inspector.get_table_names()

    target_tables = [t for t in (tables or []) if t in all_tables] or all_tables

    schemas = []
    ddl_parts: list[str] = []

    with engine.connect() as conn:
        for table_name in target_tables:
            cols = inspector.get_columns(table_name)
            pk = inspector.get_pk_constraint(table_name)
            fks = inspector.get_foreign_keys(table_name)

            # Row count
            row_count = conn.execute(
                text(f'SELECT COUNT(*) FROM "{table_name}"')
            ).scalar()

            # Sample distinct values for key categorical columns
            sample_values: dict[str, list] = {}
            categorical_cols = [
                c["name"]
                for c in cols
                if "char" in str(c["type"]).lower() or "text" in str(c["type"]).lower()
            ]
            for col in categorical_cols[:6]:
                try:
                    rows = conn.execute(
                        text(
                            f'SELECT DISTINCT "{col}" FROM "{table_name}" '
                            f'WHERE "{col}" IS NOT NULL LIMIT 8'
                        )
                    ).fetchall()
                    sample_values[col] = [r[0] for r in rows]
                except Exception:
                    pass

            col_defs = []
            for c in cols:
                nullable = "" if c.get("nullable", True) else " NOT NULL"
                col_defs.append(f'  "{c["name"]}" {c["type"]}{nullable}')

            pk_line = ""
            if pk and pk.get("constrained_columns"):
                pk_line = (
                    f',\n  PRIMARY KEY ({", ".join(pk["constrained_columns"])})'
                )

            fk_lines = ""
            for fk in fks:
                cols_str = ", ".join(fk["constrained_columns"])
                ref_cols = ", ".join(fk["referred_columns"])
                fk_lines += (
                    f',\n  FOREIGN KEY ({cols_str}) '
                    f'REFERENCES {fk["referred_table"]}({ref_cols})'
                )

            ddl = (
                f'CREATE TABLE "{table_name}" (\n'
                + ",\n".join(col_defs)
                + pk_line
                + fk_lines
                + "\n);"
            )
            ddl_parts.append(ddl)

            schemas.append(
                {
                    "table_name": table_name,
                    "columns": [
                        {
                            "name": c["name"],
                            "data_type": str(c["type"]),
                            "nullable": c.get("nullable", True),
                        }
                        for c in cols
                    ],
                    "row_count": row_count,
                    "sample_values": sample_values,
                }
            )

    ddl_summary = "\n\n".join(ddl_parts)
    return {"schemas": schemas, "ddl_summary": ddl_summary}


def get_schema_context(tables: list[str] | None = None) -> str:
    """
    Return a compact schema context string suitable for injection into an LLM prompt.

    Args:
        tables: optional list of table names.

    Returns:
        Multi-line string with DDL + sample values.
    """
    result = inspect_schema(tables)
    lines = ["## Database Schema (PostgreSQL)\n", result["ddl_summary"], ""]

    for schema in result["schemas"]:
        if schema["sample_values"]:
            lines.append(f"### Sample values — {schema['table_name']}")
            for col, vals in schema["sample_values"].items():
                vals_str = ", ".join(f"'{v}'" for v in vals)
                lines.append(f"  - {col}: {vals_str}")
            lines.append("")

    return "\n".join(lines)
