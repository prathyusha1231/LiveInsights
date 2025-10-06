import os, re
from typing import Tuple
import duckdb

from utils_schema import SchemaAwareNL2SQL

SYSTEM_PROMPT = (
    "You are a meticulous data analyst who writes exactly one DuckDB SQL SELECT query.\n"
    "Output rules:\n"
    "- Output exactly one SQL statement ending with a semicolon. No comments, no markdown, no extra text.\n"
    "- No DDL/DML (no CREATE/DROP/INSERT/UPDATE/DELETE).\n"
    "- Use short, readable table aliases.\n"
    "- If a column looks like a date but is TEXT, use CAST(column AS DATE).\n"
    "- Include LIMIT 200 unless the question is a pure aggregate (COUNT/SUM/AVG without detail rows).\n"
)

def safe_sql(sql: str, allowed_joins=None) -> Tuple[bool, str]:
    """
    Check SQL safety, optionally allowing whitelisted joins.
    """
    s = sql.strip()
    statements = [stmt.strip() for stmt in s.strip().split(";") if stmt.strip()]
    if len(statements) > 1:
        return False, "Multiple statements not allowed."
    if not re.match(r"^select\s", s, re.I):
        return False, "Only SELECT queries are allowed."

    forbidden = ["insert", "update", "delete", "drop", "alter", "create"]
    if any(re.search(fr"\b{kw}\b", s, re.I) for kw in forbidden):
        return False, "DDL/DML not allowed."

    if allowed_joins and "join" in s.lower():
        if not any(j.lower() in s.lower() for j in allowed_joins):
            return False, f"Join not permitted. Allowed joins: {allowed_joins}"

    return True, "OK"


def llm_call(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    text = resp.choices[0].message.content.strip()
    text = re.sub(r"^```[\w-]*\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"^\s*(sql|query)\s*:\s*", "", text, flags=re.IGNORECASE)

    m = re.search(r"(?is)(select\s.+)$", text)
    out = m.group(1).strip() if m else text.strip()

    parts = re.split(r";\s*(?=select)", out, flags=re.IGNORECASE)
    out = parts[0].strip()
    if not out.endswith(";"):
        out += ";"
    return out




def build_schema_aware_prompt(df_map, user_question: str) -> str:
    """
    df_map is a dict[str, pandas.DataFrame] for all currently loaded tables.
    """
    n2s = SchemaAwareNL2SQL(df_map)
    return n2s.build_prompt(user_question)


def postprocess_generated_sql(df_map, sql_text: str):
    n2s = SchemaAwareNL2SQL(df_map)
    final_sql, flags = n2s.postprocess(sql_text)
    return final_sql, flags


def run_sql_with_explain(con: duckdb.DuckDBPyConnection, sql_text: str):
    # quick EXPLAIN to surface obvious issues before executing
    try:
        con.execute(f"EXPLAIN {sql_text}")
    except Exception as e:
        raise RuntimeError(f"Query failed EXPLAIN check: {e}") from e
    return con.execute(sql_text).fetchdf()


