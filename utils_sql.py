import os, re
from typing import Tuple

SYSTEM_PROMPT = (
    "You are a meticulous data analyst that writes a single DuckDB SQL SELECT query. "
    "Rules: One statement. No semicolons. No INSERT/UPDATE/DELETE/DDL. "
    "Always LIMIT to 500 rows. Alias aggregates as value."
    "If the column is stored as a string but looks like a date, always cast it to DATE using CAST(column AS DATE)"
)

def safe_sql(sql: str, allowed_joins=None) -> Tuple[bool, str]:
    """
    Check SQL safety, optionally allowing whitelisted joins.
    """
    s = sql.strip()
    if ";" in s:
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
    """
    Call OpenAI API to generate SQL safely.
    """
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
    return m.group(1).strip() if m else text
