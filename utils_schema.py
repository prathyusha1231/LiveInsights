"""
NL→SQL: Schema-aware prompting for multi-table queries (with better joins)

This file gives you:
1) Lightweight schema introspection from loaded DataFrames
2) Join-key detection (heuristics + scores)
3) A compact schema summary string for LLM prompting
4) A schema-aware prompt template with join guidance & few-shots
5) Post-generation SQL sanity checks (avoid Cartesian joins, enforce LIMIT, column validation)

Designed for: DuckDB/SQLite/Postgres-ish SQL. Keep dialect-neutral where possible.
Dependencies: pandas, sqlparse (optional but recommended)
"""
from __future__ import annotations
import pandas as pd
import re
from typing import Dict, List, Tuple

try:
    import sqlparse 
except Exception:  
    sqlparse = None


def profile_dataframe(df: pd.DataFrame) -> Dict:
    cols = []
    for c in df.columns:
        s = df[c]
        nunique = int(s.nunique(dropna=True))
        null_pct = float(s.isna().mean()) if len(s) else 0.0
        dtype = str(s.dtype)
        card = (
            "unique" if nunique == len(s) else
            "high" if nunique/ max(1, len(s)) > 0.5 else
            "medium" if nunique/ max(1, len(s)) > 0.1 else
            "low"
        )
        cols.append({
            "name": c,
            "dtype": dtype,
            "nunique": nunique,
            "null_pct": round(null_pct, 4),
            "cardinality": card
        })
    return {"n_rows": int(len(df)), "n_cols": int(len(df.columns)), "columns": cols}


def infer_primary_keys(df: pd.DataFrame, table: str) -> List[str]:
    """Heuristics: exact-unique int keys; name patterns: id, <table>_id."""
    cands = []
    lname = table.lower()
    for c in df.columns:
        s = df[c]
        if s.isna().any():
            continue
        if s.nunique(dropna=False) == len(s):
            if re.fullmatch(r".*_id|id", c.lower()) or c.lower() in {f"{lname}_id"}:
                cands.append(c)
            elif pd.api.types.is_integer_dtype(s):
                cands.append(c)
    cands = sorted(set(cands), key=lambda x: (0 if re.fullmatch(r".*_id|id", x.lower()) else 1, x))
    return cands[:2]


def infer_foreign_keys(df_map: Dict[str, pd.DataFrame]) -> List[Tuple[str, str, str, str, float]]:
    """
    Returns [(left_table, left_col, right_table, right_col, score)]
    Scoring mix:
      - column-name similarity
      - RHS is unique-like (candidate PK)
      - dtype compatibility
    """

    pk_map: Dict[str, List[str]] = {}
    dtypes: Dict[Tuple[str, str], str] = {}
    for t, df in df_map.items():
        pk_map[t] = infer_primary_keys(df, t)
        for c in df.columns:
            dtypes[(t, c)] = str(df[c].dtype)

    fks = []
    for lt, ldf in df_map.items():
        for lc in ldf.columns:
            lc_low = lc.lower()
            for rt, rdf in df_map.items():
                if lt == rt:
                    continue
                for rc in pk_map.get(rt, []) or rdf.columns.tolist():
                    score = 0.0
                    if lc_low == f"{rt.lower()}_id" or lc_low == "id":
                        score += 0.7
                    if lc_low.endswith("_id") and (rc.lower().endswith("_id") or rc.lower()=="id"):
                        score += 0.3
                    if rt.lower() in lc_low:
                        score += 0.5
                    if dtypes[(lt, lc)] == dtypes[(rt, rc)]:
                        score += 0.3
                    if rc in infer_primary_keys(rdf, rt):
                        score += 0.5
                    if score >= 0.8:
                        fks.append((lt, lc, rt, rc, round(score, 3)))
    
    best: Dict[Tuple[str, str], Tuple[str, str, float]] = {}
    for lt, lc, rt, rc, sc in fks:
        k = (lt, lc)
        if k not in best or sc > best[k][2]:
            best[k] = (rt, rc, sc)
    out = [(lt, lc, rt, rc, sc) for (lt, lc), (rt, rc, sc) in best.items()]
    return sorted(out, key=lambda x: (-x[4], x[0], x[1]))


def build_schema_summary(df_map: Dict[str, pd.DataFrame]) -> str:
    lines = []
    for t, df in df_map.items():
        prof = profile_dataframe(df)
        pks = infer_primary_keys(df, t)
        col_lines = ", ".join([f"{c['name']}:{c['dtype']}" for c in prof["columns"]])
        lines.append(f"TABLE {t} (rows={prof['n_rows']}, pk~{pks}): {col_lines}")

    fks = infer_foreign_keys(df_map)
    if fks:
        lines.append("FOREIGN-KEY CANDIDATES:")
        for lt, lc, rt, rc, sc in fks:
            lines.append(f"  {lt}.{lc} -> {rt}.{rc} (score={sc})")
    return "\n".join(lines)



SCHEMA_AWARE_TEMPLATE = """
You translate a business question into a SINGLE, syntactically valid SQL query.

Rules:
- Target SQL dialect: DuckDB-compatible.
- Use ONLY the tables/columns from the provided schema.
- Prefer INNER JOIN when filtering to matching records; use LEFT JOIN only if the question implies missing matches.
- Select the MINIMAL set of columns required to answer.
- If aggregation is used, include GROUP BY only for the non-aggregated selected columns.
- Never create CROSS JOINs; always join with ON conditions using discovered keys.
- For date filters, prefer BETWEEN or DATE_TRUNC where applicable.
- Always include a LIMIT 200 unless the question asks for counts/aggregates only.
- Aliases must be short and readable (c, o, p...).

Dates & casting (IMPORTANT):
- CSV date columns are often TEXT/VARCHAR. **Always** `CAST(<alias>.<date_col> AS DATE)` before:
  - comparing in WHERE / BETWEEN
  - using DATE_TRUNC / EXTRACT / date arithmetic
- Use half-open ranges for periods: `[start, end)` e.g. `>= DATE '2019-10-01' AND < DATE '2020-01-01'`.


Join guidance:
- Prefer keys ending with `_id` or `{{table}}.id`.
- If multiple join paths exist, choose the shortest path connecting the necessary tables.
- Validate that join keys have compatible types (e.g., int with int).

Return ONLY one SQL statement ending with a semicolon. No markdown fences, comments, or explanations.


SCHEMA:
{schema}

EXAMPLES:
Q: Which customers placed the most orders in 2024?
SELECT c.customer_id, c.name, COUNT(*) AS order_count
FROM customers c
JOIN orders o ON o.customer_id = c.customer_id
WHERE CAST(o.order_date AS DATE) >= DATE '2024-01-01'
  AND CAST(o.order_date AS DATE) <  DATE '2025-01-01'
GROUP BY c.customer_id, c.name
ORDER BY order_count DESC
LIMIT 200;


Q: Total revenue by product category last quarter.
WITH oo AS (
  SELECT *, CAST(order_date AS DATE) AS order_date_d
  FROM orders
)
SELECT p.category, SUM(oi.quantity * oi.unit_price) AS revenue
FROM order_items oi
JOIN products p ON p.product_id = oi.product_id
JOIN oo ON oo.order_id = oi.order_id
WHERE oo.order_date_d >= DATE_TRUNC('quarter', CURRENT_DATE - INTERVAL '1 quarter')
  AND oo.order_date_d <  DATE_TRUNC('quarter', CURRENT_DATE)
GROUP BY p.category
ORDER BY revenue DESC
LIMIT 200;


Q: List orders with customer names where a promo code was used.

SELECT o.order_id, c.name AS customer_name, o.promo_code
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
WHERE o.promo_code IS NOT NULL
ORDER BY o.order_id
LIMIT 200;


Q: Average revenue by customer age group.
SELECT
  CASE
    WHEN c.cust_age BETWEEN 18 AND 25 THEN '18-25'
    WHEN c.cust_age BETWEEN 26 AND 35 THEN '26-35'
    WHEN c.cust_age BETWEEN 36 AND 45 THEN '36-45'
    WHEN c.cust_age BETWEEN 46 AND 60 THEN '46-60'
    ELSE '60+'
  END AS age_group,
  AVG(s.product_quantity * p.product_price) AS avg_revenue
FROM sales_transactions s
JOIN customer_dim c ON c.cust_id = s.cust_id
JOIN product_dim p   ON p.product_id = s.product_id
WHERE p.current_ind = 'Y'
GROUP BY age_group
ORDER BY age_group;

USER QUESTION:
{question}
""".strip()


def enforce_limit(sql: str, default_limit: int = 200) -> str:
    q = sql.strip().rstrip(';')
    if re.search(r"\bcount\s*\(\s*\*\s*\)", q, re.I):
        return q + ";" 
    if re.search(r"\blimit\b\s*\d+", q, re.I):
        return q + ";"
    return q + f"\nLIMIT {default_limit};"


def detect_cartesian(sql: str) -> bool:
    
    if "," in re.split(r"\bfrom\b", sql, flags=re.I)[-1].split(";")[0]:
        return True
    joins = re.findall(r"\bjoin\b", sql, flags=re.I)
    ons = re.findall(r"\bon\b", sql, flags=re.I)
    return len(joins) > 0 and len(ons) < len(joins)


def validate_columns(sql: str, df_map: Dict[str, pd.DataFrame]) -> Tuple[bool, List[str]]:
    
    if sqlparse is None:
        return True, []
    parsed = sqlparse.parse(sql)
    known_cols = set()
    for t, df in df_map.items():
        for c in df.columns:
            known_cols.add((t.lower(), c.lower()))
    unknown = []
    for stmt in parsed:
        for token in stmt.flatten():
            if token.ttype is None and "." in token.value:
                parts = token.value.split(".")
                if len(parts) == 2:
                    ta, ca = parts[0].strip('"').lower(), parts[1].strip('"').lower()
                    if (ta, ca) not in known_cols:
                        unknown.append(token.value)
    return (len(unknown) == 0), unknown


class SchemaAwareNL2SQL:
    def __init__(self, tables: Dict[str, pd.DataFrame]):
        self.tables = tables
        self.schema_str = build_schema_summary(tables)

    def build_prompt(self, question: str) -> str:
        notes = dynamic_business_notes(self.tables)
        schema_plus = self.schema_str + (("\n\nBusiness-hints:\n" + notes) if notes else "")
        return SCHEMA_AWARE_TEMPLATE.format(schema=schema_plus, question=question.strip())


    def postprocess(self, sql: str) -> Tuple[str, List[str]]:
        flags = []
        if detect_cartesian(sql):
            flags.append("Possible cartesian join detected – ensure ON conditions are present.")
        sql2 = enforce_limit(sql)
        ok, unknown = validate_columns(sql2, self.tables)
        if not ok:
            flags.append(f"Unknown columns referenced: {sorted(set(unknown))}")
        return sql2, flags


if __name__ == "__main__":
    customers = pd.DataFrame({
        "customer_id": [1,2],
        "name": ["Ana","Ben"],
        "region": ["NE","SW"],
    })
    orders = pd.DataFrame({
        "order_id": [10,11],
        "customer_id": [1,2],
        "order_date": ["2024-06-01","2024-07-10"],
        "promo_code": [None, "SUMMER10"],
    })
    order_items = pd.DataFrame({
        "order_id": [10,11],
        "product_id": [100,101],
        "quantity": [2,1],
        "unit_price": [30.0, 50.0],
    })
    products = pd.DataFrame({
        "product_id": [100,101],
        "category": ["Books","Games"],
    })

    n2s = SchemaAwareNL2SQL({
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
        "products": products,
    })

    q = "Which customers used a promo code in their orders?"
    prompt = n2s.build_prompt(q)
    print("\n===== PROMPT SENT TO LLM =====\n")
    print(prompt[:1000] + ("..." if len(prompt) > 1000 else ""))

 
    sql_text = """
    SELECT o.order_id, c.name, o.promo_code
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.promo_code IS NOT NULL
    ORDER BY o.order_id
    LIMIT 200;
    """.strip()

    final_sql, flags = n2s.postprocess(sql_text)
    print("\n===== FINAL SQL =====\n", final_sql)
    if flags:
        print("\nFLAGS:", flags)


def dynamic_business_notes(df_map):
    cols = {t: {c.lower() for c in df.columns} for t, df in df_map.items()}
    def has(col): return any(col in s for s in cols.values())

    notes = []
    # revenue pattern
    if has("product_quantity") and has("product_price"):
        notes.append("- If revenue is implied, compute it as product_quantity * product_price.")
    # current rows
    if has("current_ind"):
        notes.append("- If a dimension has current_ind, filter to current rows: <alias>.current_ind = 'Y'.")
    # dates
    if has("order_date"):
        notes.append("- For date filters/grouping, CAST(text_date AS DATE) and use DATE_TRUNC when needed.")
    # age buckets
    if has("cust_age"):
        notes.append("- If the question mentions age groups, bucket cust_age into '18-25','26-35','36-45','46-60','60+' via CASE and group by age_group.")
    return "\n".join(notes)
