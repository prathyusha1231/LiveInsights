# app_llm.py
# Streamlit demo: Upload CSV → ask natural language → LLM proposes SQL → DuckDB runs → results + chart
# .env support included (OPENAI_API_KEY, OPENAI_MODEL)
# If no API key is present, the app gracefully falls back to a heuristic NL→SQL module.

import streamlit as st
import pandas as pd
import duckdb                                                                                   
import io
import os
import re
import csv
from typing import List, Tuple
from datetime import date
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype







# --- .env support ---
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass  

try:
    import openpyxl  
except Exception:
    st.info("For Excel (.xlsx) files, install:  pip install openpyxl")

st.set_page_config(page_title="Live Insights (LLM NL→SQL)", page_icon="📊", layout="wide")
st.title("Live Insights: Ask your Excel in plain English")
st.caption("Upload a CSV → Ask a question → LLM generates SQL (safely) → Results + chart. If no key, uses a heuristic fallback.")


st.caption("Upload a CSV or Excel file")
uploaded = st.file_uploader("Upload a data file", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.info("Tip: CSV delimiters can vary; Excel files are fine too. Encodings like cp1252/latin-1 are supported.")
    st.stop()

def load_table(uploaded_file):
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()

    # Excel
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(raw))

    # CSV
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None

   
    sample = raw[:4096].decode("latin1", errors="ignore")
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=[",", ";", "\t", "|"])
        guessed_sep = dialect.delimiter
    except Exception:
        guessed_sep = None  # let pandas infer

    for enc in encodings:
        try:
            return pd.read_csv(
                io.BytesIO(raw),
                encoding=enc,
                sep=guessed_sep if guessed_sep else None,
                engine="python",            # better with odd encodings/delimiters
                on_bad_lines="skip"         # skip malformed rows instead of failing
            )
        except Exception as e:
            last_err = e

    # Replace undecodable bytes to avoid crash
    try:
        text = raw.decode("latin1", errors="replace")
        return pd.read_csv(
            io.StringIO(text),
            sep=guessed_sep if guessed_sep else None,
            engine="python",
            on_bad_lines="skip"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to read file. Last error: {e or last_err}")

df = load_table(uploaded)


# content = file.read()
# df = pd.read_csv(io.BytesIO(content))

# Attempt to parse likely date columns
for c in df.columns:
    if re.search(r"date|time|_at$", c, re.I):
        try:
            df[c] = pd.to_datetime(df[c], errors="ignore")
        except Exception:
            pass

st.subheader("Preview")
st.dataframe(df.head(10), use_container_width=True)

# Register DuckDB view
con = duckdb.connect()
con.register("data", df)

# ---------- Helpers ----------

def safe_sql(sql: str) -> Tuple[bool, str]:
    """Allow only a single SELECT on `data` (no DDL/DML, joins, semicolons)."""
    s = sql.replace("\ufeff", "").strip()
    if ";" in s:
        return False, "Multiple statements/semicolons are not allowed."
    if not re.match(r"^select\s", s, re.I):
        return False, "Only SELECT queries are allowed."
    forbidden = [r"\bjoin\b", r"\bunion\b", r"\bwith\b", r"\binsert\b", r"\bupdate\b", r"\bdelete\b", r"\bdrop\b", r"\balter\b", r"\bcreate\b"]
    if any(re.search(pat, s, re.I) for pat in forbidden):
        return False, "Joins/CTEs/DDL/DML are not allowed."
    # Must reference only `data`
    m = re.search(r"\bfrom\s+([a-zA-Z0-9_\.]+)", s, re.I)
    if m and m.group(1).lower() != "data":
        return False, "Only the `data` table is permitted."
    return True, "OK"


def schema_hint(df: pd.DataFrame, sample_rows: int = 5) -> str:
    cols = []
    for c in df.columns:
        dtype = str(df[c].dtype)
        cols.append(f"- {c}: {dtype}")
    head = df.head(sample_rows).to_dict(orient="records")
    return ("Columns:\n" + "\n".join(cols) + "\n\n" + 
            "Sample rows (up to 5):\n" + str(head))

SYSTEM_PROMPT = (
    "You are a meticulous data analyst that writes a single DuckDB SQL SELECT query. "
    "You ONLY query a table named `data`. You must never modify data. "
    "Rules: One statement. No semicolons. No JOIN/UNION/WITH/INSERT/UPDATE/DELETE/DDL. "
    "Prefer using provided column names exactly. If time grain is needed, use date_trunc. "
    "If the request is ambiguous, choose a reasonable interpretation and aggregate. "
    "Always LIMIT results to at most 500 rows unless it's a single scalar row."
    "Always alias aggregates as `value` (e.g., SELECT SUM(revenue) AS value). "
)

USER_PROMPT_TEMPLATE = (
    "Question: {question}\n\n"
    "Table: `data`\n"
    "{schema}\n\n"
    "Write ONLY the SQL. No commentary."
)

# ---------- LLM call ----------

def llm_call(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Add it to your .env or environment.")
    try:
        from openai import OpenAI  # pip install openai>=1.0.0
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        # text = resp.choices[0].message.content.strip()
        # Strip fenced code blocks if present
        #text = re.sub(r"^```sql\n|\n```$", "", text)
        
        text = resp.choices[0].message.content

        # Strip ```sql/``` fences and any "SQL:" prefix, then keep the first SELECT
        text = re.sub(r"^```[\w-]*\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r"^\s*(sql|query)\s*:\s*", "", text, flags=re.IGNORECASE)
        m = re.search(r"(?is)(select\s.+)$", text)  # first SELECT to end
        text = m.group(1).strip() if m else text.strip()
        return text
    except Exception as e:
        raise RuntimeError(f"LLM error: {e}")



# ---------- Heuristic fallback ----------

def heuristic_sql(question: str, cols: List[str]) -> str:
    lc = question.lower()
    # Measure
    measure = None
    for target in ["revenue", "sales", "units"]:
        for c in cols:
            if c.lower() == target:
                measure = c
                break
        if measure:
            break
    if not measure:
        measure = cols[0]
    # Group by via "by X"
    group = None
    m = re.search(r"by\s+([a-zA-Z_]+)", lc)
    if m:
        token = m.group(1)
        for c in cols:
            if c.lower() == token or token in c.lower():
                group = c
                break
    
    # Top N
    topn = 5
    m2 = re.search(r"top\s+(\d+)", lc)
    if m2:
        topn = int(m2.group(1))
    # Date filter
    date_col = None
    for c in cols:
        if re.search(r"date|time|_at$", c, re.I):
            date_col = c
            break
    where = ""
    if "this month" in lc and date_col:
        start = pd.Timestamp(date.today()).replace(day=1).strftime("%Y-%m-%d")
        where = f" WHERE {date_col} >= DATE '{start}'"
    if group is None:
        for pat in ["product_name","product","item","sku","category","country","channel","segment"]:
            for c in cols:
                if c.lower() == pat:
                    group = c
                    break
            if group:
                break
    # Build SQL
    if group:
        return f"SELECT SUM({measure}) AS value, {group} FROM data{where} GROUP BY {group} ORDER BY value DESC LIMIT {topn}"
    return f"SELECT SUM({measure}) AS value FROM data{where}"


# ---------- UI: question → SQL ----------
st.subheader("Ask a question")
q = st.text_input("Examples: 'Top 5 products by revenue this month', 'Total revenue by month', 'Average order value by channel'")
if not q:
    st.stop()

schema = schema_hint(df)
prompt = USER_PROMPT_TEMPLATE.format(question=q, schema=schema)

sql = None
llm_error = None
try:
    candidate = llm_call(prompt)
    ok, msg = safe_sql(candidate)
    if not ok:
        raise RuntimeError(f"Rejected SQL: {msg}")
    sql = candidate
except Exception as e:
    llm_error = str(e)
    sql = heuristic_sql(q, df.columns.tolist())

st.markdown("**Proposed SQL**")
st.code(sql, language="sql")
if llm_error:
    st.warning(f"LLM issue: {llm_error}. Used heuristic fallback.")

# ---------- Execute ----------
try:
    result = con.execute(sql).df()
except Exception as e:
    st.error(f"Query failed: {e}")
    st.stop()

st.success("Results:")
st.dataframe(result, use_container_width=True)

## ---------- Chart (auto-pick) ----------
def auto_chart(df_result: pd.DataFrame):
    if df_result is None or df_result.empty:
        return

    df = df_result.copy()

    # Identify value column
    val_col = None
    if "value" in [c.lower() for c in df.columns]:
        val_col = next(c for c in df.columns if c.lower() == "value")
    else:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        val_col = num_cols[0] if num_cols else df.columns[0]

    # Identify dimension column 
    dim_candidates = [c for c in df.columns if c != val_col]
    dim_col = None

    # Prefer explicit date-ish columns
    for c in dim_candidates:
        if is_datetime64_any_dtype(df[c]) or re.search(r"date|time|month|year", c, re.I):
            dim_col = c
            break

    if dim_col is None:
        # First non-numeric
        non_num = df.select_dtypes(exclude="number").columns.tolist()
        non_num = [c for c in non_num if c != val_col]
        if non_num:
            dim_col = non_num[0]

    if dim_col is None and dim_candidates:
        # Fallback to whatever is left
        dim_col = dim_candidates[0]

    # Single-row → metric
    if len(df) == 1:
        label = dim_col if dim_col else "value"
        try:
            st.metric(label=str(df.iloc[0].get(dim_col, "Result")), value=f"{df[val_col].iloc[0]:,.2f}")
        except Exception:
            st.write(df)
        return

    # If dim is numeric but small cardinality, treat as category (bar)
    if dim_col and is_numeric_dtype(df[dim_col]) and df[dim_col].nunique() <= 50:
        df = df.sort_values(val_col, ascending=False)
        df[dim_col] = df[dim_col].astype(str)
        st.bar_chart(df.set_index(dim_col)[val_col])
        return

    # Time series → line
    if dim_col and (is_datetime64_any_dtype(df[dim_col]) or re.search(r"date|time|month|year", dim_col, re.I)):
        try:
            st.line_chart(df.set_index(dim_col)[val_col])
            return
        except Exception:
            pass

    # Categorical → bar (cap to top 20 for readability)
    if dim_col:
        df = df.sort_values(val_col, ascending=False)
        top = df.head(20)
        # If dim is not string, cast for prettier axis labels
        if not top[dim_col].dtype == object:
            top[dim_col] = top[dim_col].astype(str)
        st.bar_chart(top.set_index(dim_col)[val_col])
        return

    # Fallback
    st.write(df)

auto_chart(result)


def generate_insights(df_result: pd.DataFrame, question: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "⚠️ No API key found. Add OPENAI_API_KEY in your .env to enable insights."
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        # Convert result to compact CSV/JSON snippet for context
        table_preview = df_result.head(50).to_csv(index=False)

        prompt = f"""
        You are a data analyst. The user asked: "{question}".
        Here is the query result (sample of up to 50 rows):

        {table_preview}

        Summarize the most important insights in 2-3 short sentences.
        If there are trends over time, compare changes (like month-over-month).
        If categorical, highlight top contributors.
        Keep it concise and easy to understand.
        """

        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": "You are a data insights assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        return f"Insight generation failed: {e}"
    
st.subheader("Auto Insights")
try:
    # pick the value column
    val_col = next((c for c in result.columns if c.lower() == "value"), None)
    if val_col is None:
        numeric_cols = result.select_dtypes(include="number").columns.tolist()
        val_col = numeric_cols[0] if numeric_cols else None

    if val_col:
        dims = [c for c in result.columns if c != val_col]
        if dims:
            dim = dims[0]
            top = result.sort_values(val_col, ascending=False).head(3)
            bullets = "\n".join(
                f"- **{row[dim]}**: {row[val_col]:,.0f}"
                for _, row in top.iterrows()
            )
            total = result[val_col].sum()
            st.markdown(
                f"The top results for **{q.strip()}** are:\n\n{bullets}\n\n"
                f"**Total (shown rows):** {total:,.0f}"
            )
        else:
            st.markdown(f"**Answer:** {result[val_col].iloc[0]:,.0f}")
    else:
        st.caption("(No numeric value column to summarize.)")
except Exception as e:
    st.caption(f"(Couldn't generate auto insights: {e})")




st.caption("Safety: Only single SELECT queries against 'data' are allowed. No joins or data modification.")
