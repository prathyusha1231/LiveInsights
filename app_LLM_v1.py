import duckdb, streamlit as st
from utils_data import load_table, schema_hint
from utils_sql import safe_sql, llm_call

import pandas as pd
import re
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from dotenv import load_dotenv
import os





load_dotenv()


st.set_page_config(page_title="Live Insights", page_icon="📊", layout="wide")
st.title("Live Insights: Ask your data in plain English")

uploaded = st.file_uploader("Upload CSV/Excel file", type=["csv","xlsx","xls"])
if not uploaded:
    st.stop()

df = load_table(uploaded)
st.subheader("Preview")
st.dataframe(df.head(10), use_container_width=True)

con = duckdb.connect()
con.register("data", df)

# Build schema prompt
schema = schema_hint(df)
q = st.text_input("Ask a question (e.g., 'Total revenue by month')")
if not q:
    st.stop()

prompt = f"Question: {q}\n\nTable: data\n{schema}\n\nWrite ONLY the SQL."

try:
    candidate = llm_call(prompt)
    is_ok, msg = safe_sql(candidate)
    if not is_ok:
        raise RuntimeError(f"Rejected SQL: {msg}")
    sql = candidate
except Exception as e:
    st.error(f"Error generating SQL: {e}")
    st.stop()

st.markdown("**Proposed SQL**")
st.code(sql, language="sql")

try:
    result = con.execute(sql).df()
    st.success("Results")
    st.dataframe(result, use_container_width=True)
except Exception as e:
    st.error(f"Query failed: {e}")
    st.stop()




## ------ Auto Insights -------


def generate_insights(df_result: pd.DataFrame, question: str, df: pd.DataFrame) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "No API key found. Add OPENAI_API_KEY in your .env to enable insights."
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        table_preview = df_result.head(50).to_csv(index=False)

        # Build ID → name/category mapping if available
        mapping_info = []
        cols = {c.lower(): c for c in df.columns}  # normalize 

        if "product_name" in cols:
            mapping_sample = df[[cols["product_id"], cols["product_name"]]].drop_duplicates().head(50)
            mapping_info.append("Here are sample Product_ID to Product_Name mappings:\n" + mapping_sample.to_string(index=False))

        if "product_category" in cols:
            mapping_sample = df[[cols["product_id"], cols["product_category"]]].drop_duplicates().head(50)
            mapping_info.append("Here are sample Product_ID to Product_Category mappings:\n" + mapping_sample.to_string(index=False))



        extra_context = "\n".join(mapping_info)

        prompt = f"""
        You are a data analyst. The user asked: "{question}".

        Here is the query result (up to 50 rows):

        {table_preview}

        {extra_context}

        Please summarize in 2–3 short bullet points using clean Markdown:
            - Mention products using both Product_ID and their Product_Name/Category if available.
            - Highlight top contributors and quantities/revenues.
            - Keep it clear, professional, and business-friendly.
            - Format the answer clearly in Markdown, using bullet points or bold text where appropriate, and keep it to 1–2 short sentences.

        """

        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a data insights assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        return f"Insight generation failed: {e}"



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
        
        if not top[dim_col].dtype == object:
            top[dim_col] = top[dim_col].astype(str)
        st.bar_chart(top.set_index(dim_col)[val_col])
        return

    # Fallback
    st.write(df)

auto_chart(result)

st.subheader("Auto Insights")
insights = generate_insights(result, q, df)
cleaned = re.sub(r'[*_]{2,}', '', insights)
st.markdown(cleaned, unsafe_allow_html=False)
