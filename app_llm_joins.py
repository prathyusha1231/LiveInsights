import re, os
import duckdb, streamlit as st
from dotenv import load_dotenv
from utils_data import load_table
from utils_sql import (
    safe_sql,
    llm_call,
    build_schema_aware_prompt,
    postprocess_generated_sql,
    run_sql_with_explain,
)
from utils_schema import build_schema_summary
from pandas.api.types import is_datetime64_any_dtype




load_dotenv()


st.set_page_config(page_title="Live Insights (Joins)", page_icon="🔗", layout="wide")
st.title("Live Insights with Multi-Table Support")


uploaded_files = st.file_uploader("Upload multiple CSV/Excel files",
    type=["csv","xlsx","xls"], accept_multiple_files=True)
if not uploaded_files:
    st.stop()

con = duckdb.connect()
dfs = {}
for f in uploaded_files:
    df = load_table(f)
    name = f.name.split(".")[0].lower().replace(" ","_")
    dfs[name] = df
    con.register(name, df)


with st.expander("Detected schema & join hints"):
    st.code(build_schema_summary(dfs))

q = st.text_input("Ask a question (e.g., 'Revenue by category')")
if not q:
    st.stop()

prompt = build_schema_aware_prompt(dfs, q)

try:
    candidate = llm_call(prompt) 
    final_sql, flags = postprocess_generated_sql(dfs, candidate)

    is_ok, msg = safe_sql(final_sql)
    if not is_ok:
        raise RuntimeError(f"Rejected SQL: {msg}")
    sql = final_sql

except Exception as e:
    st.error(f"Error generating SQL: {e}")
    st.stop()

st.markdown("**Proposed SQL**")
st.code(sql, language="sql")
if flags:
    st.warning(" • " + "\n • ".join(flags))



def _enrich_with_names(df_result, dfs):
    """
    If the result has ID columns (e.g., product_id, cust_id) and the uploaded
    data includes the corresponding dimensions, attach readable names.

    - product_id -> product_dim.product_name (current_ind='Y' if present)
    - cust_id / customer_id -> customer_dim.<name-like column> (best effort)
    """
    import pandas as pd

    if df_result is None or df_result.empty:
        return df_result

    out = df_result.copy()

    # Product name enrichment
    if "product_id" in out.columns and "product_dim" in dfs:
        p_dim = dfs["product_dim"]
        # Choose columns safely
        cols = [c for c in ["product_id", "product_name", "current_ind"] if c in p_dim.columns]
        if cols:
            dim = p_dim[cols].copy()
            # keep only current rows if column exists
            if "current_ind" in dim.columns:
                dim = dim[dim["current_ind"].astype(str).str.upper().eq("Y")]
            # keep only id + name if name exists
            if "product_name" in dim.columns:
                dim = dim[["product_id", "product_name"]].drop_duplicates("product_id")
                out = out.merge(dim, on="product_id", how="left")
                # move product_name right after product_id
                cols_out = list(out.columns)
                if "product_name" in cols_out:
                    cols_out.insert(cols_out.index("product_id") + 1, cols_out.pop(cols_out.index("product_name")))
                    out = out[cols_out]

    # Customer name enrichment 
    cust_id_candidates = [c for c in ["cust_id", "customer_id"] if c in out.columns]
    if cust_id_candidates and "customer_dim" in dfs:
        c_dim = dfs["customer_dim"].copy()
        # pick a reasonable name-like column 
        name_candidates = [n for n in ["customer_name", "cust_name", "full_name", "name"] if n in c_dim.columns]
        if name_candidates:
            name_col = name_candidates[0]
            for cid in cust_id_candidates:
                if cid in c_dim.columns:
                    dim = c_dim[[cid, name_col]].drop_duplicates(cid)
                else:
                    if cid == "cust_id" and "customer_id" in c_dim.columns:
                        dim = c_dim[["customer_id", name_col]].rename(columns={"customer_id": "cust_id"}).drop_duplicates("cust_id")
                    elif cid == "customer_id" and "cust_id" in c_dim.columns:
                        dim = c_dim[["cust_id", name_col]].rename(columns={"cust_id": "customer_id"}).drop_duplicates("customer_id")
                    else:
                        dim = None
                if dim is not None:
                    out = out.merge(dim, on=cid, how="left")
                    cols_out = list(out.columns)
                    if name_col in cols_out:
                        cols_out.insert(cols_out.index(cid) + 1, cols_out.pop(cols_out.index(name_col)))
                        out = out[cols_out]

    return out



# ------- Auto Insights -------
def generate_insights_joins(df_result, question, schema_text: str):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "No API key found. Add OPENAI_API_KEY in your .env to enable insights."

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    table_preview = df_result.head(50).to_csv(index=False)

    prompt = f"""
You are a data analyst. The user asked: "{question}".

The query result (up to 50 rows):
{table_preview}

Relevant schema (abbreviated):
{schema_text}

Please summarize in 2–3 short bullet points in clean Markdown:
- Call out top contributors, spikes or outliers if visible.
- Keep it crisp and business-friendly.
"""

    try:
        resp = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
            input=[
                {"role": "system", "content": "You are a data insights assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        text = resp.output[0].content[0].text if getattr(resp, "output", None) else resp.to_dict().get("output_text", "")
        return text.strip()
    except Exception as e:
        return f"Insight generation failed: {e}"


# ------- Auto Chart -------
def auto_chart(df_result):

    if df_result is None or df_result.empty:
        return

    df = df_result.copy()

    # choose value column
    val_col = None
    lower = {c.lower(): c for c in df.columns}
    if "value" in lower:
        val_col = lower["value"]
    else:
        nums = df.select_dtypes(include="number").columns.tolist()
        val_col = nums[0] if nums else df.columns[0]

    # choose dimension column
    dim_col = next((c for c in df.columns if c != val_col), None)

    # prefer date-like dimension
    if dim_col:
        if not (is_datetime64_any_dtype(df[dim_col]) or re.search(r"date|time|month|year", dim_col, re.I)):
            for c in df.columns:
                if c == val_col:
                    continue
                if is_datetime64_any_dtype(df[c]) or re.search(r"date|time|month|year", c, re.I):
                    dim_col = c
                    break

    # single-row -> metric
    if len(df) == 1:
        label = dim_col if dim_col else "value"
        try:
            st.metric(label=str(df.iloc[0].get(dim_col, "Result")),
                      value=f"{df[val_col].iloc[0]:,.2f}")
        except Exception:
            st.write(df)
        return

    # time series -> line
    if dim_col and (is_datetime64_any_dtype(df[dim_col]) or re.search(r"date|time|month|year", dim_col, re.I)):
        try:
            st.line_chart(df.set_index(dim_col)[val_col])
            return
        except Exception:
            pass

    # categorical -> bar
    df_sorted = df.sort_values(val_col, ascending=False)
    top = df_sorted.head(20)
    if not top[dim_col].dtype == object:
        top[dim_col] = top[dim_col].astype(str)

    st.bar_chart(top.set_index(dim_col)[val_col])



try:
    result = run_sql_with_explain(con, sql)

    st.success("Results")
    # Auto chart
    enriched_result = _enrich_with_names(result, dfs)

    # Chart (or KPI) from the enriched result
    if enriched_result.shape[0] > 1 and enriched_result.shape[1] > 1:
        st.subheader("Chart")
        auto_chart(enriched_result)
    else:
        st.subheader("KPI")
        auto_chart(enriched_result)

    # Auto insights
    st.subheader("Auto Insights")
    schema_text = build_schema_summary(dfs)
    table_for_llm = enriched_result.copy()
    insights = generate_insights_joins(table_for_llm , q, schema_text) 
    cleaned = insights.replace('**', '')
    st.markdown(cleaned, unsafe_allow_html=False)

    st.dataframe(result, use_container_width=True)
except Exception as e:
    st.error(f"Query failed: {e}")
    st.stop()