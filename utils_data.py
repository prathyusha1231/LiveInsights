import io, csv, os
import pandas as pd

def load_table(uploaded_file) -> pd.DataFrame:
    """
    Load a CSV or Excel file into a DataFrame with robust encoding handling.
    """
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    try:
        sample = raw[:4096].decode("latin1", errors="ignore")
        dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"])
        sep = dialect.delimiter
    except Exception:
        sep = None

    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, sep=sep)
        except Exception:
            continue

    text = raw.decode("latin1", errors="replace")
    return pd.read_csv(io.StringIO(text), sep=sep)


def schema_hint(df: pd.DataFrame, sample_rows: int = 5) -> str:
    """
    Generate schema hint for single-table mode.
    """
    cols = [f"- {c}: {df[c].dtype}" for c in df.columns]
    head = df.head(sample_rows).to_dict(orient="records")
    return "Columns:\n" + "\n".join(cols) + "\n\nSample rows:\n" + str(head)


def schema_hint_with_joins(dfs: dict, allowed_joins: list[str]) -> str:
    """
    Generate schema hint for multi-table mode with join keys.
    """
    out = ["Tables available:"]
    for t, df in dfs.items():
        cols = ", ".join(df.columns)
        out.append(f"- {t}({cols})")

    if allowed_joins:
        out.append("\nAllowed joins:")
        for j in allowed_joins:
            out.append(f"- {j}")
    return "\n".join(out)


def detect_join_keys(dfs: dict) -> list[str]:
    """
    Detect potential join keys based on identical column names across tables.
    """
    joinable = []
    cols_map = {t: set(df.columns) for t, df in dfs.items()}
    for t1, cols1 in cols_map.items():
        for t2, cols2 in cols_map.items():
            if t1 < t2:
                common = cols1 & cols2
                for col in common:
                    joinable.append(f"{t1}.{col} = {t2}.{col}")
    return joinable
