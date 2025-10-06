# Live Insights (NL → SQL with Streamlit + DuckDB + OpenAI)

A lightweight Streamlit app that lets you **upload a CSV/Excel file, ask questions in plain English, and get answers as SQL queries + results + charts**.  
If no API key is provided, the app falls back to a heuristic NL→SQL generator.

---

## Features
- Upload **CSV/Excel** files (handles multiple encodings and delimiters).
- Ask **natural language questions** → get back SQL queries (DuckDB).
- **Safe SQL sandboxing** (only single `SELECT` queries allowed).
- **Heuristic fallback** if no LLM is available.
- **Auto-generated charts** (line, bar, or metric) based on query result.
- **Auto Insights**: LLM summarizes results into plain English insights  

---

## Updated Features
- Multi-file upload (CSV/Excel) with robust parsing and DuckDB registration.
- Join-aware NL→SQL (casts text dates, uses CTEs for intermediate steps).
- Safe SQL: single-statement `SELECT` only, auto-`LIMIT 200`, cartesian-join warning.
- Name enrichment in results (e.g., `product_name` next to `product_id`).
- Auto visuals: KPI for single value, line for time, bar for categories.
- Auto Insights (LLM): short bullets, prefers human-friendly names over IDs.
- EXPLAIN-first execution to surface errors early.

---


## Project Structure
```

├── app_LLM_v1.py              
├── app_llm_joins.py        
├── utils_data.py           
├── utils_sql.py            
├── requirements.txt 
├── sample_questions.txt 
├── sample_questions_mulltitable.txt
├── .env
├── .gitignore 
└── README.md 
```
---

## Setup & Installation
 1. Clone the repository.

 2. Create a virtual environment:
    python -m venv .venv

    Activate it:
      - Windows (PowerShell):
        .venv\Scripts\Activate.ps1
      - Mac/Linux:
        source .venv/bin/activate

 3. Install dependencies:
    pip install -r requirements.txt

 4. Set up environment variables:
    Create a `.env` file in the project root and add:
      OPENAI_API_KEY=your_api_key_here
      OPENAI_MODEL=gpt-4o-mini

 5. Run the app:
    streamlit run app_llm.py


## Working On (Future Roadmap)

 - Multi-file joins (support multiple CSV/Excel uploads)
 - Richer visualizations (pie charts, heatmaps, stacked bars)
 - Query history + downloadable dashboards
 - Deployment on Streamlit Cloud / Hugging Face Spaces
 - Built-in anomaly detection & auto-insights expansion
