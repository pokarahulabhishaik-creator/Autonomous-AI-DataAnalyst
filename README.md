# 🤖 AutoAnalyst AI
### *Autonomous AI Data Analyst & Intelligent Dashboard Generator*

AutoAnalyst AI is a production-quality, autonomous data analytics system. Unlike superficial chatbots that guess numbers or static BI dashboards that cannot adapt, AutoAnalyst AI serves as a **true autonomous data analyst**: the LLM acts as the reasoning and planning layer, while **every single calculation, metric, statistical test, machine learning model, and visualization is computed deterministically** using Python, Pandas, SQLite, SciPy, and Scikit-learn.

> 🛡️ **Zero-Hallucination Guarantee:** The system never invents numbers. Every metric, correlation, outlier, and insight is mathematically derived from the real uploaded data.

---

## 🌟 Key Highlights & Capabilities

1. **Autonomous Dataset Profiling & Scoring:**
   - Detects dimensions, numerical, categorical, and datetime columns.
   - Computes statistical distributions (mean, median, IQR, skewness, outliers via 1.5x IQR).
   - Generates an automated **Data Quality Score (0–100%)** assessing completeness, uniqueness, and validity.
2. **Non-Destructive Autonomous Cleaning:**
   - Detects missing values, duplicate rows, currency/percentage string types, and extreme outliers.
   - Creates a cleaned working copy with a complete audit trail (never mutates raw data).
3. **Multi-Agent Reasoning Architecture:**
   - **Question Analyzer:** Maps user questions to arbitrary column schemas without hardcoding column names.
   - **Analysis Planner:** Formulates multi-step execution plans with testable hypotheses.
   - **Autonomous Analyst Agent:** Executes calculations across Python, SQL, Statistics, and ML tools.
   - **Insight Agent:** Distills results into Four Grounded Pillars (Observed Facts, Statistical Findings, AI Interpretations, Recommendations).
   - **Report Agent:** Compiles publication-ready reports.
4. **Dynamic, Question-Specific Dashboards:**
   - Automatically adapts charts and KPI cards to the specific analytical question asked.
   - Interactive Plotly visualizations (Line trends, Grouped bars, Distributions, Scatters, Composition pies).
   - Dynamic drill-down filter controls and data slice downloads.
5. **RAG Domain Knowledge Base:**
   - Vector knowledge base (ChromaDB + sentence-transformers / TF-IDF) embedding statistical rules (e.g. *Correlation does not imply causation*, Simpson's paradox, p-value thresholds, KPI formulas).
6. **Executive Multi-Format Exports:**
   - 📄 **PDF Report:** Formatted with ReportLab (Executive Summary, KPI tables, Audit findings).
   - 📝 **Word Report (.docx):** Formatted with python-docx.
   - 📊 **Excel Workbook (.xlsx):** Multi-tab structured workbook with openpyxl.
   - 💾 **Cleaned Data Export (CSV).**

---

## 🏗️ System Architecture

```
User Query + CSV / Excel Dataset
               │
               ▼
   [ Autonomous Data Profiler ] ──► Computes Quality Score & Distributions
               │
               ▼
   [ Autonomous Data Cleaner ]  ──► Non-destructive working copy + Audit trail
               │
               ▼
   [ Question Analyzer Agent ]  ──► Extracts intent & maps semantic columns
               │
               ▼
   [ Analysis Planner Agent ]   ──► Builds multi-tool hypothesis plan
               │
    ┌──────────┴──────────────────────────────┐
    ▼                                         ▼
[ Analytics RAG Knowledge Base ]       [ Tool Execution Layer ]
 (ChromaDB Statistical Rules)           • Python Pandas Tool
                                        • SQLite In-Memory Tool
                                        • SciPy Statistics Tool
                                        • Scikit-learn ML & Anomaly Tool
                                        • Plotly Visualization Tool
                                              │
                                              ▼
                                 [ Four-Pillar Insight Agent ]
                                  1. Observed Facts (Computed)
                                  2. Statistical Findings (p-values)
                                  3. AI Interpretations (Mechanisms)
                                  4. Strategic Recommendations
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        [ Dynamic Streamlit Dashboard ]                 [ Multi-Format Report Generator ]
         • KPI Metric Cards                              • PDF Report (ReportLab)
         • Interactive Plotly Charts                     • Word (.docx) Report
         • Interactive Filter & Slice                    • Multi-Tab Excel Workbook
```

---

## 📁 Project Structure

```
autonomous-data-analyst/
│
├── frontend/
│   └── app.py                     # Streamlit web application interface
│
├── backend/
│   ├── agents/
│   │   ├── analyst_agent.py       # Core execution orchestrator
│   │   ├── question_analyzer.py   # Intent & column semantic mapper
│   │   ├── planner.py             # Multi-step analysis planner
│   │   ├── insight_agent.py       # Four-pillar insight generator
│   │   └── report_agent.py        # Report compiler
│   │
│   ├── tools/
│   │   ├── data_profiler.py       # Profiling & quality scoring
│   │   ├── data_cleaner.py        # Non-destructive data cleaning
│   │   ├── python_tool.py         # Sandboxed analytical Python runner
│   │   ├── sql_tool.py            # Analytical SQLite (SELECT-only) runner
│   │   ├── statistics_tool.py     # Descriptive, ANOVA, t-tests, correlations
│   │   ├── ml_tool.py             # Regression, Classification, Clustering, Forecasting
│   │   ├── anomaly_tool.py        # Isolation Forest anomaly detection
│   │   └── visualization_tool.py  # Plotly chart generator
│   │
│   ├── dashboard/
│   │   └── dashboard_generator.py # Question-tailored dynamic dashboard
│   │
│   ├── reports/
│   │   ├── pdf_report.py          # ReportLab PDF generator
│   │   ├── docx_report.py         # python-docx generator
│   │   └── excel_report.py        # openpyxl multi-tab Excel generator
│   │
│   ├── rag/
│   │   ├── knowledge_base.py      # Statistical & KPI guidelines
│   │   ├── vector_store.py        # ChromaDB & fallback vector store
│   │   ├── ingest.py              # Ingestion pipeline
│   │   └── retriever.py           # Top-k guideline retriever
│   │
│   ├── llm/
│   │   └── client.py              # Gemini, Groq, OpenAI & heuristic fallback
│   │
│   └── data/
│       └── samples/               # 1-click sample demo datasets
│           ├── superstore_sales.csv
│           └── customer_churn.csv
│
├── app.py                         # Top-level entrypoint for Streamlit Cloud
├── requirements.txt               # Production dependencies
├── .env.example                   # Environment variable template
├── .gitignore                     # Git ignore rules
└── .streamlit/
    └── config.toml                # Cloud deployment & theme configuration
```

---

## 🚀 Quick Start (Local Development)

### 1. Clone the repository & install dependencies
```bash
git clone https://github.com/your-username/autoanalyst-ai.git
cd autoanalyst-ai
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your API key for Google Gemini, Groq, or OpenAI.
*(Note: If no API key is provided, AutoAnalyst AI automatically runs in Autonomous Heuristic Mode without crashing!)*

### 3. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 Public Deployment (Streamlit Community Cloud)

1. Push this repository to **GitHub**.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **New App**.
3. Select your repository, branch (`main`), and set the Main file path to:
   ```
   app.py
   ```
4. Click **Advanced Settings** -> **Secrets** and optionally paste your API keys:
   ```toml
   LLM_PROVIDER = "auto"
   GEMINI_API_KEY = "your-gemini-key"
   GROQ_API_KEY = "your-groq-key"
   OPENAI_API_KEY = "your-openai-key"
   ```
5. Click **Deploy!** Your application is immediately live on a public HTTPS URL.

---

## 🔒 Security & Privacy Guarantees

- **No Remote File Storage:** User datasets are stored strictly in temporary, session-scoped memory and discarded when the session ends.
- **Sandboxed Execution:** Python execution is restricted to safe analytical libraries (AST inspection blocks `os`, `sys`, `subprocess`, socket operations, and file modifications).
- **SELECT-Only SQL:** Queries are executed against an ephemeral in-memory SQLite database with mutating statements (`DROP`, `DELETE`, `UPDATE`) forbidden.
- **No Cross-User Leaks:** User sessions are completely isolated in Streamlit.
