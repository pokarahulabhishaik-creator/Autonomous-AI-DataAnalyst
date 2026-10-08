import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.tools.data_profiler import DataProfiler
from backend.tools.data_cleaner import DataCleaner
from backend.agents.question_analyzer import QuestionAnalyzer
from backend.agents.planner import AnalysisPlanner
from backend.agents.analyst_agent import AutonomousAnalystAgent
from backend.agents.insight_agent import InsightAgent
from backend.agents.report_agent import ReportAgent
from backend.dashboard.dashboard_generator import DashboardGenerator
from backend.reports.pdf_report import PDFReportGenerator
from backend.reports.docx_report import DocxReportGenerator
from backend.reports.excel_report import ExcelReportGenerator
from backend.llm.client import LLMClient

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AutoAnalyst AI - Autonomous AI Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern, Executive UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #3B82F6 100%);
        color: white;
        padding: 26px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.15);
    }
    .main-header h1 {
        color: white !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .main-header p {
        color: #E0E7FF !important;
        font-size: 1.05rem;
        margin: 0;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .pillar-card {
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .fact-card {
        background-color: #EFF6FF;
        border-left: 4px solid #2563EB;
        color: #1E3A8A;
    }
    .stat-card {
        background-color: #F0FDF4;
        border-left: 4px solid #10B981;
        color: #065F46;
    }
    .interp-card {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        color: #92400E;
    }
    .rec-card {
        background-color: #FAF5FF;
        border-left: 4px solid #8B5CF6;
        color: #5B21B6;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "dataset_filename" not in st.session_state:
    st.session_state.dataset_filename = ""
if "dataset_profile" not in st.session_state:
    st.session_state.dataset_profile = None
if "cleaning_audit" not in st.session_state:
    st.session_state.cleaning_audit = None
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "active_results" not in st.session_state:
    st.session_state.active_results = None
if "active_insights" not in st.session_state:
    st.session_state.active_insights = None
if "active_plan" not in st.session_state:
    st.session_state.active_plan = None
if "active_question" not in st.session_state:
    st.session_state.active_question = ""
if "active_report_payload" not in st.session_state:
    st.session_state.active_report_payload = None

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/analytics.png", width=64)
    st.title("AutoAnalyst AI")
    st.caption("Autonomous AI Data Analyst & Dashboard Generator")
    st.markdown("---")

    # 1. Dataset Upload & Sample Loader
    st.subheader("📂 Dataset Upload")
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Upload files up to 50MB. Safe session-only processing."
    )

    st.markdown("**— OR Quick Load Demo Datasets —**")
    col_s1, col_s2 = st.columns(2)
    sample_to_load = None
    with col_s1:
        if st.button("🛒 Superstore Sales", use_container_width=True):
            sample_to_load = "backend/data/samples/superstore_sales.csv"
            sample_name = "superstore_sales.csv"
    with col_s2:
        if st.button("🔄 Customer Churn", use_container_width=True):
            sample_to_load = "backend/data/samples/customer_churn.csv"
            sample_name = "customer_churn.csv"

    # Handle dataset loading
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                loaded_df = pd.read_csv(uploaded_file)
            else:
                loaded_df = pd.read_excel(uploaded_file)
            st.session_state.raw_df = loaded_df
            st.session_state.dataset_filename = uploaded_file.name
            st.session_state.active_results = None
            st.session_state.active_insights = None
            st.success(f"Uploaded: {uploaded_file.name} ({len(loaded_df):,} rows)")
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")

    elif sample_to_load:
        try:
            loaded_df = pd.read_csv(sample_to_load)
            st.session_state.raw_df = loaded_df
            st.session_state.dataset_filename = sample_name
            st.session_state.active_results = None
            st.session_state.active_insights = None
            st.success(f"Loaded: {sample_name} ({len(loaded_df):,} rows)")
        except Exception as e:
            st.error(f"Error reading sample dataset: {e}")

    # Process profiling & cleaning if dataset present but profile absent
    if st.session_state.raw_df is not None:
        if st.session_state.dataset_profile is None or st.session_state.dataset_profile.get("filename") != st.session_state.dataset_filename:
            with st.spinner("Autonomous profiler analyzing dataset..."):
                profiler = DataProfiler(st.session_state.raw_df, filename=st.session_state.dataset_filename)
                st.session_state.dataset_profile = profiler.profile()

                cleaner = DataCleaner(st.session_state.raw_df)
                cleaned, issues, ops = cleaner.clean()
                st.session_state.cleaned_df = cleaned
                st.session_state.cleaning_audit = {
                    "issues": issues,
                    "operations": ops
                }

    st.markdown("---")

    # 2. Dataset Information in Sidebar
    if st.session_state.dataset_profile:
        p = st.session_state.dataset_profile
        dims = p["dimensions"]
        dq = p["data_quality"]

        st.subheader("ℹ️ Dataset Metadata")
        st.write(f"**File:** `{p['filename']}`")
        st.write(f"**Rows:** {dims['rows']:,} | **Columns:** {dims['columns']}")
        st.write(f"**Memory:** {dims['memory_mb']} MB")
        
        score = dq['score']
        badge_color = "green" if score >= 80 else ("orange" if score >= 60 else "red")
        st.markdown(f"**Data Quality Score:** :{badge_color}[**{score}/100 ({dq['rating']})**]")
        st.progress(score / 100.0)
        st.markdown("---")

    # 3. Settings & LLM Configuration
    with st.expander("⚙️ AI Engine Settings", expanded=False):
        llm_provider = st.selectbox(
            "LLM Reasoning Engine",
            ["Auto-Detect / Heuristic", "Google Gemini", "Groq", "OpenAI"],
            index=0
        )
        provider_map = {
            "Auto-Detect / Heuristic": "auto",
            "Google Gemini": "gemini",
            "Groq": "groq",
            "OpenAI": "openai"
        }
        api_key_input = st.text_input(
            "Custom API Key (Optional)",
            type="password",
            help="If not provided, AutoAnalyst AI uses environment/Streamlit secrets or its built-in empirical heuristic engine."
        )
        temperature = st.slider("Reasoning Temperature", 0.0, 1.0, 0.2, 0.05)

    # 4. Analysis Session History
    if st.session_state.analysis_history:
        st.markdown("---")
        st.subheader("📜 Analysis History")
        for idx, h_item in enumerate(reversed(st.session_state.analysis_history[-5:])):
            if st.button(f"🔍 {h_item['question'][:30]}...", key=f"hist_{idx}"):
                st.session_state.active_question = h_item["question"]
                st.session_state.active_plan = h_item["plan"]
                st.session_state.active_results = h_item["results"]
                st.session_state.active_insights = h_item["insights"]
                st.session_state.active_report_payload = h_item["report_payload"]
                st.rerun()

    # 5. About Project
    with st.expander("💡 About AutoAnalyst AI"):
        st.markdown("""
        **AutoAnalyst AI** is an autonomous data analyst system.
        
        - **Reasoning Layer:** LLM / Multi-Agent Planner
        - **Calculation Layer:** Python, Pandas, SQLite, SciPy, Scikit-learn
        - **Guaranteed Rigor:** Zero hallucination of statistics.
        - **RAG Core:** ChromaDB + Statistical Knowledge Base
        - **Export:** PDF (ReportLab), Word (python-docx), Excel (openpyxl)
        """)

# --- MAIN PAGE ---

# Header Banner
st.markdown("""
<div class="main-header">
    <h1>AutoAnalyst AI</h1>
    <p>Autonomous AI Data Analyst & Intelligent Dashboard Generator — Powered by Multi-Agent Tool Orchestration & Empirical Verification</p>
</div>
""", unsafe_allow_html=True)

# Check if dataset is loaded
if st.session_state.raw_df is None:
    st.info("👋 **Welcome to AutoAnalyst AI!** Please upload a CSV/Excel dataset from the sidebar or click one of the sample datasets to begin instant exploration.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🚀 Key Capabilities
        - **Autonomous Dataset Profiling & Quality Scoring**
        - **Natural Language Question Interpretation**
        - **Multi-Step Analytical Planning & Hypothesis Testing**
        - **Empirical Execution (Pandas, SQL, SciPy, Scikit-learn)**
        - **Dynamic Question-Specific Dashboard Generation**
        - **Four-Pillar Grounded Insights (Facts, Stats, AI, Recs)**
        - **One-Click Multi-Format Export (PDF, DOCX, Excel)**
        """)
    with col2:
        st.markdown("""
        ### 🛡️ Analytical Integrity Guarantee
        - **No Hallucinated Figures:** The LLM plans and interprets; reliable Python tools perform every calculation.
        - **Correlation != Causation:** Rigorous statistical hypothesis testing.
        - **Non-Destructive Cleaning:** Original data is never silently modified.
        - **Cloud-Ready & Secure:** In-memory, session-isolated, zero data leakage.
        """)
    st.stop()

# --- DATASET TABS ---
df_active = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.raw_df
profile = st.session_state.dataset_profile

tab_overview, tab_profiler, tab_cleaning = st.tabs([
    "📋 Dataset Preview",
    "🔍 Autonomous Profiler & Quality Audit",
    "🧹 Data Cleaning & Changelog"
])

with tab_overview:
    col_o1, col_o2 = st.columns([3, 1])
    with col_o1:
        st.markdown(f"**Showing first 100 rows of `{st.session_state.dataset_filename}` ({len(df_active):,} total rows):**")
        st.dataframe(df_active.head(100), use_container_width=True)
    with col_o2:
        st.markdown("**Column Schema:**")
        col_types = pd.DataFrame({
            "Column": df_active.columns,
            "Type": [str(t) for t in df_active.dtypes],
            "Non-Null": [int(df_active[c].notna().sum()) for c in df_active.columns]
        })
        st.dataframe(col_types, use_container_width=True, hide_index=True)

with tab_profiler:
    dq = profile["data_quality"]
    dims = profile["dimensions"]

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Rows", f"{dims['rows']:,}")
    with m2:
        st.metric("Total Columns", f"{dims['columns']}")
    with m3:
        st.metric("Quality Score", f"{dq['score']}/100", delta=dq['rating'])
    with m4:
        st.metric("Missing Cells", f"{dq['total_missing_cells']:,}", delta=f"{dq['overall_missing_pct']}%", delta_color="inverse")
    with m5:
        st.metric("Duplicate Rows", f"{dq['duplicate_rows']:,}", delta=f"{dq['duplicate_pct']}%", delta_color="inverse")

    st.markdown("---")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("##### 🔢 Numerical Distribution Overview")
        if profile.get("numerical_statistics"):
            num_stat_df = pd.DataFrame(profile["numerical_statistics"]).T[["mean", "std", "median", "min", "max", "skewness", "outliers_count"]]
            st.dataframe(num_stat_df, use_container_width=True)
        else:
            st.info("No numerical columns detected.")

    with col_p2:
        st.markdown("##### 🔤 Categorical Breakdown")
        if profile.get("categorical_statistics"):
            cat_stat_rows = []
            for col_name, c_info in profile["categorical_statistics"].items():
                top_cats = ", ".join([f"{k} ({v})" for k, v in list(c_info["top_categories"].items())[:3]])
                cat_stat_rows.append({
                    "Column": col_name,
                    "Unique Values": c_info["unique_count"],
                    "Mode": c_info["mode"],
                    "Top Categories": top_cats
                })
            st.dataframe(pd.DataFrame(cat_stat_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No categorical columns detected.")

    if profile.get("top_correlations"):
        st.markdown("##### 🔗 Notable Correlations (|r| > 0.3)")
        st.dataframe(pd.DataFrame(profile["top_correlations"]), use_container_width=True, hide_index=True)

with tab_cleaning:
    audit = st.session_state.cleaning_audit
    if audit:
        st.markdown("#### 🛡️ Autonomous Non-Destructive Data Cleaning")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown(f"**Issues Detected Before Cleaning ({len(audit['issues'])}):**")
            if audit["issues"]:
                for iss in audit["issues"]:
                    st.warning(f"⚠️ **{iss.get('type', 'Issue').replace('_', ' ').title()}:** {iss.get('description')}")
            else:
                st.success("✨ Zero critical data quality defects detected in original dataset.")

        with col_c2:
            st.markdown(f"**Cleaning Operations Performed ({len(audit['operations'])}):**")
            if audit["operations"]:
                for op in audit["operations"]:
                    st.info(f"🔧 **{op.get('operation', 'Operation').replace('_', ' ').title()}:** {op.get('details')}")
            else:
                st.success("No modifications required. Dataset is pristine.")

st.markdown("---")

# --- NATURAL LANGUAGE ANALYSIS SECTION ---
st.markdown("### 💬 Ask AutoAnalyst AI a Question")
st.caption("Ask questions in natural language. AutoAnalyst AI autonomously devises an analytical plan, executes tools, generates charts, and synthesizes strategic findings.")

# Clickable example question chips
st.markdown("**Example Questions:**")
example_cols = st.columns(4)
example_questions = [
    "What are the most important trends?",
    "Why are profits decreasing?",
    "Which products are performing poorly?",
    "Which region performs best?",
    "Find anomalies in this dataset.",
    "What factors affect profit?",
    "Predict next month's sales.",
    "Compare this year's performance with last year's."
]

selected_example = None
for i, q_text in enumerate(example_questions):
    col = example_cols[i % 4]
    with col:
        if st.button(f"💡 {q_text}", key=f"ex_{i}", use_container_width=True):
            selected_example = q_text

# Text input for query
user_query = st.text_input(
    "Your Question:",
    value=selected_example or st.session_state.active_question or "",
    placeholder="e.g., Which products or categories are dragging down total profit?",
    label_visibility="collapsed"
)

col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    analyze_clicked = st.button("🚀 Analyze Dataset", type="primary", use_container_width=True)

# Execution Flow
if analyze_clicked and user_query:
    st.session_state.active_question = user_query
    progress_bar = st.progress(0)
    status_text = st.empty()

    llm = LLMClient(
        provider=provider_map[llm_provider],
        api_key=api_key_input if api_key_input else None,
        temperature=temperature
    )

    try:
        # Step 1: Question Understanding
        status_text.markdown("⏳ **Step 1/6:** Understanding dataset semantics & question intent...")
        progress_bar.progress(15)
        analyzer = QuestionAnalyzer(llm_client=llm)
        intent_data = analyzer.analyze(user_query, profile)

        # Step 2: Planning
        status_text.markdown("⏳ **Step 2/6:** Formulating multi-tool autonomous analysis plan...")
        progress_bar.progress(35)
        planner = AnalysisPlanner(llm_client=llm)
        plan = planner.create_plan(user_query, intent_data, profile)
        st.session_state.active_plan = plan

        # Step 3: Execution Layer
        status_text.markdown("⏳ **Step 3/6:** Executing empirical computations (Pandas, Statistics, ML, Anomalies)...")
        progress_bar.progress(55)
        agent = AutonomousAnalystAgent(df_active)
        exec_results = agent.execute_plan(plan)
        st.session_state.active_results = exec_results

        # Step 4: Visualizations
        status_text.markdown("⏳ **Step 4/6:** Synthesizing question-specific interactive Plotly charts...")
        progress_bar.progress(75)

        # Step 5: Insights
        status_text.markdown("⏳ **Step 5/6:** Generating four-pillar grounded business insights with RAG guidance...")
        progress_bar.progress(90)
        insight_agent = InsightAgent(llm_client=llm)
        insights = insight_agent.generate_insights(user_query, exec_results)
        st.session_state.active_insights = insights

        # Step 6: Compile Report Payload
        status_text.markdown("⏳ **Step 6/6:** Finalizing dynamic dashboard and report artifacts...")
        progress_bar.progress(100)
        report_agent = ReportAgent()
        report_payload = report_agent.compile_report(user_query, profile, plan, exec_results, insights)
        st.session_state.active_report_payload = report_payload

        # Save to history
        st.session_state.analysis_history.append({
            "question": user_query,
            "plan": plan,
            "results": exec_results,
            "insights": insights,
            "report_payload": report_payload,
            "timestamp": time.strftime("%H:%M:%S")
        })

        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        st.rerun()

    except Exception as e:
        status_text.empty()
        progress_bar.empty()
        st.error(f"Analysis Error: {str(e)}")

# --- DISPLAY ACTIVE ANALYSIS RESULTS ---
if st.session_state.active_results is not None:
    exec_res = st.session_state.active_results
    insights = st.session_state.active_insights
    plan = st.session_state.active_plan
    q = st.session_state.active_question
    report_payload = st.session_state.active_report_payload

    # Analysis Plan Accordion
    with st.expander("📝 View Autonomous Analysis Plan", expanded=False):
        st.markdown(f"**Intent:** `{plan.get('intent')}` | **Focal Metric:** `{plan.get('focal_metric')}` | **Focal Dimension:** `{plan.get('focal_dimension')}`")
        st.markdown("**Execution Steps:**")
        for step in plan.get("execution_steps", []):
            st.markdown(f"{step['step_id']}. **{step['title']}** (`{step['tool']}`) — {step['action']}")
        if plan.get("hypotheses_to_test"):
            st.markdown("**Hypotheses Evaluated:**")
            for h in plan["hypotheses_to_test"]:
                st.markdown(f"- *{h}*")

    st.markdown("---")

    # Render Dynamic Dashboard
    dash_gen = DashboardGenerator(df_active)
    dash_gen.render_dashboard(q, exec_res, insights)

    st.markdown("---")

    # Four Pillars of Grounded Findings
    st.markdown("### 🧠 AI Insights & Strategic Synthesis")
    col_i1, col_i2 = st.columns(2)

    with col_i1:
        st.markdown("#### 📌 1. Observed Facts (Computed)")
        for fact in insights.get("observed_facts", []):
            st.markdown(f"<div class='pillar-card fact-card'><b>Empirical Fact:</b> {fact}</div>", unsafe_allow_html=True)

        st.markdown("#### 📐 2. Statistical Findings")
        for stat in insights.get("statistical_findings", []):
            st.markdown(f"<div class='pillar-card stat-card'><b>Statistical Rigor:</b> {stat}</div>", unsafe_allow_html=True)

    with col_i2:
        st.markdown("#### 💡 3. AI Interpretations")
        for interp in insights.get("ai_interpretations", []):
            st.markdown(f"<div class='pillar-card interp-card'><b>Analytical Rationale:</b> {interp}</div>", unsafe_allow_html=True)

        st.markdown("#### 🎯 4. Strategic Recommendations")
        for rec in insights.get("recommendations", []):
            st.markdown(f"<div class='pillar-card rec-card'><b>Recommended Action:</b> {rec}</div>", unsafe_allow_html=True)

    if insights.get("limitations"):
        st.caption(f"**Analytical Caveats:** {insights['limitations']}")

    st.markdown("---")

    # --- REPORT EXPORT SECTION ---
    st.markdown("### 📥 Export Executive Analysis Report")
    st.caption("Generate publication-ready PDF, Word (DOCX), or Excel reports containing all computed metrics, findings, and recommendations.")

    col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)

    # 1. PDF Report Download
    with col_exp1:
        try:
            pdf_gen = PDFReportGenerator()
            pdf_bytes = pdf_gen.generate(report_payload)
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"AutoAnalyst_Report_{int(time.time())}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.button("📄 PDF Export Error", disabled=True, use_container_width=True)

    # 2. DOCX Report Download
    with col_exp2:
        try:
            docx_gen = DocxReportGenerator()
            docx_bytes = docx_gen.generate(report_payload)
            st.download_button(
                label="📝 Download Word (.docx)",
                data=docx_bytes,
                file_name=f"AutoAnalyst_Report_{int(time.time())}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
        except Exception as e:
            st.button("📝 DOCX Export Error", disabled=True, use_container_width=True)

    # 3. Excel Multi-Tab Report Download
    with col_exp3:
        try:
            excel_gen = ExcelReportGenerator()
            excel_bytes = excel_gen.generate(report_payload, raw_df=df_active)
            st.download_button(
                label="📊 Download Excel Report",
                data=excel_bytes,
                file_name=f"AutoAnalyst_DataReport_{int(time.time())}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except Exception as e:
            st.button("📊 Excel Export Error", disabled=True, use_container_width=True)

    # 4. Cleaned Dataset Download
    with col_exp4:
        clean_csv = df_active.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Cleaned CSV",
            data=clean_csv,
            file_name=f"cleaned_{st.session_state.dataset_filename}",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("---")

    # --- CONVERSATIONAL FOLLOW-UP SECTION ---
    st.markdown("### 🔄 Conversational Follow-Up")
    st.caption("Ask follow-up questions about this dataset or drill down into specific segments, regions, or drivers.")

    follow_up_query = st.text_input(
        "Ask a follow-up question:",
        placeholder="e.g., Which region contributed most to this? or Show me a chart for that category.",
        key="follow_up_input"
    )

    if st.button("💬 Send Follow-up"):
        if follow_up_query:
            st.session_state.active_question = f"{q} -> Follow-up: {follow_up_query}"
            st.rerun()
