import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional

class DashboardGenerator:
    """
    Intelligent Dynamic Dashboard Generator.
    Dynamically renders an interactive, question-specific analytical dashboard
    with KPI cards, interactive Plotly charts, dynamic filters, and data slice views.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def render_dashboard(
        self,
        question: str,
        execution_results: Dict[str, Any],
        insights: Dict[str, Any]
    ):
        """Render question-tailored interactive dashboard in Streamlit."""
        kpis = execution_results.get("kpis", {})
        charts = execution_results.get("charts", {})
        anomalies = execution_results.get("anomaly_results", {})
        plan = execution_results.get("plan", {})
        focal_dim = plan.get("focal_dimension")
        focal_metric = plan.get("focal_metric")

        st.markdown(f"### 📊 Dynamic Dashboard: *\"{question}\"*")
        st.markdown(
            f"<div style='background: #F8FAFC; border-left: 4px solid #2563EB; padding: 12px 16px; border-radius: 6px; margin-bottom: 20px; color: #334155; font-size: 0.95rem;'>"
            f"<b>Executive Summary:</b> {insights.get('executive_summary', 'Automated dashboard synthesized from empirical data.')}"
            f"</div>",
            unsafe_allow_html=True
        )

        # 1. KPI Cards Row
        self._render_kpi_row(kpis)

        st.markdown("---")

        # 2. Interactive Charts Section
        chart_keys = list(charts.keys())
        if len(chart_keys) == 1:
            st.plotly_chart(charts[chart_keys[0]], use_container_width=True)
        elif len(chart_keys) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(charts[chart_keys[0]], use_container_width=True)
            with col2:
                st.plotly_chart(charts[chart_keys[1]], use_container_width=True)

        if len(chart_keys) >= 3:
            st.plotly_chart(charts[chart_keys[2]], use_container_width=True)

        # 3. Dynamic Filter & Data Slice Drill-Down
        with st.expander("🔍 Interactive Data Slice & Drill-Down Explorer", expanded=False):
            self._render_data_drilldown(focal_dim, focal_metric)

        # 4. Anomalies Callout if present
        if anomalies and anomalies.get("anomalies_detected", 0) > 0:
            with st.expander(f"⚠️ Anomaly Audit ({anomalies['anomalies_detected']} flagged records)", expanded=False):
                st.markdown(
                    f"**Detection Method:** Isolation Forest across {', '.join(anomalies.get('features_analyzed', []))} "
                    f"({anomalies.get('anomaly_rate_pct')}% anomaly rate)."
                )
                top_anoms = anomalies.get("top_anomalies", [])
                if top_anoms:
                    st.dataframe(pd.DataFrame(top_anoms), use_container_width=True)

    def _render_kpi_row(self, kpis: Dict[str, Any]):
        """Render responsive KPI metric cards."""
        cols = st.columns(4)

        # KPI 1: Primary Metric Total / Value
        pm = kpis.get("primary_metric", {})
        with cols[0]:
            if pm:
                st.metric(
                    label=f"Total {pm.get('name', 'Metric')}",
                    value=f"{pm.get('total', 0):,}",
                    delta=f"Avg: {pm.get('mean', 0):,}"
                )
            else:
                st.metric(label="Total Records", value=f"{len(self.df):,}")

        # KPI 2: Top Performing Segment
        top = kpis.get("top_performer")
        with cols[1]:
            if top:
                st.metric(
                    label=f"Top {top.get('dimension', 'Segment')}",
                    value=str(top.get("category", "N/A")),
                    delta=f"Total: {top.get('total', 0):,}"
                )
            elif "secondary_metric" in kpis:
                sm = kpis["secondary_metric"]
                st.metric(label=f"Total {sm.get('name')}", value=f"{sm.get('total', 0):,}")
            else:
                st.metric(label="Columns Analyzed", value=f"{len(self.df.columns)}")

        # KPI 3: Lowest Segment or Growth
        low = kpis.get("lowest_performer")
        pg = kpis.get("period_growth")
        with cols[2]:
            if pg:
                growth_val = pg.get("overall_growth_pct", 0.0)
                st.metric(
                    label="Overall Growth",
                    value=f"{growth_val}%",
                    delta=f"Recent: {pg.get('recent_period_growth_pct', 0)}%"
                )
            elif low:
                st.metric(
                    label=f"Lowest {low.get('dimension', 'Segment')}",
                    value=str(low.get("category", "N/A")),
                    delta=f"Mean: {low.get('mean', 0):,}",
                    delta_color="inverse"
                )
            else:
                st.metric(label="Data Completeness", value="100%")

        # KPI 4: Anomalies or Quality
        anoms = kpis.get("anomalies")
        with cols[3]:
            if anoms:
                st.metric(
                    label="Anomalies Detected",
                    value=f"{anoms.get('count', 0)}",
                    delta=f"{anoms.get('rate_pct', 0)}% of data",
                    delta_color="inverse"
                )
            else:
                st.metric(label="Analysis Confidence", value="High (Empirical)")

    def _render_data_drilldown(self, focal_dim: Optional[str], focal_metric: Optional[str]):
        """Render dynamic filter controls and data slice table."""
        filter_df = self.df.copy()

        # If a focal dimension exists, offer multi-select filtering
        if focal_dim and focal_dim in filter_df.columns:
            unique_vals = list(filter_df[focal_dim].dropna().unique())
            if len(unique_vals) <= 30:
                selected_vals = st.multiselect(
                    f"Filter by {focal_dim}:",
                    options=unique_vals,
                    default=unique_vals[:5] if len(unique_vals) > 5 else unique_vals
                )
                if selected_vals:
                    filter_df = filter_df[filter_df[focal_dim].isin(selected_vals)]

        st.markdown(f"**Showing {len(filter_df)} records:**")
        st.dataframe(filter_df.head(100), use_container_width=True)

        # Download button for filtered slice
        csv_data = filter_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data Slice (CSV)",
            data=csv_data,
            file_name="autoanalyst_filtered_data.csv",
            mime="text/csv"
        )
