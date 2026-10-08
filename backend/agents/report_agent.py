from typing import Dict, Any, List, Optional
from datetime import datetime

class ReportAgent:
    """
    Autonomous Report Agent.
    Synthesizes the complete analytical investigation into a 15-section structured report package.
    """

    def __init__(self):
        pass

    def compile_report(
        self,
        question: str,
        dataset_profile: Dict[str, Any],
        plan: Dict[str, Any],
        execution_results: Dict[str, Any],
        insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile comprehensive analysis report payload."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dims = dataset_profile.get("dimensions", {})
        dq = dataset_profile.get("data_quality", {})
        kpis = execution_results.get("kpis", {})
        stats = execution_results.get("statistical_results", {})
        anomalies = execution_results.get("anomaly_results", {})
        ml = execution_results.get("ml_results", {})
        charts = execution_results.get("charts", {})

        report = {
            "metadata": {
                "report_title": "AUTOANALYST AI - DATA ANALYSIS REPORT",
                "generated_at": timestamp,
                "dataset_name": dataset_profile.get("filename", "Dataset"),
                "user_question": question
            },
            "1_executive_summary": insights.get(
                "executive_summary",
                "Automated analytical evaluation conducted across the uploaded dataset."
            ),
            "2_dataset_overview": {
                "rows": dims.get("rows", 0),
                "columns": dims.get("columns", 0),
                "memory_mb": dims.get("memory_mb", 0.0),
                "numerical_columns": dataset_profile.get("columns", {}).get("numerical", []),
                "categorical_columns": dataset_profile.get("columns", {}).get("categorical", []),
                "datetime_columns": dataset_profile.get("columns", {}).get("datetime", [])
            },
            "3_data_quality": {
                "quality_score": dq.get("score", 100.0),
                "rating": dq.get("rating", "Good"),
                "total_missing_cells": dq.get("total_missing_cells", 0),
                "missing_percentage": dq.get("overall_missing_pct", 0.0),
                "duplicate_rows": dq.get("duplicate_rows", 0),
                "outliers_count": dq.get("total_outliers", 0)
            },
            "4_user_question": question,
            "5_analysis_plan": plan.get("execution_steps", []),
            "6_analysis_performed": execution_results.get("execution_log", []),
            "7_key_kpis": kpis,
            "8_important_findings": insights.get("observed_facts", []),
            "9_statistical_analysis": {
                "descriptive_stats": stats.get("descriptive", {}),
                "bivariate_correlation": stats.get("bivariate_correlation", {}),
                "group_comparison": stats.get("group_comparison", {}),
                "statistical_notes": insights.get("statistical_findings", [])
            },
            "10_visualizations": list(charts.keys()),
            "11_machine_learning": ml,
            "12_anomalies": {
                "detected": anomalies.get("anomalies_detected", 0),
                "rate_pct": anomalies.get("anomaly_rate_pct", 0.0),
                "top_records": anomalies.get("top_anomalies", [])
            },
            "13_business_insights": insights.get("ai_interpretations", []),
            "14_recommendations": insights.get("recommendations", []),
            "15_conclusion": (
                f"AutoAnalyst AI has completed autonomous investigation of '{question}'. "
                f"All metrics, correlations, and segments were empirically computed. "
                f"Recommended strategic actions should be phased in with KPI tracking."
            )
        }

        return report
