from typing import Dict, Any, List, Optional
from backend.llm.client import LLMClient
import json
import re

class AnalysisPlanner:
    """
    Autonomous Analysis Planner.
    Synthesizes question intent and dataset schema into an ordered, multi-tool analytical execution plan.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def create_plan(
        self,
        question: str,
        analysis_intent: Dict[str, Any],
        dataset_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate structured execution plan containing steps, tools, and visual layouts."""
        intent = analysis_intent.get("intent", "general_overview")
        metric = analysis_intent.get("primary_metric")
        dimension = analysis_intent.get("primary_dimension")
        sec_metric = analysis_intent.get("secondary_metric")
        time_col = analysis_intent.get("time_column")

        # Fallback to defaults if missing
        num_cols = dataset_profile.get("columns", {}).get("numerical", [])
        cat_cols = dataset_profile.get("columns", {}).get("categorical", [])
        date_cols = dataset_profile.get("columns", {}).get("datetime", [])

        if not metric and num_cols:
            metric = num_cols[0]
        if not dimension and cat_cols:
            dimension = cat_cols[0]

        # Determine tools to invoke
        tools_to_run = ["statistics", "python"]
        if analysis_intent.get("requires_anomaly"):
            tools_to_run.append("anomaly")
        if analysis_intent.get("requires_ml"):
            tools_to_run.append("ml")
        tools_to_run.append("visualization")

        # Build dynamic steps based on intent
        steps = []
        hypotheses = []

        steps.append({
            "step_id": 1,
            "title": "Compute Baseline Descriptive Metrics",
            "tool": "statistics",
            "action": f"Calculate parametric and non-parametric distribution statistics (mean, median, IQR, 95% CI) for '{metric}'."
        })

        if dimension and dimension in (cat_cols + date_cols):
            steps.append({
                "step_id": 2,
                "title": f"Segment Analysis across {dimension}",
                "tool": "statistics",
                "action": f"Compare '{metric}' across '{dimension}' categories using hypothesis testing (t-test / ANOVA) to isolate divergence."
            })
            hypotheses.append(f"Performance of '{metric}' differs significantly across '{dimension}' segments.")

        if sec_metric and sec_metric in num_cols:
            steps.append({
                "step_id": len(steps) + 1,
                "title": f"Bivariate Correlation: {metric} vs {sec_metric}",
                "tool": "statistics",
                "action": f"Evaluate Pearson & Spearman correlation between '{metric}' and '{sec_metric}' with statistical significance testing."
            })
            hypotheses.append(f"Changes in '{metric}' are inversely or directly associated with '{sec_metric}'.")

        if time_col and (analysis_intent.get("requires_time_series") or intent == "trend_analysis"):
            steps.append({
                "step_id": len(steps) + 1,
                "title": "Temporal Trend & Period-over-Period Analysis",
                "tool": "python",
                "action": f"Resample '{metric}' over time using '{time_col}' to evaluate growth trajectory, acceleration, and seasonality."
            })

        if analysis_intent.get("requires_anomaly") or intent in ("root_cause", "anomaly_detection"):
            steps.append({
                "step_id": len(steps) + 1,
                "title": "Multi-Dimensional Anomaly Isolation",
                "tool": "anomaly",
                "action": "Run Isolation Forest across numerical attributes to isolate extreme records that distort overall performance."
            })

        if analysis_intent.get("requires_ml") or intent == "forecasting_ml":
            steps.append({
                "step_id": len(steps) + 1,
                "title": "Machine Learning Predictive Modeling",
                "tool": "ml",
                "action": f"Train predictive/forecasting model on '{metric}' or target variable with cross-validated evaluation metrics."
            })

        steps.append({
            "step_id": len(steps) + 1,
            "title": "Interactive Visual Dashboard Synthesis",
            "tool": "visualization",
            "action": f"Generate question-tailored Plotly dashboard visualizing trends, category breakdowns, and diagnostic drill-downs."
        })

        steps.append({
            "step_id": len(steps) + 1,
            "title": "Rigorous Business Synthesis & Recommendations",
            "tool": "insight",
            "action": "Generate four-part structured business insights grounded strictly on computed results with zero hallucination."
        })

        plan = {
            "question": question,
            "intent": intent,
            "focal_metric": metric,
            "focal_dimension": dimension,
            "secondary_metric": sec_metric,
            "time_dimension": time_col,
            "tools_required": tools_to_run,
            "hypotheses_to_test": hypotheses,
            "execution_steps": steps,
            "plan_summary": f"Autonomous plan to investigate '{intent}' regarding '{metric}' partitioned by '{dimension}'."
        }

        return plan
