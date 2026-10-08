import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import time

from backend.tools.statistics_tool import StatisticsTool
from backend.tools.python_tool import PythonAnalysisTool
from backend.tools.sql_tool import SQLAnalysisTool
from backend.tools.ml_tool import MLAnalysisTool
from backend.tools.anomaly_tool import AnomalyDetectionTool
from backend.tools.visualization_tool import VisualizationTool

class AutonomousAnalystAgent:
    """
    Core Autonomous Analyst Agent.
    Executes the analytical plan using deterministic Python, SQL, statistical, and ML tools.
    Strictly verifies all numerical computations against the real dataset.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.stats_tool = StatisticsTool(self.df)
        self.py_tool = PythonAnalysisTool(self.df)
        self.sql_tool = SQLAnalysisTool(self.df)
        self.ml_tool = MLAnalysisTool(self.df)
        self.anomaly_tool = AnomalyDetectionTool(self.df)
        self.viz_tool = VisualizationTool(self.df)

    def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the multi-step analysis plan and assemble verifiable analytical results."""
        start_time = time.time()
        metric = plan.get("focal_metric")
        dimension = plan.get("focal_dimension")
        sec_metric = plan.get("secondary_metric")
        time_col = plan.get("time_dimension")
        intent = plan.get("intent", "general_overview")

        kpis = {}
        statistical_results = {}
        group_breakdowns = {}
        time_series_summary = {}
        anomaly_results = {}
        ml_results = {}
        generated_charts = {}
        execution_log = []

        # 1. Compute Primary Metric KPIs & Descriptive Stats
        if metric and metric in self.df.columns and pd.api.types.is_numeric_dtype(self.df[metric]):
            metric_series = self.df[metric].dropna()
            total_sum = float(metric_series.sum())
            mean_val = float(metric_series.mean())
            median_val = float(metric_series.median())
            min_val = float(metric_series.min())
            max_val = float(metric_series.max())
            std_val = float(metric_series.std()) if len(metric_series) > 1 else 0.0

            kpis["primary_metric"] = {
                "name": metric,
                "total": round(total_sum, 2),
                "mean": round(mean_val, 2),
                "median": round(median_val, 2),
                "min": round(min_val, 2),
                "max": round(max_val, 2),
                "std": round(std_val, 2)
            }
            statistical_results["descriptive"] = self.stats_tool.calculate_descriptive_stats(metric)
            execution_log.append(f"Computed descriptive statistics for '{metric}' (Total: {round(total_sum, 2)}, Mean: {round(mean_val, 2)}).")

        # 2. Secondary Metric Baseline
        if sec_metric and sec_metric in self.df.columns and pd.api.types.is_numeric_dtype(self.df[sec_metric]):
            sec_series = self.df[sec_metric].dropna()
            kpis["secondary_metric"] = {
                "name": sec_metric,
                "total": round(float(sec_series.sum()), 2),
                "mean": round(float(sec_series.mean()), 2),
                "median": round(float(sec_series.median()), 2)
            }
            # Correlation between primary and secondary metric
            if metric:
                corr_res = self.stats_tool.calculate_correlation(metric, sec_metric)
                statistical_results["bivariate_correlation"] = corr_res
                execution_log.append(f"Evaluated correlation between '{metric}' and '{sec_metric}' (r={corr_res.get('pearson_r')}).")

        # 3. Categorical Segment Breakdown & Hypothesis Testing
        if metric and dimension and dimension in self.df.columns:
            group_comp = self.stats_tool.perform_group_comparison(metric, dimension)
            statistical_results["group_comparison"] = group_comp
            if "group_breakdown" in group_comp:
                group_breakdowns[dimension] = group_comp["group_breakdown"]
                # Store Top and Lowest Performer KPIs
                top_group = group_comp["group_breakdown"][0]
                bottom_group = group_comp["group_breakdown"][-1]
                kpis["top_performer"] = {
                    "dimension": dimension,
                    "category": top_group["group"],
                    "mean": top_group["mean"],
                    "total": top_group["sum"]
                }
                kpis["lowest_performer"] = {
                    "dimension": dimension,
                    "category": bottom_group["group"],
                    "mean": bottom_group["mean"],
                    "total": bottom_group["sum"]
                }
            execution_log.append(f"Partitioned '{metric}' across '{dimension}' (Identified top group: {top_group['group']}).")

        # 4. Temporal Trend Analysis
        if metric and time_col and time_col in self.df.columns:
            try:
                date_df = self.df[[time_col, metric]].dropna().copy()
                date_df[time_col] = pd.to_datetime(date_df[time_col], errors='coerce')
                date_df = date_df.dropna().sort_values(time_col)
                
                # Resample monthly
                monthly = date_df.set_index(time_col).resample('M')[metric].sum().reset_index()
                if len(monthly) >= 2:
                    monthly["growth_pct"] = monthly[metric].pct_change() * 100
                    monthly["growth_pct"] = monthly["growth_pct"].fillna(0.0).round(2)
                    first_val = float(monthly[metric].iloc[0])
                    last_val = float(monthly[metric].iloc[-1])
                    overall_growth = round(((last_val - first_val) / abs(first_val) * 100), 2) if first_val != 0 else 0.0

                    time_series_summary = {
                        "cadence": "Monthly",
                        "periods_count": len(monthly),
                        "overall_growth_pct": overall_growth,
                        "recent_growth_pct": round(float(monthly["growth_pct"].iloc[-1]), 2),
                        "period_data": [
                            {
                                "period": row[time_col].strftime("%Y-%m"),
                                "value": round(float(row[metric]), 2),
                                "growth_pct": float(row["growth_pct"])
                            }
                            for _, row in monthly.iterrows()
                        ]
                    }
                    kpis["period_growth"] = {
                        "overall_growth_pct": overall_growth,
                        "recent_period_growth_pct": float(monthly["growth_pct"].iloc[-1])
                    }
                    execution_log.append(f"Evaluated temporal trajectory over {len(monthly)} monthly periods (Overall growth: {overall_growth}%).")
            except Exception as e:
                execution_log.append(f"Temporal analysis note: {str(e)}")

        # 5. Anomaly Detection if planned or relevant
        if "anomaly" in plan.get("tools_required", []) or intent in ("anomaly_detection", "root_cause"):
            anomaly_res = self.anomaly_tool.detect_anomalies(top_n=8)
            if anomaly_res.get("success"):
                anomaly_results = anomaly_res
                kpis["anomalies"] = {
                    "count": anomaly_res["anomalies_detected"],
                    "rate_pct": anomaly_res["anomaly_rate_pct"]
                }
                execution_log.append(f"Isolated {anomaly_res['anomalies_detected']} anomalous records ({anomaly_res['anomaly_rate_pct']}% anomaly rate).")

        # 6. Machine Learning / Forecasting if planned
        if "ml" in plan.get("tools_required", []) or intent == "forecasting_ml":
            if time_col and metric:
                # Forecasting
                forecast_res = self.ml_tool.forecast_metric(metric, time_col, horizon_periods=3)
                if forecast_res.get("success"):
                    ml_results["forecasting"] = forecast_res
                    execution_log.append(f"Generated 3-period forward forecast for '{metric}'.")
            elif metric:
                # Regression
                reg_res = self.ml_tool.train_regression(metric)
                if reg_res.get("success"):
                    ml_results["regression"] = reg_res
                    execution_log.append(f"Trained regression model on '{metric}' (Test R²: {reg_res['metrics']['test_r2']}).")

        # 7. Generate Targeted Visualizations
        # Chart 1: Primary Answer Chart
        if time_col and (plan.get("requires_time_series") or intent in ("trend_analysis", "forecasting_ml")):
            fig_main = self.viz_tool.create_chart("line", x_col=time_col, y_col=metric, title=f"{metric} Trend Over Time")
            generated_charts["primary_trend"] = fig_main
        elif dimension and metric:
            fig_main = self.viz_tool.create_chart("bar", x_col=dimension, y_col=metric, title=f"{metric} by {dimension}")
            generated_charts["primary_breakdown"] = fig_main

        # Chart 2: Secondary / Diagnostic Chart
        if sec_metric and metric:
            fig_diag = self.viz_tool.create_chart("scatter", x_col=sec_metric, y_col=metric, color_col=dimension, title=f"{metric} vs {sec_metric}")
            generated_charts["diagnostic_scatter"] = fig_diag
        elif metric:
            fig_dist = self.viz_tool.create_chart("distribution", x_col=metric, title=f"Distribution of {metric}")
            generated_charts["metric_distribution"] = fig_dist

        # Chart 3: Composition / Additional Dimension
        cat_cols = [c for c in self.df.select_dtypes(include=['object', 'category']).columns if c != dimension]
        if cat_cols and metric:
            fig_comp = self.viz_tool.create_chart("pie", x_col=cat_cols[0], y_col=metric, title=f"{metric} Share by {cat_cols[0]}")
            generated_charts["secondary_composition"] = fig_comp

        execution_time_ms = round((time.time() - start_time) * 1000, 1)
        execution_log.append(f"Successfully completed analysis in {execution_time_ms}ms.")

        return {
            "plan": plan,
            "kpis": kpis,
            "statistical_results": statistical_results,
            "group_breakdowns": group_breakdowns,
            "time_series_summary": time_series_summary,
            "anomaly_results": anomaly_results,
            "ml_results": ml_results,
            "charts": generated_charts,
            "execution_log": execution_log,
            "execution_time_ms": execution_time_ms
        }
