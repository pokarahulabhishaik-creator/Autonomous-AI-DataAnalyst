import json
import re
from typing import Dict, Any, List, Optional
from backend.llm.client import LLMClient
from backend.rag.retriever import AnalyticsRetriever

class InsightAgent:
    """
    Autonomous Insight & Recommendation Generator.
    Synthesizes computed analytical results into four distinct pillars:
    1. Observed Facts (Verifiable empirical numbers)
    2. Statistical Findings (Distributions, tests, correlations)
    3. AI Interpretations (Analytical reasoning)
    4. Strategic Recommendations (Actionable business next steps)
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.retriever = AnalyticsRetriever()

    def generate_insights(
        self,
        question: str,
        execution_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate four-pillar structured insights and strategic recommendations."""
        kpis = execution_results.get("kpis", {})
        stats = execution_results.get("statistical_results", {})
        ts = execution_results.get("time_series_summary", {})
        anomalies = execution_results.get("anomaly_results", {})
        ml = execution_results.get("ml_results", {})

        # Retrieve domain RAG guidelines
        rag_context = self.retriever.get_context_string(f"{question} correlation causation KPI recommendations", top_k=2)

        # Build factual data context string for prompt
        fact_lines = []
        if "primary_metric" in kpis:
            pm = kpis["primary_metric"]
            fact_lines.append(f"Primary Metric '{pm['name']}': Total={pm['total']}, Mean={pm['mean']}, Median={pm['median']}, Min={pm['min']}, Max={pm['max']}.")
        if "secondary_metric" in kpis:
            sm = kpis["secondary_metric"]
            fact_lines.append(f"Secondary Metric '{sm['name']}': Total={sm['total']}, Mean={sm['mean']}.")
        if "top_performer" in kpis and "lowest_performer" in kpis:
            top = kpis["top_performer"]
            low = kpis["lowest_performer"]
            fact_lines.append(f"Segmentation across '{top['dimension']}': Top is '{top['category']}' (Mean={top['mean']}) vs Lowest '{low['category']}' (Mean={low['mean']}).")
        if "period_growth" in kpis:
            pg = kpis["period_growth"]
            fact_lines.append(f"Growth Trajectory: Overall growth={pg['overall_growth_pct']}%, Recent period growth={pg['recent_period_growth_pct']}%.")
        if "bivariate_correlation" in stats:
            bc = stats["bivariate_correlation"]
            fact_lines.append(f"Correlation: Pearson r={bc.get('pearson_r')} (p={bc.get('pearson_p_value'):.4e}, strength={bc.get('strength')}, significant={bc.get('is_significant')}).")
        if "group_comparison" in stats:
            gc = stats["group_comparison"]
            fact_lines.append(f"Hypothesis Test: {gc.get('test')} stat={gc.get('statistic')}, p-value={gc.get('p_value'):.4e}, significant={gc.get('is_significant')}.")
        if anomalies:
            fact_lines.append(f"Anomalies: {anomalies.get('anomalies_detected')} anomalous records ({anomalies.get('anomaly_rate_pct')}%) detected via Isolation Forest.")
        if ml:
            if "regression" in ml:
                fact_lines.append(f"ML Regression: Test R²={ml['regression']['metrics']['test_r2']}, RMSE={ml['regression']['metrics']['rmse']}.")
            if "forecasting" in ml:
                fact_lines.append(f"Forecasting: Model trend is '{ml['forecasting']['trend']}', Next forecast={ml['forecasting']['forecasts'][0]['forecast']}.")

        data_facts_text = "\n".join(fact_lines)

        if self.llm.is_configured():
            try:
                system_prompt = (
                    "You are a Principal Data Scientist and Business Strategy Advisor.\n"
                    "Generate rigorous, truthful analytical insights from the computed dataset facts.\n"
                    "CRITICAL RULES:\n"
                    "1. Never invent or hallucinate numbers. Every statistic must reference the provided facts.\n"
                    "2. Do NOT mistake correlation for causation.\n"
                    "3. Clearly separate: Observed Fact, Statistical Finding, AI Interpretation, and Strategic Recommendation.\n"
                    "Output ONLY valid JSON."
                )
                prompt = (
                    f"User Question: \"{question}\"\n\n"
                    f"COMPUTED DATASET FACTS:\n{data_facts_text}\n\n"
                    f"ANALYTICAL GUIDANCE:\n{rag_context}\n\n"
                    "Respond with JSON adhering to this schema:\n"
                    "{\n"
                    "  \"executive_summary\": \"1-2 sentence high-level summary directly answering the question.\",\n"
                    "  \"observed_facts\": [\"Fact 1 with computed numbers\", \"Fact 2 with computed numbers\"],\n"
                    "  \"statistical_findings\": [\"Statistical result with p-value or distribution note\"],\n"
                    "  \"ai_interpretations\": [\"Analytical interpretation explaining mechanism without unwarranted causal claims\"],\n"
                    "  \"recommendations\": [\"Actionable business recommendation 1\", \"Actionable business recommendation 2\"],\n"
                    "  \"limitations\": \"Any analytical caveats or data limitations\"\n"
                    "}"
                )
                res_str = self.llm.generate(prompt=prompt, system_prompt=system_prompt, json_mode=True)
                parsed = json.loads(re.sub(r'```json|```', '', res_str).strip())
                return parsed
            except Exception:
                pass

        # Deterministic Grounded Insights Fallback
        return self._generate_deterministic_insights(question, kpis, stats, ts, anomalies, ml)

    def _generate_deterministic_insights(
        self,
        question: str,
        kpis: Dict[str, Any],
        stats: Dict[str, Any],
        ts: Dict[str, Any],
        anomalies: Dict[str, Any],
        ml: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize verified, strictly computed findings when LLM is offline or unconfigured."""
        observed_facts = []
        statistical_findings = []
        interpretations = []
        recommendations = []

        # 1. Primary Metric facts
        pm = kpis.get("primary_metric", {})
        metric_name = pm.get("name", "Metric")
        if pm:
            observed_facts.append(
                f"Total {metric_name} stands at {pm['total']:,} across evaluated transactions, with an average of {pm['mean']:,} and median of {pm['median']:,}."
            )

        # 2. Segmentation facts
        top = kpis.get("top_performer")
        low = kpis.get("lowest_performer")
        if top and low:
            diff_pct = round(((top["mean"] - low["mean"]) / abs(low["mean"]) * 100), 1) if low["mean"] != 0 else 0.0
            observed_facts.append(
                f"Segment analysis across '{top['dimension']}' reveals '{top['category']}' as the highest-performing category (mean: {top['mean']:,}, total: {top['total']:,}), "
                f"outperforming the lowest segment '{low['category']}' (mean: {low['mean']:,}) by {diff_pct}%."
            )

        # 3. Growth / Temporal facts
        pg = kpis.get("period_growth")
        if pg:
            direction = "expanded" if pg["overall_growth_pct"] >= 0 else "contracted"
            observed_facts.append(
                f"Over observed time horizons, {metric_name} {direction} by {abs(pg['overall_growth_pct'])}% overall, "
                f"with the most recent period shifting by {pg['recent_period_growth_pct']}%."
            )

        # 4. Statistical findings
        if "bivariate_correlation" in stats:
            bc = stats["bivariate_correlation"]
            r_val = bc.get("pearson_r", 0.0)
            sig_text = "statistically significant (p < 0.05)" if bc.get("is_significant") else "not statistically significant (p >= 0.05)"
            statistical_findings.append(
                f"Calculated Pearson correlation between '{bc.get('var1')}' and '{bc.get('var2')}' is r = {r_val} ({bc.get('strength')} {bc.get('direction').lower()} association, {sig_text}). "
                f"Caution: Correlation denotes mathematical co-movement, not causal proof."
            )

        if "group_comparison" in stats:
            gc = stats["group_comparison"]
            sig_text = "indicates statistically significant variance across categories (p < 0.05)" if gc.get("is_significant") else "does not indicate statistically significant category variance (p >= 0.05)"
            statistical_findings.append(
                f"{gc.get('test')} on '{gc.get('metric')}' across '{gc.get('group_variable')}' yielded test statistic = {gc.get('statistic')}, p-value = {gc.get('p_value'):.4e}, which {sig_text}."
            )

        if anomalies:
            observed_facts.append(
                f"Isolated {anomalies.get('anomalies_detected')} anomalous records ({anomalies.get('anomaly_rate_pct')}% anomaly rate) exhibiting severe multi-variable deviations."
            )

        if ml and "regression" in ml:
            reg = ml["regression"]
            statistical_findings.append(
                f"Predictive model for '{reg['target_variable']}' achieved an R² score of {reg['metrics']['test_r2']} (RMSE: {reg['metrics']['rmse']}), with top influential feature '{reg['feature_importances'][0]['feature']}'."
            )

        # 5. Interpretations
        if top and low:
            interpretations.append(
                f"Disproportionate variance concentrated in '{low['category']}' suggests category-specific operational friction or margin erosion that dampens overall performance."
            )
        if "bivariate_correlation" in stats and abs(stats["bivariate_correlation"].get("pearson_r", 0)) > 0.4:
            bc = stats["bivariate_correlation"]
            interpretations.append(
                f"The observable {bc.get('direction').lower()} linkage between '{bc.get('var1')}' and '{bc.get('var2')}' points to interconnected operational dynamics warranting unified policy adjustments."
            )
        else:
            interpretations.append(
                f"Distribution characteristics suggest performance is driven by high-volume core segments, with tail transactions experiencing higher variability."
            )

        # 6. Recommendations
        if low:
            recommendations.append(
                f"Conduct a granular audit of '{low['category']}' operations and pricing structure to identify root drivers of performance drag."
            )
        if "bivariate_correlation" in stats and stats["bivariate_correlation"].get("pearson_r", 0) < -0.3:
            recommendations.append(
                f"Evaluate discount thresholds and promotion caps, as elevated discounts exhibit clear negative co-movement with net profitability."
            )
        if anomalies:
            recommendations.append(
                f"Establish automated alerting for the {anomalies.get('anomalies_detected')} flagged anomaly profiles to catch margin leakage and unusual spikes in real time."
            )
        recommendations.append(
            f"Double down on resource allocation toward '{top['category'] if top else 'leading segments'}' to capitalize on demonstrated comparative advantages."
        )

        exec_summary = (
            f"Analysis of '{question}' reveals a total {metric_name} of {pm.get('total', 0):,} "
            f"with top performance concentrated in '{top['category'] if top else 'key segments'}'. "
            f"Significant variance across sub-dimensions highlights targeted optimization opportunities."
        )

        return {
            "executive_summary": exec_summary,
            "observed_facts": observed_facts,
            "statistical_findings": statistical_findings,
            "ai_interpretations": interpretations,
            "recommendations": recommendations,
            "limitations": "Conclusions are strictly derived from empirical data present in the uploaded file. External macroeconomic and unmeasured variables are not accounted for."
        }
