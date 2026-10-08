import re
import json
from typing import Dict, Any, List, Optional
from backend.llm.client import LLMClient
from backend.rag.retriever import AnalyticsRetriever

class QuestionAnalyzer:
    """
    Autonomous Question Analyzer.
    Deconstructs user questions into analytical intent, maps entities to arbitrary dataset columns,
    and identifies required metric/dimension focal points.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.retriever = AnalyticsRetriever()

    def analyze(self, question: str, dataset_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze user question against the dataset profile.
        Returns structured analysis intent and entity mappings.
        """
        q_clean = question.strip()
        num_cols = dataset_profile.get("columns", {}).get("numerical", [])
        cat_cols = dataset_profile.get("columns", {}).get("categorical", [])
        date_cols = dataset_profile.get("columns", {}).get("datetime", [])
        all_cols = dataset_profile.get("columns", {}).get("all", [])

        # Retrieve relevant analytical knowledge from RAG
        rag_guidelines = self.retriever.get_context_string(q_clean, top_k=2)

        if self.llm.is_configured():
            try:
                system_prompt = (
                    "You are an expert AI Data Architect analyzing a user question about a tabular dataset.\n"
                    "Your job is to identify the analytical intent, map user entities to available columns, and specify required operations.\n"
                    "Output ONLY valid JSON."
                )
                prompt = (
                    f"User Question: \"{q_clean}\"\n\n"
                    f"Dataset Columns:\n"
                    f"- Numerical: {num_cols}\n"
                    f"- Categorical: {cat_cols}\n"
                    f"- Datetime: {date_cols}\n\n"
                    f"Relevant Analytical Guidance:\n{rag_guidelines}\n\n"
                    "Respond with JSON adhering to this schema:\n"
                    "{\n"
                    "  \"intent\": \"trend_analysis | root_cause | ranking_comparison | anomaly_detection | correlation_driver | forecasting_ml | segmentation | general_overview\",\n"
                    "  \"primary_metric\": \"<exact numerical column or null>\",\n"
                    "  \"secondary_metric\": \"<exact numerical column or null>\",\n"
                    "  \"primary_dimension\": \"<exact categorical/date column or null>\",\n"
                    "  \"time_column\": \"<exact date column or null>\",\n"
                    "  \"requires_time_series\": true/false,\n"
                    "  \"requires_ml\": true/false,\n"
                    "  \"requires_anomaly\": true/false,\n"
                    "  \"recommended_chart\": \"line | bar | scatter | histogram | box | heatmap | pie\",\n"
                    "  \"reasoning\": \"Brief explanation of intent mapping\"\n"
                    "}"
                )
                res_str = self.llm.generate(prompt=prompt, system_prompt=system_prompt, json_mode=True)
                # Parse JSON
                parsed = json.loads(re.sub(r'```json|```', '', res_str).strip())
                # Validate columns exist in dataset
                if parsed.get("primary_metric") and parsed["primary_metric"] not in all_cols:
                    parsed["primary_metric"] = self._fuzzy_match_column(parsed["primary_metric"], num_cols)
                if parsed.get("primary_dimension") and parsed["primary_dimension"] not in all_cols:
                    parsed["primary_dimension"] = self._fuzzy_match_column(parsed["primary_dimension"], cat_cols + date_cols)
                if parsed.get("time_column") and parsed["time_column"] not in all_cols:
                    parsed["time_column"] = date_cols[0] if date_cols else None
                return parsed
            except Exception:
                pass

        # Robust Heuristic Semantic Analyzer Fallback
        return self._heuristic_analyze(q_clean, dataset_profile)

    def _heuristic_analyze(self, q: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        q_lower = q.lower()
        num_cols = profile.get("columns", {}).get("numerical", [])
        cat_cols = profile.get("columns", {}).get("categorical", [])
        date_cols = profile.get("columns", {}).get("datetime", [])

        # 1. Determine Intent
        intent = "general_overview"
        requires_time_series = False
        requires_ml = False
        requires_anomaly = False
        recommended_chart = "bar"

        if any(w in q_lower for w in ["predict", "forecast", "future", "next month", "next year", "churn", "cluster", "segment"]):
            intent = "forecasting_ml"
            requires_ml = True
            recommended_chart = "line" if "predict" in q_lower or "forecast" in q_lower else "scatter"
        elif any(w in q_lower for w in ["anomal", "outlier", "unusual", "strange", "fraud", "spike"]):
            intent = "anomaly_detection"
            requires_anomaly = True
            recommended_chart = "scatter"
        elif any(w in q_lower for w in ["trend", "over time", "month", "year", "quarter", "timeline", "growth"]):
            intent = "trend_analysis"
            requires_time_series = bool(date_cols)
            recommended_chart = "line"
        elif any(w in q_lower for w in ["why", "cause", "factor", "affect", "impact", "driver", "decreas", "increas"]):
            intent = "root_cause"
            recommended_chart = "scatter" if len(num_cols) >= 2 else "bar"
        elif any(w in q_lower for w in ["correlat", "relation", "association", "vs", "versus"]):
            intent = "correlation_driver"
            recommended_chart = "scatter"
        elif any(w in q_lower for w in ["which", "best", "worst", "top", "bottom", "poor", "highest", "lowest", "compare", "rank"]):
            intent = "ranking_comparison"
            recommended_chart = "bar"

        # 2. Match Primary Metric
        primary_metric = self._find_best_metric(q_lower, num_cols)
        
        # 3. Match Secondary Metric (e.g. for profit vs sales, discount vs profit)
        secondary_metric = None
        for col in num_cols:
            if col != primary_metric and col.lower() in q_lower:
                secondary_metric = col
                break

        # 4. Match Primary Dimension
        primary_dimension = self._find_best_dimension(q_lower, cat_cols, date_cols, requires_time_series)

        # 5. Time column
        time_col = date_cols[0] if (date_cols and (requires_time_series or intent in ("trend_analysis", "forecasting_ml"))) else None

        return {
            "intent": intent,
            "primary_metric": primary_metric,
            "secondary_metric": secondary_metric,
            "primary_dimension": primary_dimension,
            "time_column": time_col,
            "requires_time_series": requires_time_series or bool(time_col and intent == "trend_analysis"),
            "requires_ml": requires_ml,
            "requires_anomaly": requires_anomaly,
            "recommended_chart": recommended_chart,
            "reasoning": f"Identified intent '{intent}' focused on metric '{primary_metric}' across dimension '{primary_dimension}'."
        }

    def _find_best_metric(self, q_lower: str, num_cols: List[str]) -> Optional[str]:
        if not num_cols:
            return None
        # Exact mention in query
        for col in num_cols:
            if col.lower() in q_lower:
                return col
        # Semantic keywords matching
        synonyms = {
            "profit": ["profit", "margin", "earnings", "net", "gain", "return"],
            "sales": ["sales", "revenue", "turnover", "income", "amount", "total"],
            "discount": ["discount", "rebate", "markdown", "promo"],
            "cost": ["cost", "expense", "spend", "charge"],
            "quantity": ["quantity", "volume", "units", "count", "orders"],
            "churn": ["churn", "retention", "tenure", "calls"]
        }
        for syn_key, words in synonyms.items():
            if any(w in q_lower for w in words):
                for col in num_cols:
                    if any(w in col.lower() for w in words):
                        return col
        # Default to first non-ID numeric column
        return num_cols[0]

    def _find_best_dimension(self, q_lower: str, cat_cols: List[str], date_cols: List[str], time_pref: bool) -> Optional[str]:
        if time_pref and date_cols:
            return date_cols[0]
        # Exact mention
        for col in cat_cols:
            if col.lower() in q_lower:
                return col
        # Semantic keywords
        dim_synonyms = {
            "product": ["product", "item", "sub_category", "sku", "category"],
            "region": ["region", "area", "zone", "state", "city", "country", "territory"],
            "customer": ["customer", "segment", "client", "user", "account"],
            "contract": ["contract", "plan", "subscription", "tier"]
        }
        for dim_key, words in dim_synonyms.items():
            if any(w in q_lower for w in words):
                for col in cat_cols:
                    if any(w in col.lower() for w in words):
                        return col
        return cat_cols[0] if cat_cols else (date_cols[0] if date_cols else None)

    def _fuzzy_match_column(self, candidate: str, available_cols: List[str]) -> Optional[str]:
        if not available_cols:
            return None
        c_low = candidate.lower()
        for col in available_cols:
            if col.lower() == c_low or col.lower() in c_low or c_low in col.lower():
                return col
        return available_cols[0]
