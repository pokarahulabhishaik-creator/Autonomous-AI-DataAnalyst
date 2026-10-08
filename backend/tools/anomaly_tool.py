import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.ensemble import IsolationForest

class AnomalyDetectionTool:
    """
    Autonomous Anomaly & Outlier Detector.
    Uses Isolation Forest and statistical Z-score/IQR metrics to isolate unusual transactions.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def detect_anomalies(
        self,
        features: Optional[List[str]] = None,
        contamination: float = 0.03,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """Detect multi-dimensional and univariate anomalies using Isolation Forest."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if features:
            cols = [c for c in features if c in num_cols]
        else:
            cols = num_cols

        if len(cols) == 0:
            return {"error": "No numerical features available for anomaly detection."}

        clean_sub = self.df[cols].dropna()
        if len(clean_sub) < 15:
            return {"error": "Insufficient clean numerical rows for anomaly detection (< 15 rows)."}

        # Fit Isolation Forest
        iso = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        preds = iso.fit_predict(clean_sub)
        scores = iso.decision_function(clean_sub)

        # Flag anomalies (-1 from Isolation Forest)
        clean_sub["_is_anomaly"] = (preds == -1)
        clean_sub["_anomaly_score"] = -scores  # Higher score = more anomalous

        anomaly_indices = clean_sub[clean_sub["_is_anomaly"]].index
        total_anomalies = len(anomaly_indices)
        anomaly_rate_pct = round(total_anomalies / len(clean_sub) * 100, 2)

        # Merge with full original dataset rows to provide business context
        anomaly_df = self.df.loc[anomaly_indices].copy()
        anomaly_df["_anomaly_score"] = clean_sub.loc[anomaly_indices, "_anomaly_score"]
        anomaly_df = anomaly_df.sort_values(by="_anomaly_score", ascending=False)

        # Generate explanatory reasons for top anomalies
        reasons = []
        top_records = []

        means = {c: float(self.df[c].dropna().mean()) for c in cols}
        stds = {c: float(self.df[c].dropna().std()) for c in cols}

        for idx, row in anomaly_df.head(top_n).iterrows():
            row_dict = row.to_dict()
            # Find the most extreme variables for this row
            extreme_devs = []
            for c in cols:
                val = row_dict.get(c)
                if val is not None and stds[c] > 0:
                    z = (val - means[c]) / stds[c]
                    if abs(z) > 2.0:
                        extreme_devs.append(f"{c}={val} ({round(abs(z), 1)} std deviations {'above' if z > 0 else 'below'} mean)")

            reason_str = "; ".join(extreme_devs) if extreme_devs else "Unusual combination of feature values"
            row_dict["_anomaly_reason"] = reason_str
            clean_rec = {str(k): (round(v, 2) if isinstance(v, float) else v) for k, v in row_dict.items() if not str(k).startswith("_")}
            clean_rec["anomaly_reason"] = reason_str
            clean_rec["anomaly_score"] = round(float(row_dict["_anomaly_score"]), 3)
            top_records.append(clean_rec)

        summary = (
            f"Detected {total_anomalies} anomalous records ({anomaly_rate_pct}% anomaly rate) "
            f"using multi-dimensional Isolation Forest across {cols}. "
            f"Top anomalies exhibit extreme deviations in values (e.g. {top_records[0]['anomaly_reason'] if top_records else 'None'})."
        )

        return {
            "success": True,
            "total_records_evaluated": len(clean_sub),
            "anomalies_detected": total_anomalies,
            "anomaly_rate_pct": anomaly_rate_pct,
            "features_analyzed": cols,
            "top_anomalies": top_records,
            "summary": summary
        }
