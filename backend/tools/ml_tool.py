import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score
)
from sklearn.preprocessing import StandardScaler

class MLAnalysisTool:
    """
    Autonomous Machine Learning tool for AutoAnalyst AI.
    Executes Regression, Classification, Clustering, and Time-Series Forecasting
    with rigorous feature encoding and metric reporting.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _prepare_features(
        self,
        target_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        max_cardinality: int = 15
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
        """Preprocess and encode features cleanly."""
        data = self.df.copy()
        
        # If no explicit features specified, pick reasonable numerical and low-cardinality categorical
        if not feature_cols:
            exclude = [target_col] if target_col else []
            candidate_cols = [c for c in data.columns if c not in exclude]
            feature_cols = []
            for c in candidate_cols:
                # Exclude high cardinality strings or IDs
                if data[c].nunique() > len(data) * 0.7 and len(data) > 50:
                    continue
                if pd.api.types.is_numeric_dtype(data[c]):
                    feature_cols.append(c)
                elif data[c].nunique() <= max_cardinality:
                    feature_cols.append(c)

        if not feature_cols:
            raise ValueError("No suitable feature columns found for modeling.")

        clean_data = data[feature_cols + ([target_col] if target_col else [])].dropna()
        if len(clean_data) < 15:
            raise ValueError("Insufficient rows remaining after dropping missing values (< 15 rows).")

        X_raw = clean_data[feature_cols]
        y_raw = clean_data[target_col] if target_col else None

        # One-hot encode categoricals in X
        X_encoded = pd.get_dummies(X_raw, drop_first=True)
        # Ensure all columns are numeric
        X_encoded = X_encoded.astype(float)

        return X_encoded, y_raw, list(X_encoded.columns)

    def train_regression(
        self,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        model_type: str = "random_forest"
    ) -> Dict[str, Any]:
        """Train regression model to predict continuous numeric target."""
        if target_col not in self.df.columns:
            return {"error": f"Target column '{target_col}' not found in dataset."}

        try:
            X, y, feature_names = self._prepare_features(target_col=target_col, feature_cols=feature_cols)
            y = pd.to_numeric(y, errors='coerce')
            valid_mask = y.notna()
            X, y = X[valid_mask], y[valid_mask]

            if len(X) < 20:
                return {"error": "Need at least 20 samples to train a meaningful regression model."}

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

            if model_type == "linear":
                model = Ridge(alpha=1.0)
                model_name = "Ridge Linear Regression"
            else:
                model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
                model_name = "Random Forest Regressor"

            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            r2 = float(r2_score(y_test, y_pred_test))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
            mae = float(mean_absolute_error(y_test, y_pred_test))
            train_r2 = float(r2_score(y_train, y_pred_train))

            # Feature importances
            importances = []
            if hasattr(model, "feature_importances_"):
                raw_imp = model.feature_importances_
                for f_name, imp in zip(feature_names, raw_imp):
                    importances.append({"feature": f_name, "importance": round(float(imp), 4)})
            elif hasattr(model, "coef_"):
                for f_name, coef in zip(feature_names, model.coef_):
                    importances.append({"feature": f_name, "importance": round(float(abs(coef)), 4)})
            importances.sort(key=lambda x: x["importance"], reverse=True)

            # Sample predictions comparison
            test_samples = []
            for actual, pred in list(zip(y_test.values, y_pred_test))[:10]:
                test_samples.append({
                    "actual": round(float(actual), 2),
                    "predicted": round(float(pred), 2),
                    "absolute_error": round(float(abs(actual - pred)), 2)
                })

            summary = (
                f"{model_name} trained to predict '{target_col}' using {len(feature_names)} features. "
                f"Test R² = {round(r2, 3)} (explains {round(r2 * 100, 1)}% of variance), "
                f"RMSE = {round(rmse, 2)}, MAE = {round(mae, 2)}. "
                f"Top influential predictor: '{importances[0]['feature']}' (importance: {importances[0]['importance']})."
            )

            return {
                "success": True,
                "task": "Regression",
                "model_name": model_name,
                "target_variable": target_col,
                "features_used": feature_names,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "metrics": {
                    "test_r2": round(r2, 4),
                    "train_r2": round(train_r2, 4),
                    "rmse": round(rmse, 2),
                    "mae": round(mae, 2)
                },
                "feature_importances": importances[:8],
                "sample_predictions": test_samples,
                "summary": summary,
                "limitations": "Model performance is conditioned on the provided historical dataset features. Extrapolation outside historical feature ranges may degrade accuracy."
            }
        except Exception as e:
            return {"error": f"Regression modeling error: {str(e)}"}

    def train_classification(
        self,
        target_col: str,
        feature_cols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Train classification model to predict categorical target (e.g. churn)."""
        if target_col not in self.df.columns:
            return {"error": f"Target column '{target_col}' not found in dataset."}

        try:
            X, y_raw, feature_names = self._prepare_features(target_col=target_col, feature_cols=feature_cols)
            y = y_raw.astype(str)

            classes = sorted(list(y.unique()))
            if len(classes) < 2:
                return {"error": f"Target '{target_col}' has fewer than 2 distinct classes."}
            if len(classes) > 10:
                return {"error": f"Target '{target_col}' has too many classes ({len(classes)}) for standard classification."}

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

            model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))
            f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            prec = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            rec = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))

            # Feature importances
            importances = []
            for f_name, imp in zip(feature_names, model.feature_importances_):
                importances.append({"feature": f_name, "importance": round(float(imp), 4)})
            importances.sort(key=lambda x: x["importance"], reverse=True)

            cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()

            summary = (
                f"Random Forest Classifier trained to predict '{target_col}' across classes {classes}. "
                f"Test Accuracy = {round(acc * 100, 1)}%, Weighted F1-Score = {round(f1, 3)}, "
                f"Precision = {round(prec, 3)}, Recall = {round(rec, 3)}. "
                f"Leading factor influencing outcome: '{importances[0]['feature']}'."
            )

            return {
                "success": True,
                "task": "Classification",
                "model_name": "Random Forest Classifier",
                "target_variable": target_col,
                "classes": classes,
                "features_used": feature_names,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "metrics": {
                    "accuracy": round(acc, 4),
                    "f1_score": round(f1, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4)
                },
                "feature_importances": importances[:8],
                "confusion_matrix": {"classes": classes, "matrix": cm},
                "summary": summary,
                "limitations": "Ensure class representation is balanced. Unbalanced datasets can yield high accuracy while underperforming on minority classes."
            }
        except Exception as e:
            return {"error": f"Classification modeling error: {str(e)}"}

    def perform_clustering(
        self,
        n_clusters: int = 3,
        feature_cols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform K-Means clustering segmentation."""
        try:
            num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if feature_cols:
                cols = [c for c in feature_cols if c in num_cols]
            else:
                cols = num_cols[:5]

            if len(cols) < 2:
                return {"error": "Need at least 2 numerical features for clustering segmentation."}

            clean_data = self.df[cols].dropna()
            if len(clean_data) < 20:
                return {"error": "Insufficient rows for clustering (< 20 rows)."}

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(clean_data)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            sil_score = float(silhouette_score(X_scaled, labels)) if len(set(labels)) > 1 else 0.0

            clean_data["Cluster"] = [f"Cluster {i+1}" for i in labels]
            cluster_summary = []
            for cl in sorted(clean_data["Cluster"].unique()):
                sub = clean_data[clean_data["Cluster"] == cl]
                profile = {"cluster": cl, "count": int(len(sub)), "share_pct": round(len(sub) / len(clean_data) * 100, 1)}
                for col in cols:
                    profile[f"{col}_mean"] = round(float(sub[col].mean()), 2)
                cluster_summary.append(profile)

            summary = (
                f"K-Means clustered dataset into {n_clusters} segments using features {cols}. "
                f"Silhouette score is {round(sil_score, 3)} "
                f"({'Good structure' if sil_score > 0.4 else 'Moderate/overlapping clusters'})."
            )

            return {
                "success": True,
                "task": "Clustering",
                "model_name": "K-Means Clustering",
                "n_clusters": n_clusters,
                "silhouette_score": round(sil_score, 3),
                "features_used": cols,
                "clusters": cluster_summary,
                "summary": summary
            }
        except Exception as e:
            return {"error": f"Clustering error: {str(e)}"}

    def forecast_metric(
        self,
        metric_col: str,
        date_col: str,
        horizon_periods: int = 3,
        freq: str = "M"
    ) -> Dict[str, Any]:
        """Simple time series trend forecasting using autoregression / trend regression."""
        try:
            clean = self.df[[date_col, metric_col]].dropna().copy()
            clean[date_col] = pd.to_datetime(clean[date_col], errors='coerce')
            clean[metric_col] = pd.to_numeric(clean[metric_col], errors='coerce')
            clean = clean.dropna().sort_values(date_col)

            if len(clean) < 10:
                return {"error": "Need at least 10 observations with valid dates for forecasting."}

            # Aggregate by period
            grouped = clean.set_index(date_col).resample(freq)[metric_col].sum().reset_index()
            grouped = grouped.dropna()

            if len(grouped) < 4:
                # Try daily or weekly if monthly has too few periods
                grouped = clean.set_index(date_col).resample('W')[metric_col].sum().reset_index().dropna()
                freq = 'W'

            if len(grouped) < 4:
                return {"error": "Insufficient time-series aggregated points for forecasting."}

            # Build index-based regression
            n = len(grouped)
            X = np.arange(n).reshape(-1, 1)
            y = grouped[metric_col].values

            # Fit polynomial trend (degree 1)
            slope, intercept = np.polyfit(X.flatten(), y, 1)
            residuals = y - (slope * X.flatten() + intercept)
            std_err = float(np.std(residuals))

            # Forecast next horizon periods
            future_X = np.arange(n, n + horizon_periods).reshape(-1, 1)
            future_preds = slope * future_X.flatten() + intercept

            last_dt = grouped[date_col].iloc[-1]
            future_dates = []
            for i in range(1, horizon_periods + 1):
                if freq == 'M':
                    next_dt = last_dt + pd.DateOffset(months=i)
                else:
                    next_dt = last_dt + pd.DateOffset(weeks=i)
                future_dates.append(str(next_dt.strftime('%Y-%m-%d')))

            forecast_records = []
            for dt_str, pred in zip(future_dates, future_preds):
                forecast_records.append({
                    "date": dt_str,
                    "forecast": round(float(pred), 2),
                    "lower_bound_95": round(float(pred - 1.96 * std_err), 2),
                    "upper_bound_95": round(float(pred + 1.96 * std_err), 2)
                })

            trend_direction = "Growth (Upward trend)" if slope > 0 else "Decline (Downward trend)"

            summary = (
                f"Historical time-series trend model fitted across {n} periods ({freq} frequency). "
                f"Detected overall {trend_direction} with slope = {round(slope, 2)} units per period. "
                f"Next period forecast is {round(future_preds[0], 2)} "
                f"[95% CI: {round(future_preds[0] - 1.96 * std_err, 2)} to {round(future_preds[0] + 1.96 * std_err, 2)}]."
            )

            return {
                "success": True,
                "task": "Time Series Forecasting",
                "metric": metric_col,
                "date_column": date_col,
                "frequency": freq,
                "historical_points": n,
                "slope": round(float(slope), 2),
                "trend": trend_direction,
                "forecasts": forecast_records,
                "summary": summary,
                "limitations": "Linear trend extrapolation assumes past velocity and macroeconomic conditions persist without structural shocks."
            }
        except Exception as e:
            return {"error": f"Forecasting error: {str(e)}"}
