import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import re

class DataProfiler:
    """
    Autonomous Data Profiler.
    Inspects, characterizes, and scores arbitrary CSV/Excel datasets.
    """

    def __init__(self, df: pd.DataFrame, filename: str = "dataset"):
        self.df = df
        self.filename = filename

    def profile(self) -> Dict[str, Any]:
        """Perform comprehensive statistical and semantic profiling."""
        total_rows = int(len(self.df))
        total_cols = int(len(self.df.columns))
        memory_usage_mb = round(float(self.df.memory_usage(deep=True).sum()) / (1024 * 1024), 2)
        
        # Categorize columns
        num_cols = []
        cat_cols = []
        date_cols = []
        id_cols = []
        constant_cols = []
        high_cardinality_cols = []
        
        for col in self.df.columns:
            series = self.df[col]
            n_unique = int(series.nunique(dropna=True))
            
            # Constant check
            if n_unique <= 1:
                constant_cols.append(col)
                
            # Date check
            if pd.api.types.is_datetime64_any_dtype(series):
                date_cols.append(col)
            elif series.dtype == 'object' or isinstance(series.dtype, pd.StringDtype):
                # Try sniffing date format if column name suggests date or sample values parse
                col_lower = str(col).lower()
                is_likely_date = any(k in col_lower for k in ['date', 'time', 'timestamp', 'day', 'month', 'year'])
                if is_likely_date:
                    try:
                        sample = series.dropna().head(20)
                        if len(sample) > 0:
                            pd.to_datetime(sample, errors='raise')
                            date_cols.append(col)
                            continue
                    except Exception:
                        pass
                
                # ID check: high uniqueness relative to rows and contains id/uuid/key
                is_likely_id = any(k in col_lower for k in ['id', 'uuid', 'guid', 'key', 'code', 'hash'])
                if (n_unique > 0.8 * total_rows and total_rows > 50) or (is_likely_id and n_unique > 0.5 * total_rows):
                    id_cols.append(col)
                elif n_unique > 50 and n_unique > 0.5 * total_rows:
                    high_cardinality_cols.append(col)
                else:
                    cat_cols.append(col)
                    
            elif pd.api.types.is_numeric_dtype(series):
                col_lower = str(col).lower()
                is_likely_id = any(k in col_lower for k in ['id', 'uuid', 'key', 'zip', 'fips'])
                if is_likely_id and n_unique > 0.8 * total_rows:
                    id_cols.append(col)
                elif n_unique == 2:
                    # Binary flag
                    cat_cols.append(col)
                else:
                    num_cols.append(col)
            else:
                cat_cols.append(col)

        # Missing values & Duplicates
        missing_by_col = {col: int(self.df[col].isna().sum()) for col in self.df.columns}
        missing_pct_by_col = {
            col: round((cnt / total_rows * 100), 2) if total_rows > 0 else 0.0
            for col, cnt in missing_by_col.items()
        }
        total_missing_cells = int(sum(missing_by_col.values()))
        total_cells = total_rows * total_cols
        overall_missing_pct = round((total_missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0
        
        duplicate_rows = int(self.df.duplicated().sum())
        duplicate_pct = round((duplicate_rows / total_rows * 100), 2) if total_rows > 0 else 0.0

        # Detailed Numerical Stats & Outliers
        numerical_stats = {}
        outlier_counts = {}
        for col in num_cols:
            s = self.df[col].dropna()
            if len(s) == 0:
                continue
            q25 = float(s.quantile(0.25))
            q75 = float(s.quantile(0.75))
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outliers = int(((s < lower_bound) | (s > upper_bound)).sum())
            outlier_counts[col] = outliers
            
            skewness = float(s.skew()) if len(s) > 2 else 0.0
            numerical_stats[col] = {
                "mean": round(float(s.mean()), 2),
                "std": round(float(s.std()), 2) if len(s) > 1 else 0.0,
                "median": round(float(s.median()), 2),
                "min": round(float(s.min()), 2),
                "max": round(float(s.max()), 2),
                "q25": round(q25, 2),
                "q75": round(q75, 2),
                "iqr": round(iqr, 2),
                "skewness": round(skewness, 2),
                "outliers_count": outliers,
                "outliers_pct": round(outliers / len(s) * 100, 2) if len(s) > 0 else 0.0
            }

        # Categorical Summary
        categorical_stats = {}
        for col in cat_cols:
            s = self.df[col].dropna()
            if len(s) == 0:
                continue
            val_counts = s.value_counts().head(5).to_dict()
            # Clean string keys
            clean_counts = {str(k): int(v) for k, v in val_counts.items()}
            categorical_stats[col] = {
                "unique_count": int(s.nunique()),
                "top_categories": clean_counts,
                "mode": str(s.mode()[0]) if not s.mode().empty else None
            }

        # Date Range Summary
        date_stats = {}
        for col in date_cols:
            try:
                s = pd.to_datetime(self.df[col].dropna(), errors='coerce').dropna()
                if len(s) > 0:
                    min_dt = s.min()
                    max_dt = s.max()
                    date_stats[col] = {
                        "min_date": str(min_dt.strftime("%Y-%m-%d")),
                        "max_date": str(max_dt.strftime("%Y-%m-%d")),
                        "span_days": int((max_dt - min_dt).days)
                    }
            except Exception:
                pass

        # Correlation Analysis
        correlation_matrix = {}
        top_correlations = []
        if len(num_cols) >= 2:
            try:
                corr_df = self.df[num_cols].corr()
                # Clean correlation matrix
                for c1 in num_cols:
                    correlation_matrix[c1] = {}
                    for c2 in num_cols:
                        val = corr_df.loc[c1, c2]
                        correlation_matrix[c1][c2] = round(float(val), 3) if not np.isnan(val) else 0.0
                        
                # Extract top strong correlations (|r| >= 0.5, c1 < c2)
                for i in range(len(num_cols)):
                    for j in range(i + 1, len(num_cols)):
                        c1, c2 = num_cols[i], num_cols[j]
                        r_val = corr_df.loc[c1, c2]
                        if not np.isnan(r_val) and abs(r_val) >= 0.3:
                            top_correlations.append({
                                "var1": c1,
                                "var2": c2,
                                "correlation": round(float(r_val), 3),
                                "strength": "Strong" if abs(r_val) >= 0.7 else ("Moderate" if abs(r_val) >= 0.5 else "Weak-to-Moderate"),
                                "direction": "Positive" if r_val > 0 else "Negative"
                            })
                top_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            except Exception:
                pass

        # Potential Target Candidates
        target_candidates = []
        target_keywords = ['target', 'label', 'churn', 'revenue', 'sales', 'profit', 'price', 'converted', 'status', 'default', 'salary', 'rating']
        for col in self.df.columns:
            col_l = str(col).lower()
            if any(k in col_l for k in target_keywords):
                target_candidates.append(col)

        # Compute Data Quality Score (0 - 100)
        # Completeness (40% weight): 100 - overall_missing_pct
        # Uniqueness (30% weight): 100 - duplicate_pct
        # Validity & Outliers (30% weight): proportion of non-extreme rows
        completeness_score = max(0.0, 100.0 - overall_missing_pct)
        uniqueness_score = max(0.0, 100.0 - (duplicate_pct * 2.0))
        total_num_values = sum(len(self.df[c].dropna()) for c in num_cols) if num_cols else 1
        total_outliers = sum(outlier_counts.values()) if outlier_counts else 0
        outlier_ratio = (total_outliers / total_num_values) * 100 if total_num_values > 0 else 0
        validity_score = max(0.0, 100.0 - (outlier_ratio * 1.5))
        
        quality_score = round(0.40 * completeness_score + 0.30 * uniqueness_score + 0.30 * validity_score, 1)

        profile_summary = {
            "filename": self.filename,
            "dimensions": {
                "rows": total_rows,
                "columns": total_cols,
                "memory_mb": memory_usage_mb,
                "total_cells": total_cells
            },
            "columns": {
                "all": list(self.df.columns),
                "numerical": num_cols,
                "categorical": cat_cols,
                "datetime": date_cols,
                "id_columns": id_cols,
                "constant_columns": constant_cols,
                "high_cardinality": high_cardinality_cols
            },
            "data_quality": {
                "score": quality_score,
                "rating": "Excellent" if quality_score >= 90 else ("Good" if quality_score >= 75 else ("Fair" if quality_score >= 60 else "Needs Attention")),
                "total_missing_cells": total_missing_cells,
                "overall_missing_pct": overall_missing_pct,
                "missing_by_column": missing_by_col,
                "missing_pct_by_column": missing_pct_by_col,
                "duplicate_rows": duplicate_rows,
                "duplicate_pct": duplicate_pct,
                "total_outliers": total_outliers
            },
            "numerical_statistics": numerical_stats,
            "categorical_statistics": categorical_stats,
            "date_statistics": date_stats,
            "correlation_matrix": correlation_matrix,
            "top_correlations": top_correlations[:8],
            "target_candidates": target_candidates
        }

        return profile_summary

    def get_agent_context(self) -> str:
        """Create a concise string overview of dataset profile suitable for LLM prompt context."""
        p = self.profile()
        dims = p["dimensions"]
        cols = p["columns"]
        dq = p["data_quality"]

        num_summary = []
        for c, s in list(p["numerical_statistics"].items())[:6]:
            num_summary.append(f"{c} (mean={s['mean']}, median={s['median']}, min={s['min']}, max={s['max']})")

        cat_summary = []
        for c, s in list(p["categorical_statistics"].items())[:6]:
            cats = list(s["top_categories"].keys())[:3]
            cat_summary.append(f"{c} ({s['unique_count']} unique: {', '.join(cats)})")

        context = (
            f"DATASET CONTEXT:\n"
            f"- File: {p['filename']} | Rows: {dims['rows']} | Cols: {dims['columns']} | Quality Score: {dq['score']}/100 ({dq['rating']})\n"
            f"- Numerical Columns: {', '.join(cols['numerical']) if cols['numerical'] else 'None'}\n"
            f"- Categorical Columns: {', '.join(cols['categorical']) if cols['categorical'] else 'None'}\n"
            f"- Datetime Columns: {', '.join(cols['datetime']) if cols['datetime'] else 'None'}\n"
            f"- ID/Key Columns: {', '.join(cols['id_columns']) if cols['id_columns'] else 'None'}\n"
            f"- Missing Cells: {dq['total_missing_cells']} ({dq['overall_missing_pct']}%) | Duplicate Rows: {dq['duplicate_rows']}\n"
            f"- Numerical Distributions: {'; '.join(num_summary)}\n"
            f"- Categorical Dimensions: {'; '.join(cat_summary)}\n"
            f"- Target Candidates: {', '.join(p['target_candidates']) if p['target_candidates'] else 'None'}"
        )
        return context
