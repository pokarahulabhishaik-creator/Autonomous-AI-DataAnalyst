import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Dict, Any, List, Optional, Tuple

class StatisticsTool:
    """
    Autonomous Statistical Analysis Tool.
    Performs descriptive statistics, correlation matrices with p-values,
    and hypothesis tests (t-tests, ANOVA, Chi-square).
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def calculate_descriptive_stats(self, column: str) -> Dict[str, Any]:
        """Compute exhaustive descriptive statistics and 95% confidence intervals."""
        if column not in self.df.columns:
            return {"error": f"Column '{column}' not found in dataset."}

        s = pd.to_numeric(self.df[column], errors='coerce').dropna()
        if len(s) == 0:
            return {"error": f"Column '{column}' has no valid numeric values."}

        mean_val = float(s.mean())
        std_val = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        n = len(s)
        
        # 95% Confidence Interval for the mean
        if n > 1 and std_val > 0:
            se = std_val / np.sqrt(n)
            ci_low, ci_high = stats.t.interval(0.95, df=n-1, loc=mean_val, scale=se)
        else:
            ci_low, ci_high = mean_val, mean_val

        q25 = float(s.quantile(0.25))
        q50 = float(s.median())
        q75 = float(s.quantile(0.75))
        iqr = q75 - q25

        mode_res = s.mode()
        mode_val = float(mode_res[0]) if not mode_res.empty else None

        skew = float(s.skew()) if n > 2 else 0.0
        kurt = float(s.kurtosis()) if n > 3 else 0.0

        return {
            "column": column,
            "sample_size": n,
            "mean": round(mean_val, 2),
            "median": round(q50, 2),
            "mode": round(mode_val, 2) if mode_val is not None else None,
            "std": round(std_val, 2),
            "variance": round(float(s.var(ddof=1)), 2) if n > 1 else 0.0,
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
            "range": round(float(s.max() - s.min()), 2),
            "q25": round(q25, 2),
            "q75": round(q75, 2),
            "iqr": round(iqr, 2),
            "p90": round(float(s.quantile(0.90)), 2),
            "p95": round(float(s.quantile(0.95)), 2),
            "p99": round(float(s.quantile(0.99)), 2),
            "ci_95_low": round(float(ci_low), 2),
            "ci_95_high": round(float(ci_high), 2),
            "skewness": round(skew, 2),
            "kurtosis": round(kurt, 2),
            "distribution_shape": "Positively Skewed (Right-tailed)" if skew > 1 else (
                "Negatively Skewed (Left-tailed)" if skew < -1 else "Approximately Symmetric"
            )
        }

    def calculate_correlation(self, col1: str, col2: str) -> Dict[str, Any]:
        """Compute Pearson and Spearman correlation with p-value significance."""
        clean_data = self.df[[col1, col2]].dropna()
        if len(clean_data) < 3:
            return {"error": "Insufficient data points for correlation calculation."}

        x = pd.to_numeric(clean_data[col1], errors='coerce')
        y = pd.to_numeric(clean_data[col2], errors='coerce')
        valid_mask = x.notna() & y.notna()
        x, y = x[valid_mask], y[valid_mask]

        if len(x) < 3 or x.std() == 0 or y.std() == 0:
            return {"error": "Zero variance or insufficient valid data."}

        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)

        # Strength description
        abs_r = abs(pearson_r)
        if abs_r >= 0.7:
            strength = "Strong"
        elif abs_r >= 0.4:
            strength = "Moderate"
        elif abs_r >= 0.2:
            strength = "Weak"
        else:
            strength = "Negligible"

        direction = "Positive" if pearson_r > 0 else "Negative"

        interpretation = (
            f"There is a {strength.lower()} {direction.lower()} linear correlation (r = {round(pearson_r, 3)}, "
            f"p = {pearson_p:.4e}). "
            f"{'The relationship is statistically significant (p < 0.05).' if pearson_p < 0.05 else 'The relationship is not statistically significant (p >= 0.05).'}\n"
            f"CRITICAL REMINDER: Correlation does NOT prove causation. Confounding business variables or operational policies may drive both metrics."
        )

        return {
            "var1": col1,
            "var2": col2,
            "sample_size": len(x),
            "pearson_r": round(float(pearson_r), 3),
            "pearson_p_value": float(pearson_p),
            "spearman_r": round(float(spearman_r), 3),
            "spearman_p_value": float(spearman_p),
            "is_significant": bool(pearson_p < 0.05),
            "strength": strength,
            "direction": direction,
            "interpretation": interpretation
        }

    def perform_group_comparison(self, metric_col: str, group_col: str) -> Dict[str, Any]:
        """
        Compare a continuous metric across categories using Welch's t-test (2 groups) or One-way ANOVA (3+ groups).
        """
        clean = self.df[[metric_col, group_col]].dropna()
        clean[metric_col] = pd.to_numeric(clean[metric_col], errors='coerce')
        clean = clean.dropna()

        groups = clean[group_col].unique()
        group_data = [clean[clean[group_col] == g][metric_col].values for g in groups if len(clean[clean[group_col] == g]) > 1]
        active_groups = [g for g in groups if len(clean[clean[group_col] == g]) > 1]

        if len(group_data) < 2:
            return {"error": "Need at least 2 categories with >1 observation each to compare."}

        # Calculate per-group stats
        summary_by_group = []
        for g in active_groups:
            vals = clean[clean[group_col] == g][metric_col]
            summary_by_group.append({
                "group": str(g),
                "count": int(len(vals)),
                "mean": round(float(vals.mean()), 2),
                "median": round(float(vals.median()), 2),
                "std": round(float(vals.std()), 2) if len(vals) > 1 else 0.0,
                "sum": round(float(vals.sum()), 2)
            })
        summary_by_group.sort(key=lambda x: x["mean"], reverse=True)

        test_name = ""
        stat_val = 0.0
        p_val = 1.0

        if len(group_data) == 2:
            test_name = "Welch's Two-Sample t-test"
            stat_res = stats.ttest_ind(group_data[0], group_data[1], equal_var=False)
            stat_val = float(stat_res.statistic)
            p_val = float(stat_res.pvalue)
        else:
            test_name = "One-way ANOVA (F-test)"
            stat_res = stats.f_oneway(*group_data)
            stat_val = float(stat_res.statistic)
            p_val = float(stat_res.pvalue)

        is_sig = bool(p_val < 0.05)
        highest_grp = summary_by_group[0]
        lowest_grp = summary_by_group[-1]

        diff_pct = 0.0
        if lowest_grp["mean"] != 0:
            diff_pct = round(((highest_grp["mean"] - lowest_grp["mean"]) / abs(lowest_grp["mean"])) * 100, 1)

        finding = (
            f"{test_name} between '{group_col}' categories on '{metric_col}': "
            f"stat={round(stat_val, 3)}, p-value={p_val:.4e}. "
            f"{'Statistically significant difference detected (p < 0.05).' if is_sig else 'No statistically significant difference found (p >= 0.05).'}\n"
            f"Top performer is '{highest_grp['group']}' (mean: {highest_grp['mean']}) vs lowest '{lowest_grp['group']}' (mean: {lowest_grp['mean']}), "
            f"a difference of {diff_pct}%."
        )

        return {
            "test": test_name,
            "metric": metric_col,
            "group_variable": group_col,
            "num_groups": len(active_groups),
            "statistic": round(stat_val, 3),
            "p_value": p_val,
            "is_significant": is_sig,
            "group_breakdown": summary_by_group,
            "summary_finding": finding
        }

    def chi_square_test(self, cat_col1: str, cat_col2: str) -> Dict[str, Any]:
        """Chi-square test of independence between two categorical variables."""
        clean = self.df[[cat_col1, cat_col2]].dropna()
        if len(clean) < 10:
            return {"error": "Insufficient sample size for Chi-square test."}

        contingency_table = pd.crosstab(clean[cat_col1], clean[cat_col2])
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)

        is_sig = bool(p_val < 0.05)
        finding = (
            f"Chi-square test of independence between '{cat_col1}' and '{cat_col2}': "
            f"Chi2={round(float(chi2), 2)}, dof={dof}, p-value={p_val:.4e}. "
            f"{'Significant association exists between these categories (p < 0.05).' if is_sig else 'Variables appear statistically independent (p >= 0.05).'}"
        )

        return {
            "test": "Chi-Square Test of Independence",
            "var1": cat_col1,
            "var2": cat_col2,
            "chi2_statistic": round(float(chi2), 2),
            "dof": int(dof),
            "p_value": float(p_val),
            "is_significant": is_sig,
            "contingency_table": contingency_table.to_dict(),
            "summary_finding": finding
        }
