import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import re

class DataCleaner:
    """
    Autonomous Data Cleaner.
    Performs non-destructive data cleaning with an audit trail of detected issues and changes.
    """

    def __init__(self, df: pd.DataFrame):
        self.original_df = df
        self.cleaned_df = df.copy(deep=True)
        self.issues_detected: List[Dict[str, Any]] = []
        self.operations_performed: List[Dict[str, Any]] = []

    def clean(
        self,
        handle_missing: bool = True,
        remove_duplicates: bool = True,
        convert_types: bool = True,
        standardize_text: bool = True,
        cap_outliers: bool = False
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Execute automated cleaning pipeline and return (cleaned_df, issues_detected, operations_performed).
        """
        self.cleaned_df = self.original_df.copy(deep=True)
        self.issues_detected = []
        self.operations_performed = []

        # 1. Detect & Remove Duplicates
        if remove_duplicates:
            self._handle_duplicates()

        # 2. Type Inference & Conversion (Currencies, percentages, dates)
        if convert_types:
            self._handle_data_types()

        # 3. Text Standardization (strip whitespace, clean string padding)
        if standardize_text:
            self._standardize_text_columns()

        # 4. Handle Missing Values
        if handle_missing:
            self._handle_missing_values()

        # 5. Outlier Capping (optional winsorization)
        if cap_outliers:
            self._cap_extreme_outliers()

        return self.cleaned_df, self.issues_detected, self.operations_performed

    def _handle_duplicates(self):
        dups = int(self.cleaned_df.duplicated().sum())
        if dups > 0:
            self.issues_detected.append({
                "type": "duplicate_rows",
                "severity": "medium",
                "description": f"Found {dups} exact duplicate rows in the dataset ({round(dups / len(self.cleaned_df) * 100, 2)}%)."
            })
            before_len = len(self.cleaned_df)
            self.cleaned_df = self.cleaned_df.drop_duplicates().reset_index(drop=True)
            self.operations_performed.append({
                "operation": "drop_duplicates",
                "details": f"Removed {dups} duplicate rows (row count reduced from {before_len} to {len(self.cleaned_df)})."
            })

    def _handle_data_types(self):
        for col in self.cleaned_df.columns:
            series = self.cleaned_df[col]
            if series.dtype == 'object':
                non_nulls = series.dropna().astype(str).str.strip()
                if len(non_nulls) == 0:
                    continue

                # Check if values are currency strings (e.g. "$1,250.50", "€400", "£99.99")
                currency_pattern = re.compile(r'^[\$\€\£\¥\₹]?\s*-?\d{1,3}(,\d{3})*(\.\d+)?\s*[\$\€\£\¥\₹]?$')
                sample = non_nulls.head(50)
                if all(bool(currency_pattern.match(val)) for val in sample if val):
                    try:
                        clean_num = non_nulls.str.replace(r'[\$\€\£\¥\₹,\s]', '', regex=True)
                        self.cleaned_df[col] = pd.to_numeric(clean_num, errors='coerce')
                        self.issues_detected.append({
                            "type": "string_currency_type",
                            "severity": "low",
                            "column": col,
                            "description": f"Column '{col}' stored numeric currency values formatted as text strings."
                        })
                        self.operations_performed.append({
                            "operation": "type_conversion",
                            "column": col,
                            "details": f"Converted '{col}' from string formatted currency to float64 numeric values."
                        })
                        continue
                    except Exception:
                        pass

                # Check if values are percentage strings (e.g. "15%", "4.5%")
                pct_pattern = re.compile(r'^-?\d+(\.\d+)?\s*%$')
                if all(bool(pct_pattern.match(val)) for val in sample if val):
                    try:
                        clean_pct = non_nulls.str.replace('%', '', regex=False).str.strip()
                        self.cleaned_df[col] = pd.to_numeric(clean_pct, errors='coerce') / 100.0
                        self.issues_detected.append({
                            "type": "string_percentage_type",
                            "severity": "low",
                            "column": col,
                            "description": f"Column '{col}' stored percentages formatted as text strings."
                        })
                        self.operations_performed.append({
                            "operation": "type_conversion",
                            "column": col,
                            "details": f"Converted '{col}' from percentage strings to float ratios (0.0 - 1.0)."
                        })
                        continue
                    except Exception:
                        pass

                # Check if column is a date/time string
                col_lower = str(col).lower()
                is_date_name = any(k in col_lower for k in ['date', 'time', 'timestamp', 'day', 'month', 'year'])
                if is_date_name:
                    try:
                        parsed_dates = pd.to_datetime(self.cleaned_df[col], errors='coerce')
                        if parsed_dates.notna().sum() > 0.8 * len(non_nulls):
                            self.cleaned_df[col] = parsed_dates
                            self.issues_detected.append({
                                "type": "string_date_type",
                                "severity": "low",
                                "column": col,
                                "description": f"Column '{col}' stored dates as raw strings."
                            })
                            self.operations_performed.append({
                                "operation": "type_conversion",
                                "column": col,
                                "details": f"Parsed '{col}' strings into native datetime64[ns] timestamps."
                            })
                    except Exception:
                        pass

    def _standardize_text_columns(self):
        for col in self.cleaned_df.columns:
            if self.cleaned_df[col].dtype == 'object':
                orig = self.cleaned_df[col]
                stripped = orig.astype(str).str.strip()
                # Check if stripping caused any modifications
                diffs = (orig.notna() & (orig != stripped)).sum()
                if diffs > 0:
                    self.cleaned_df[col] = self.cleaned_df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
                    self.operations_performed.append({
                        "operation": "standardize_text",
                        "column": col,
                        "details": f"Trimmed leading and trailing whitespace from {diffs} text values in '{col}'."
                    })

    def _handle_missing_values(self):
        for col in self.cleaned_df.columns:
            missing_count = int(self.cleaned_df[col].isna().sum())
            if missing_count > 0:
                pct = round(missing_count / len(self.cleaned_df) * 100, 2)
                severity = "high" if pct > 30 else ("medium" if pct > 10 else "low")
                self.issues_detected.append({
                    "type": "missing_values",
                    "severity": severity,
                    "column": col,
                    "description": f"Column '{col}' has {missing_count} missing values ({pct}%)."
                })

                if pd.api.types.is_numeric_dtype(self.cleaned_df[col]):
                    # Impute with median for robust imputation against skew
                    median_val = float(self.cleaned_df[col].median())
                    self.cleaned_df[col] = self.cleaned_df[col].fillna(median_val)
                    self.operations_performed.append({
                        "operation": "impute_numeric_missing",
                        "column": col,
                        "details": f"Imputed {missing_count} missing values in '{col}' with median ({round(median_val, 2)})."
                    })
                elif pd.api.types.is_datetime64_any_dtype(self.cleaned_df[col]):
                    # Forward-fill dates or leave NaT
                    self.cleaned_df[col] = self.cleaned_df[col].bfill().ffill()
                    self.operations_performed.append({
                        "operation": "impute_date_missing",
                        "column": col,
                        "details": f"Filled {missing_count} missing dates in '{col}' via forward/backward interpolation."
                    })
                else:
                    # Categorical: impute with 'Unknown'
                    self.cleaned_df[col] = self.cleaned_df[col].fillna("Unknown")
                    self.operations_performed.append({
                        "operation": "impute_categorical_missing",
                        "column": col,
                        "details": f"Replaced {missing_count} missing categorical entries in '{col}' with 'Unknown'."
                    })

    def _cap_extreme_outliers(self):
        num_cols = self.cleaned_df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            s = self.cleaned_df[col]
            q25 = float(s.quantile(0.25))
            q75 = float(s.quantile(0.75))
            iqr = q75 - q25
            lower_bound = q25 - 3.0 * iqr
            upper_bound = q75 + 3.0 * iqr
            
            extreme_mask = (s < lower_bound) | (s > upper_bound)
            extreme_cnt = int(extreme_mask.sum())
            if extreme_cnt > 0:
                self.issues_detected.append({
                    "type": "extreme_outliers",
                    "severity": "medium",
                    "column": col,
                    "description": f"Found {extreme_cnt} extreme values beyond 3x IQR in '{col}'."
                })
                self.cleaned_df[col] = s.clip(lower=lower_bound, upper=upper_bound)
                self.operations_performed.append({
                    "operation": "cap_outliers",
                    "column": col,
                    "details": f"Winsorized {extreme_cnt} extreme values in '{col}' to bounds [{round(lower_bound, 2)}, {round(upper_bound, 2)}]."
                })
