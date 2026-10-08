import sqlite3
import pandas as pd
import time
import re
from typing import Dict, Any, Optional

FORBIDDEN_SQL_KEYWORDS = [
    r'\bDROP\b', r'\bDELETE\b', r'\bINSERT\b', r'\bUPDATE\b',
    r'\bALTER\b', r'\bCREATE\b', r'\bTRUNCATE\b', r'\bREPLACE\b',
    r'\bATTACH\b', r'\bDETACH\b', r'\bPRAGMA\b', r'\bEXEC\b'
]

class SQLAnalysisTool:
    """
    In-memory SQLite analytical query runner.
    Ensures safe, SELECT-only read access to dataset tables.
    """

    def __init__(self, df: pd.DataFrame, table_name: str = "dataset"):
        self.df = df
        self.table_name = table_name
        self.conn = sqlite3.connect(":memory:")
        self._load_table()

    def _load_table(self):
        """Clean column names for SQL compatibility and write DataFrame to in-memory SQLite."""
        # Sanitize column names: replace spaces, punctuation with underscores
        sql_df = self.df.copy()
        clean_cols = [re.sub(r'[^\w]', '_', str(col)).strip('_') for col in sql_df.columns]
        # Resolve duplicates
        seen = {}
        unique_cols = []
        for c in clean_cols:
            if c in seen:
                seen[c] += 1
                unique_cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                unique_cols.append(c)
        sql_df.columns = unique_cols
        self.column_mapping = dict(zip(self.df.columns, unique_cols))
        self.reverse_column_mapping = dict(zip(unique_cols, self.df.columns))
        
        # Write to SQLite
        sql_df.to_sql(self.table_name, self.conn, if_exists="replace", index=False)

    def validate_query(self, query: str) -> Optional[str]:
        """Verify that SQL query is SELECT-only and does not contain mutating keywords."""
        q_upper = query.strip().upper()
        if not q_upper.startswith("SELECT") and not q_upper.startswith("WITH"):
            return "Security Restriction: Only SELECT or WITH (Common Table Expression) queries are permitted."

        for pattern in FORBIDDEN_SQL_KEYWORDS:
            if re.search(pattern, q_upper):
                return f"Security Restriction: Mutating SQL operations ({pattern.strip(r'\\b')}) are strictly forbidden."

        return None

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute validated analytical SQL query and return DataFrame result."""
        validation_error = self.validate_query(query)
        if validation_error:
            return {
                "success": False,
                "error": validation_error,
                "data": None,
                "row_count": 0,
                "execution_time_ms": 0.0
            }

        start_time = time.time()
        try:
            result_df = pd.read_sql_query(query, self.conn)
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "success": True,
                "error": None,
                "data": result_df,
                "records": result_df.head(100).to_dict(orient="records"),
                "columns": list(result_df.columns),
                "row_count": int(len(result_df)),
                "execution_time_ms": elapsed_ms
            }
        except Exception as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "success": False,
                "error": f"SQL Execution Error: {str(e)}",
                "data": None,
                "records": [],
                "columns": [],
                "row_count": 0,
                "execution_time_ms": elapsed_ms
            }

    def get_table_schema(self) -> str:
        """Return table schema description for agent SQL generation."""
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({self.table_name})")
        rows = cursor.fetchall()
        schema_lines = [f"Table: {self.table_name}"]
        for r in rows:
            col_name = r[1]
            col_type = r[2]
            orig_col = self.reverse_column_mapping.get(col_name, col_name)
            schema_lines.append(f" - {col_name} ({col_type}) [original: '{orig_col}']")
        return "\n".join(schema_lines)
