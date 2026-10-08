import sys
import io
import ast
import traceback
import pandas as pd
import numpy as np
import scipy
from typing import Dict, Any, Optional

# Restricted built-ins for analytical safety
SAFE_BUILTINS = {
    'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool, 'dict': dict,
    'enumerate': enumerate, 'filter': filter, 'float': float, 'format': format,
    'int': int, 'isinstance': isinstance, 'len': len, 'list': list, 'map': map,
    'max': max, 'min': min, 'pow': pow, 'print': print, 'range': range,
    'reversed': reversed, 'round': round, 'set': set, 'slice': slice,
    'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple, 'zip': zip,
    'True': True, 'False': False, 'None': None
}

FORBIDDEN_MODULES = {
    'os', 'sys', 'subprocess', 'shutil', 'socket', 'requests', 'urllib',
    'http', 'pathlib', 'builtins', '__import__', 'importlib', 'pty', 'posix'
}

class PythonAnalysisTool:
    """
    Sandboxed Python execution engine for Pandas/NumPy/SciPy analytical workflows.
    Ensures safe execution and returns structured output and created variables.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy(deep=True)

    def validate_code_safety(self, code: str) -> Optional[str]:
        """Validate code using AST to check for forbidden imports or dangerous operations."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"Syntax Error in code: {e}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_name = alias.name.split('.')[0]
                    if root_name in FORBIDDEN_MODULES:
                        return f"Security Restriction: Module '{root_name}' is forbidden."
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_name = node.module.split('.')[0]
                    if root_name in FORBIDDEN_MODULES:
                        return f"Security Restriction: Importing from '{root_name}' is forbidden."
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in ('eval', 'exec', '__import__', 'open'):
                    return f"Security Restriction: Calling '{node.func.id}' is forbidden."
        return None

    def execute(self, code: str) -> Dict[str, Any]:
        """
        Execute analytical python code in a safe environment.
        Available objects: `df`, `pd`, `np`, `scipy`.
        """
        validation_error = self.validate_code_safety(code)
        if validation_error:
            return {
                "success": False,
                "error": validation_error,
                "output": "",
                "result": None,
                "result_type": None
            }

        stdout_capture = io.StringIO()
        old_stdout = sys.stdout

        local_scope = {
            'df': self.df.copy(deep=True),
            'pd': pd,
            'np': np,
            'scipy': scipy,
            'result': None
        }
        global_scope = {
            '__builtins__': SAFE_BUILTINS,
            'pd': pd,
            'np': np,
            'scipy': scipy
        }

        try:
            sys.stdout = stdout_capture
            exec(code, global_scope, local_scope)
            sys.stdout = old_stdout

            # Identify candidate result
            res = local_scope.get('result', None)
            res_type = type(res).__name__ if res is not None else None

            # If result is DataFrame/Series, convert to dict/records for easy inspection
            serialized_result = None
            if isinstance(res, pd.DataFrame):
                serialized_result = {
                    "shape": list(res.shape),
                    "columns": list(res.columns),
                    "data": res.head(50).to_dict(orient="records")
                }
            elif isinstance(res, pd.Series):
                serialized_result = {
                    "length": len(res),
                    "name": str(res.name),
                    "data": res.head(50).to_dict()
                }
            elif isinstance(res, (int, float, str, bool, list, dict)):
                serialized_result = res
            elif res is not None:
                serialized_result = str(res)

            return {
                "success": True,
                "output": stdout_capture.getvalue().strip(),
                "result": serialized_result,
                "raw_result": res,
                "result_type": res_type,
                "error": None
            }

        except Exception as e:
            sys.stdout = old_stdout
            err_msg = traceback.format_exc()
            return {
                "success": False,
                "output": stdout_capture.getvalue().strip(),
                "result": None,
                "raw_result": None,
                "result_type": None,
                "error": f"{type(e).__name__}: {str(e)}"
            }
