"""
AutoAnalyst AI - Root Entry Point for Streamlit Community Cloud and Local Execution.
This file imports and executes the frontend application in `frontend/app.py`.
"""
import os
import sys

# Add repository root and frontend to sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Execute frontend application
from frontend.app import *
