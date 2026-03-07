# utils/__init__.py
# DeepShield utility package

import sys
import os

# Ensure the project root (parent of utils/) is always on sys.path.
# This means `from utils.x import y` works from any working directory.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
