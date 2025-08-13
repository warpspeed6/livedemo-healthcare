import os
import sys

# Ensure project root is on path for importing main.py
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import app as fastapi_app


# Vercel expects a module-level variable named 'app'
app = fastapi_app


