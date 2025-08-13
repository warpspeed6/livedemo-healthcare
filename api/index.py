import os
from main import app as fastapi_app


# Vercel expects a module-level variable named 'app'
app = fastapi_app


