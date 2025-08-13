import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.requests import Request
from main import app as fastapi_app


# Vercel expects a module-level variable named 'app'
app = fastapi_app


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


