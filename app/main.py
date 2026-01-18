from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from .routers import classify
import os

app = FastAPI(title="Email Classifier Dashboard", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Router
app.include_router(classify.router, prefix="/api", tags=["Classification"])

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Email Classifier API"}

# Serve React Frontend (Static Files) - MOUNT LAST to avoid shadowing API
# In Docker, we copy the build to /app/static
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
