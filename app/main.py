import os

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    return {"message": "hello from fastapi"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "app": "ai-infra-lab",
        "message": os.getenv("APP_MESSAGE", "default-message"),
        "hostname": os.getenv("HOSTNAME", "unknown"),
    }
