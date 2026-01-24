"""Scene capture service - port 8095."""
from fastapi import FastAPI

app = FastAPI(title="Scene Capture Service", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


# TODO: POST /sessions/create
# TODO: POST /sessions/{id}/capture
# TODO: POST /sessions/{id}/auto
# TODO: GET /sessions/{id}/status
# TODO: POST /sessions/{id}/finalize
