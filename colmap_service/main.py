"""COLMAP service - port 8096."""
from fastapi import FastAPI

app = FastAPI(title="COLMAP Service", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


# TODO: POST /reconstruct
# TODO: GET /jobs/{id}/status
# TODO: GET /jobs/{id}/result
