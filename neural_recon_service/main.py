"""Neural reconstruction service - port 8097."""
from fastapi import FastAPI

app = FastAPI(title="Neural Reconstruction Service", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


# TODO: POST /train
# TODO: GET /jobs/{id}/status
# TODO: POST /jobs/{id}/export
