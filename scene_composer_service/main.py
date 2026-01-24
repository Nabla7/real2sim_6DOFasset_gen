"""Scene composer service - port 8098."""
from fastapi import FastAPI

app = FastAPI(title="Scene Composer Service", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


# TODO: POST /compose
# TODO: POST /scenes/{id}/add_object
# TODO: GET /scenes/{id}/export
