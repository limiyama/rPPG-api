from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.ws import router as ws_router

app = FastAPI(title="rPPG API")

# WebSocket não passa por CORS; isto serve ao /health e a um eventual
# endpoint HTTP futuro. Mantido permissivo como no backend atual.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_router)


@app.get("/health")
def health():
    return {"status": "ok"}
