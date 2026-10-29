"""
Servizio di embedding per il matching prodotti (da pubblicare su Railway).

POST /embed   { "texts": ["pinza anatomica 16 cm", "forbici mayo 15 cm", ...] }
  → { "embeddings": [[...], [...]], "dim": 384 }

GET  /health  → stato del servizio, modello caricato e dimensione dei vettori

Variabili d'ambiente:
  API_TOKEN   token segreto, da inviare come "Authorization: Bearer <token>"  (obbligatorio)
  MODEL_NAME  modello da usare (default: intfloat/multilingual-e5-small, 384 dimensioni)
  PREFIX      prefisso richiesto dai modelli E5 (default: "query: ")
"""

import os

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-small")
API_TOKEN = os.getenv("API_TOKEN", "")
PREFIX = os.getenv("PREFIX", "query: ")
MAX_TESTI = 256

app = FastAPI(title="Embedding prodotti")

# Il modello viene caricato una volta sola all'avvio e resta in memoria.
model = SentenceTransformer(MODEL_NAME, device="cpu")
DIM = model.get_sentence_embedding_dimension()


class Richiesta(BaseModel):
    texts: list[str]


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME, "dim": DIM}


@app.post("/embed")
def embed(req: Richiesta, authorization: str = Header(default="")):
    if not API_TOKEN or authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="token non valido")
    if not req.texts:
        return {"embeddings": [], "dim": DIM}
    if len(req.texts) > MAX_TESTI:
        raise HTTPException(status_code=400, detail=f"massimo {MAX_TESTI} testi per richiesta")

    # Stesso prefisso per voci del bando e prodotti: il confronto è "alla pari".
    testi = [PREFIX + (t or "").strip() for t in req.texts]
    vettori = model.encode(
        testi,
        batch_size=32,
        normalize_embeddings=True,  # vettori normalizzati: la somiglianza coseno è corretta
        show_progress_bar=False,
    )
    return {"embeddings": [v.tolist() for v in vettori], "dim": DIM}
