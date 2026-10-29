"""
Rigenera UNA VOLTA gli embedding di tutti i prodotti con il nuovo modello,
scrivendoli nella colonna prodotti.embedding_v2 (quella vecchia di Gemini resta intatta).

Si può rilanciare senza problemi: elabora solo i prodotti con embedding_v2 ancora vuoto.

Variabili d'ambiente:
  SUPABASE_URL          es. https://xxxx.supabase.co
  SUPABASE_SERVICE_KEY  chiave service_role (non la anon: deve poter scrivere)
  CAMPI_TESTO           colonne da unire nel testo del prodotto (default: modello)
                        es. "modello,descrizione"
  CHIAVE_PRODOTTO       colonna identificativa del prodotto (default: id)
  MODEL_NAME / PREFIX   devono essere IDENTICI a quelli del servizio /embed
"""

import os
import sys

import requests
from sentence_transformers import SentenceTransformer

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
MODEL_NAME = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-small")
PREFIX = os.getenv("PREFIX", "query: ")
CAMPI = [c.strip() for c in os.getenv("CAMPI_TESTO", "modello").split(",") if c.strip()]
CHIAVE = os.getenv("CHIAVE_PRODOTTO", "id")
LOTTO = 100

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}


def testo_prodotto(riga):
    return " ".join(str(riga.get(c) or "").strip() for c in CAMPI).strip()


def main():
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    selezione = ",".join([CHIAVE] + CAMPI)
    visti = set()
    totale = 0

    while True:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/prodotti",
            headers=HEADERS,
            params={"select": selezione, "embedding_v2": "is.null", "order": CHIAVE, "limit": LOTTO},
            timeout=60,
        )
        r.raise_for_status()
        righe = r.json()
        if not righe:
            break

        # Protezione: se un prodotto ricompare, l'aggiornamento non sta funzionando.
        for riga in righe:
            if riga[CHIAVE] in visti:
                sys.exit(f"Il prodotto {riga[CHIAVE]} non viene aggiornato: controlla CHIAVE_PRODOTTO e i permessi.")
            visti.add(riga[CHIAVE])

        vettori = model.encode(
            [PREFIX + testo_prodotto(riga) for riga in righe],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        for riga, vettore in zip(righe, vettori):
            u = requests.patch(
                f"{SUPABASE_URL}/rest/v1/prodotti",
                headers=HEADERS,
                params={CHIAVE: f"eq.{riga[CHIAVE]}"},
                json={"embedding_v2": vettore.tolist()},
                timeout=60,
            )
            u.raise_for_status()

        totale += len(righe)
        print(f"Aggiornati {totale} prodotti", flush=True)

    print(f"Fatto: {totale} prodotti con il nuovo embedding.")


if __name__ == "__main__":
    main()
