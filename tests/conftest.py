"""The tests use the real models and Qdrant:  docker compose exec api python -m pytest
Without Docker (no QDRANT_URL) Qdrant runs inside this process, so the collection is built first."""
from app import ingest
from app.config import COLLECTION
from app.search import qdrant

if not qdrant.collection_exists(COLLECTION):
    ingest.build()
