FROM python:3.12-slim

WORKDIR /app

# CPU-only PyTorch: the default wheel brings ~2.5 GB of CUDA libraries that a laptop does not use
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedder and the reranker at build time, so the container starts without internet.
# Only these two files are copied first, so a code change does not download the models again.
COPY app/__init__.py app/config.py app/encoders.py ./app/
RUN python -c "import app.encoders"
# The models are inside the image now: never contact Hugging Face at runtime
ENV HF_HUB_OFFLINE=1
# print() lines show up in `docker compose logs` at once, not when a buffer fills
ENV PYTHONUNBUFFERED=1
# For the tests (docker compose exec api python -m pytest); after the models, so it does not download them again
RUN pip install --no-cache-dir pytest==9.1.1

COPY app ./app
COPY data ./data
COPY tests ./tests

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
