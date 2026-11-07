FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<3"
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static
COPY models ./models
COPY README.md .
COPY docs ./docs
COPY evaluation ./evaluation

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
