FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libssl-dev libffi-dev libsm6 libxext6 libxrender-dev \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MODEL_DIR=/app/models

RUN mkdir -p /app/models

COPY snipshot_engine/ ./snipshot_engine/
COPY fonts/ ./fonts/
COPY main.py .
COPY entrypoint.py .

EXPOSE 8001

CMD ["python3", "entrypoint.py"]