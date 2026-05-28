FROM python:3.10

# Create a non-root user for Hugging Face compatibility
RUN useradd -m -u 1000 user
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libssl-dev libffi-dev libsm6 libxext6 libxrender-dev \
    libgl1 libglib2.0-0 wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models during the build process
RUN mkdir -p /app/models/detection /app/models/inpainting /app/models/ocr && \
    wget -q -O /app/models/detection/detect-20241225.ckpt https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/detect-20241225.ckpt && \
    wget -q -O /app/models/inpainting/lama_large_512px.ckpt https://huggingface.co/dreMaz/AnimeMangaInpainting/resolve/main/lama_large_512px.ckpt && \
    wget -q -O /app/models/ocr/ocr_ar_48px.ckpt https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr_ar_48px.ckpt && \
    wget -q -O /app/models/ocr/alphabet-all-v7.txt https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/alphabet-all-v7.txt

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MODEL_DIR=/app/models PORT=7860

COPY snipshot_engine/ ./snipshot_engine/
COPY fonts/ ./fonts/
COPY main.py .
COPY entrypoint.py .

RUN chown -R user:user /app
USER user

EXPOSE 7860

CMD ["python3", "entrypoint.py"]