FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install --no-cache-dir --no-deps kokoro-onnx==0.6.1

COPY main.py ./

EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
