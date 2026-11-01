# kokoro-tts-server

Self-hosted text-to-speech server built on [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) (v1.0, ONNX) with FastAPI. It strips markdown and emoji from the input so the voice reads only words.

> **Designed to run on the NVIDIA Jetson Orin Nano Super.** It is developed and tested on that device (aarch64, JetPack 6 / L4T r36.4). The container is based on `nvcr.io/nvidia/l4t-jetpack` and uses `onnxruntime-gpu` from the Jetson AI Lab index to run on the Orin's GPU through CUDA. It falls back to CPU if CUDA is unavailable.

## Endpoints (port 8002)

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check, returns `{"status":"ok"}` |
| POST | `/generate` | JSON in, `audio/wav` out (24 kHz, mono, 16-bit) |

`/generate` request body:

```json
{ "text": "Hello there.", "voice": "bm_daniel", "speed": 1.0, "lang": "en-us" }
```

Only `text` is required. Code fences, links, URLs, headings, tables and emoji are removed before synthesis. If nothing speakable remains, the server returns `400`.

## Models

Model files are not stored in this repo. Download them from the [kokoro-onnx releases](https://github.com/thewh1teagle/kokoro-onnx/releases) into `models/`:

```
models/
├── kokoro-v1.0.onnx
└── voices-v1.0.bin
```

## Run

Requires Docker with the NVIDIA runtime on the Jetson.

```bash
docker compose up -d --build
curl localhost:8002/health
```

The `models/` directory is bind-mounted into the container at `/app/models`.

## Example

```bash
curl -X POST localhost:8002/generate \
  -H 'content-type: application/json' \
  -d '{"text":"The quick brown fox jumps over the lazy dog."}' \
  -o out.wav
```

## Notes

- `kokoro-onnx` is installed with `--no-deps` in the Dockerfile. It depends on CPU-only `onnxruntime`, which would overwrite `onnxruntime-gpu`. See `requirements.txt` and `overrides.txt`.
- On startup the log prints `Active providers:`. Check that it lists `CUDAExecutionProvider`.
- Pairs with [canary-stt-server](https://github.com/apurvaumredkar/canary-stt-server) for a fully local voice stack.
