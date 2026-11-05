FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libopus0 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

# The model is too big for git (see MODEL.md). Either put pl_piper11.onnx / pl_multi.onnx in
# src/pltts/models/ before building, or pass --build-arg MODEL_URL=<direct link to the .onnx>.
ARG MODEL_URL=""
RUN if [ -n "$MODEL_URL" ]; then curl -fL -o "src/pltts/models/$(basename "${MODEL_URL%%\?*}")" "$MODEL_URL"; fi \
    && ls src/pltts/models/*.onnx >/dev/null \
    && pip install --no-cache-dir ".[server]"

ENV PLTTS_LANES=2 \
    PYTHONUNBUFFERED=1
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health').status==200 else 1)"
ENTRYPOINT ["pltts-server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
