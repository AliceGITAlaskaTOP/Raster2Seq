# Raster2Seq recognition service on CPU. Build from the repository root:
#   docker build -f serve/Dockerfile -t raster2seq-serve .
# The CubiCasa5K checkpoint (~1.5 GB) is not baked in: it is downloaded on
# first start into HF_HOME. Mount a named volume there (/models) so restarts
# and rebuilds reuse it.
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    PORT=8000
RUN pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
COPY serve/requirements.txt serve/requirements.txt
RUN pip install -r serve/requirements.txt
COPY . .
VOLUME /models
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD python -c "import json,urllib.request,sys; sys.exit(0 if json.load(urllib.request.urlopen('http://127.0.0.1:8000/health'))['ready'] else 1)"
CMD ["python", "-m", "serve.app"]
