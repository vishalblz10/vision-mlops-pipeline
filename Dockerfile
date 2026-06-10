# Build a slim serving image: only the runtime deps the API needs, CPU wheels.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only torch keeps the image far smaller than the default CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install numpy pillow fastapi "uvicorn[standard]" prometheus-client python-multipart

COPY pyproject.toml README.md ./
COPY vision_mlops ./vision_mlops
RUN pip install --no-deps .

# The model is mounted (or baked in) at this path; /readyz reports 503 until it exists.
ENV MODEL_PATH=/models/model.pt
VOLUME /models

EXPOSE 8000

RUN useradd --create-home appuser
USER appuser

CMD ["uvicorn", "vision_mlops.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
