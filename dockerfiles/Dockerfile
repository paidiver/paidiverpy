# syntax=docker/dockerfile:1
FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/config_files /app/output /app/input /app/metadata /app/custom_algorithms
COPY pyproject.toml .
COPY LICENSE README.rst ./
COPY src/ src/

RUN --mount=type=cache,target=/root/.cache/pip \
  python -m pip install . \
  && python -c "import cv2, geopandas, rawpy"

ENV IS_DOCKER=true

ENTRYPOINT ["paidiverpy"]

CMD ["--help"]

EXPOSE 5006
