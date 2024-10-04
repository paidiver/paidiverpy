FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libopencv-dev \
    libgdal-dev \
    libgraphviz-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

COPY pyproject.toml .
COPY configuration-schema.json .
COPY LICENSE README.md ./
COPY src/ src/

RUN pip install .

ENTRYPOINT ["paidiverpy"]

CMD ["--help"]
