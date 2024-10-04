FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

RUN apt-get update
RUN apt-get install -y apt-utils build-essential
RUN apt-get install -y libexempi-dev libxslt1-dev
RUN apt-get install -y libgraphviz-dev
RUN apt-get install -y libopencv-dev libgdal-dev
RUN apt-get install -y curl unzip
RUN rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

COPY pyproject.toml .
COPY configuration-schema.json .
COPY LICENSE README.md ./
COPY src/ src/

RUN pip install .

ENTRYPOINT ["paidiverpy"]

CMD ["--help"]
