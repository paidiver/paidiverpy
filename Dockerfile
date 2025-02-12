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

RUN mkdir /app/config_files
RUN mkdir /app/output
RUN mkdir /app/input
RUN mkdir /app/metadata
RUN mkdir /app/custom_algorithms
COPY pyproject.toml .
COPY LICENSE README.md ./
COPY src/ src/

RUN pip install .

ENV IS_DOCKER=true

ENTRYPOINT ["paidiverpy"]

CMD ["--help"]
