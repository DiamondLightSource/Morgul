FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends epics-base && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY morgul/ morgul/
RUN uv sync --no-dev --no-editable

COPY contrib/morgul-sink.py contrib/morgul-sink.py
RUN uv sync --script contrib/morgul-sink.py

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "contrib/morgul-sink.py"]
