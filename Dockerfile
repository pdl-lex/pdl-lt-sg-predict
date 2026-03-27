FROM python:3.13-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen
COPY . .
RUN uv run reflex init
EXPOSE 3000
CMD ["uv", "run", "reflex", "run", "--env", "prod", "--backend-host", "0.0.0.0"]