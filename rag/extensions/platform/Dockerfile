# Build from the checkout root: docker build -f rag/extensions/platform/Dockerfile -t mlsrag .
# Data, models, rag/.cache, results and rag/.env are mounted at run time (compose.yaml), never copied in.

FROM node:24-alpine AS web
RUN corepack enable
WORKDIR /web
COPY web/package.json web/pnpm-lock.yaml web/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY web/ ./
RUN pnpm build

FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
RUN uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cpu
# /app is the checkout root. Editable install: mlsrag finds rag/ as parents[3] of its own files.
WORKDIR /app
COPY rag/ rag/
RUN uv pip install --system --no-cache -e rag/extensions/platform
COPY --from=web /web/dist /app/web
# HOME must be writable when compose runs the container as the host user.
ENV HOME=/tmp
EXPOSE 8000
CMD ["mlsrag", "serve", "--host", "0.0.0.0", "--port", "8000", "--web-dir", "/app/web"]
