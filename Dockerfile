# ── Stage 1: Build Next.js frontend ─────────────────────────────────────────
FROM node:20-slim AS frontend-builder
WORKDIR /frontend

COPY frontend/package.json ./
RUN npm install

COPY frontend/ ./

# API routes through nginx in production
ENV NEXT_PUBLIC_API_URL=/api
RUN npm run build

# ── Stage 2: Production runtime ──────────────────────────────────────────────
FROM python:3.11-slim

# Install Node.js 20 + nginx
RUN apt-get update && apt-get install -y curl gnupg nginx && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python API dependencies (no GPU packages)
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy Python backend
COPY agents/    ./agents/
COPY api/       ./api/
COPY auditor/   ./auditor/
COPY causal/    ./causal/
COPY config/    ./config/
COPY core/      ./core/
COPY defense/   ./defense/
COPY emergence/ ./emergence/
COPY env/       ./env/
COPY logs/      ./logs/
COPY memory/    ./memory/
COPY metrics/   ./metrics/
COPY openenv/   ./openenv/
COPY rewards/   ./rewards/
COPY main.py    ./

# Copy built Next.js app
COPY --from=frontend-builder /frontend/.next          ./frontend/.next
COPY --from=frontend-builder /frontend/package.json   ./frontend/package.json
COPY --from=frontend-builder /frontend/node_modules   ./frontend/node_modules

# nginx config
COPY nginx.conf /etc/nginx/sites-available/statecraft
RUN rm -f /etc/nginx/sites-enabled/default && \
    ln -s /etc/nginx/sites-available/statecraft /etc/nginx/sites-enabled/statecraft

# Startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 7860

CMD ["/start.sh"]
