# ============================================================
# RoboMemo Platform — Production Dockerfile
# Multi-stage build: frontend (Vite) + backend (Node.js)
# ============================================================

# --- Stage 1: Build frontend ---
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY app/package.json app/package-lock.json* ./
RUN npm ci --production=false
COPY app/ ./
RUN npm run build

# --- Stage 2: Backend runtime ---
FROM node:20-alpine AS runtime

# Security: run as non-root
RUN addgroup -S robomemo && adduser -S robomemo -G robomemo

# Install ffmpeg (for frame extraction)
RUN apk add --no-cache ffmpeg python3 py3-pip

WORKDIR /app

# Backend dependencies
COPY backend/package.json backend/package-lock.json* ./
RUN npm ci --production

# Backend source
COPY backend/ ./

# Copy built frontend into backend static serve
COPY --from=frontend-build /app/frontend/dist ./public

# Create data & upload directories
RUN mkdir -p data uploads/frames && chown -R robomemo:robomemo /app

# Install Python dependencies for VLM analyzer (optional)
COPY backend/requirements.txt* ./
RUN pip3 install --no-cache-dir -r requirements.txt 2>/dev/null || true

USER robomemo

# Environment
ENV NODE_ENV=production
ENV PORT=3001

EXPOSE 3001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3001/api/health || exit 1

CMD ["node", "server.js"]
