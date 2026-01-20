# =============================================================================
# ContextFlow UI - Multi-Stage Dockerfile
# =============================================================================
# Supports both development (hot reload) and production (nginx) builds
#
# Usage:
#   Development: docker build --target development -t contextflow-ui:dev .
#   Production:  docker build --target production -t contextflow-ui:latest .
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base - Common dependencies
# -----------------------------------------------------------------------------
FROM node:20-alpine AS base

WORKDIR /app

# Install dependencies for node-gyp (if needed by some packages)
RUN apk add --no-cache libc6-compat

# Copy package files
COPY package*.json ./

# -----------------------------------------------------------------------------
# Stage 2: Dependencies - Install all dependencies
# -----------------------------------------------------------------------------
FROM base AS deps

# Install all dependencies (including devDependencies)
RUN npm ci

# -----------------------------------------------------------------------------
# Stage 3: Development - Hot reload server
# -----------------------------------------------------------------------------
FROM base AS development

# Copy dependencies from deps stage
COPY --from=deps /app/node_modules ./node_modules

# Copy source files
COPY . .

# Expose Vite dev server port
EXPOSE 5173

# Environment variables for development
ENV NODE_ENV=development
ENV VITE_API_URL=http://localhost:8000
ENV VITE_API_WS_URL=ws://localhost:8000

# Start Vite dev server with host binding for Docker
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# -----------------------------------------------------------------------------
# Stage 4: Builder - Build production assets
# -----------------------------------------------------------------------------
FROM base AS builder

# Copy dependencies from deps stage
COPY --from=deps /app/node_modules ./node_modules

# Copy source files
COPY . .

# Build arguments for API URL (can be overridden at build time)
ARG VITE_API_URL=http://localhost:8000
ARG VITE_API_WS_URL=ws://localhost:8000

# Set environment variables for build
ENV VITE_API_URL=$VITE_API_URL
ENV VITE_API_WS_URL=$VITE_API_WS_URL
ENV NODE_ENV=production

# Build the application
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 5: Production - Nginx server
# -----------------------------------------------------------------------------
FROM nginx:alpine AS production

# Install wget for healthcheck
RUN apk add --no-cache wget

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose HTTP port
EXPOSE 80

# Environment variables (for runtime config injection if needed)
ENV NODE_ENV=production

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:80/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
