# Use Node.js 18 Alpine for smaller image size
FROM infyartifactory.jfrog.io/docker-local/node:18.12.1-alpine3.15

# Set working directory
WORKDIR /app

# Install OpenSSL for certificate generation
RUN apk add --no-cache openssl

# Copy package files first for better Docker layer caching
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy application source code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/ssl /app/runs && \
    chown -R node:node /app

# Generate SSL certificates (self-signed for development)
RUN npm run generate-certs

# Switch to non-root user for security
USER node

# Expose both HTTP and HTTPS ports
EXPOSE 3000 3443

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000/health', (res) => { process.exit(res.statusCode === 200 ? 0 : 1) })"

# Start the server
CMD ["npm", "start"]
