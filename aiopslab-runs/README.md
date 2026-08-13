# AIOpsLab Viewer - Secure Server

A secure HTTP/HTTPS server for serving the AIOpsLab session analysis viewer with proper security headers, CORS configuration, and static file serving.

## Quick Start

### 🐳 **Docker (Recommended)**

```bash
### 🐳 **Docker (Recommended)**

```bash
# Clone and start with Docker
git clone <repo>
cd aiopslab-runs
docker-compose up -d

# Access at http://localhost:3000 or https://localhost:3443
```

### ☸️ **Kubernetes (Production)**

```bash
# Deploy to Kubernetes with Helm
git clone <repo>
cd aiopslab-runs
./scripts/k8s-deploy.sh

# Or manually with Helm
helm install aiopslab-viewer ./helm/aiopslab-viewer \
  --namespace aiopslab \
  --create-namespace \
  --values ./helm/aiopslab-viewer/values-dev.yaml
```

### 🔧 **Manual Setup**

```bash
# Clone and start manually
git clone <repo>
cd aiopslab-runs
npm install
cp .env.example .env  # Modify as needed
npm start
```


## Features

- 🔒 **Secure HTTPS** support with self-signed certificates for development
- 🛡️ **Security Headers** using Helmet.js (CSP, XSS protection, etc.)
- 🌐 **CORS** configured for local development
- 📊 **Rate Limiting** to prevent abuse
- 📁 **Static File Serving** with proper MIME types
- 🔍 **API Endpoints** for dynamic content
- 📝 **Request Logging** for debugging
- ⚡ **Health Check** endpoint
- 💾 **Embedded Database** with SQLite for persistent caching
- 🔄 **Smart Caching** - only reanalyzes when files change
- 📈 **Analytics** - database statistics and performance metrics

## Architecture Overview

The AIOpsLab Viewer uses a **database-first approach** for optimal scalability and performance:

### 🗄️ **Database-Driven Design**
- **SQLite Database**: Stores all run metadata, metrics, and analysis results
- **No Real-time Scanning**: API endpoints query the database directly, not the filesystem
- **Manual Import**: New runs are imported via explicit scan operations (`POST /api/runs/scan`)
- **Persistent Storage**: Run data persists even if files are moved or deleted

### 📁 **Filesystem Usage**
- **File Serving Only**: Filesystem is used only for serving actual log and evaluation files
- **Static Assets**: Log files, markdown evaluations served via `/runs/<runId>/<file>`
- **No Directory Traversal**: No real-time directory scanning during API requests

### ⚡ **Performance Benefits**
- **Fast API Responses**: Database queries are much faster than filesystem scans
- **Scalable**: Handles hundreds of runs without performance degradation
- **Reduced I/O**: Eliminates filesystem operations during normal browsing
- **Caching**: Database acts as an intelligent cache layer

### 🔄 **Data Flow**
1. **Import**: Manual scan (`🔍 Scan Files` button) analyzes filesystem and populates database
2. **Browse**: UI loads run list from database (`GET /api/runs`)
3. **View**: Individual files served directly from filesystem (`GET /runs/<runId>/<file>`)
4. **Reanalyze**: Force re-analysis of specific run (`POST /api/runs/<runId>/reanalyze`)

## Quick Start

### 1. Install Dependencies

```bash
npm install
```

For HTTPS support, generate self-signed certificates:

```bash
```

### 3. Start the Server

```bash
npm start
```

The server will start on:
- **HTTP**: http://localhost:3000
- **HTTPS**: https://localhost:3443 (if certificates are available)

## Configuration

### Environment Variables

Create a `.env` file (copy from `.env.example`) to configure the server:

```bash
# Server Configuration
PORT=3000                    # HTTP port
HTTPS_PORT=3443             # HTTPS port (optional)
HOST=0.0.0.0                # Host to bind to

# Runs Configuration
RUNS_PATH=./runs            # Directory containing AIOpsLab run folders


# Security Configuration
RATE_LIMIT_WINDOW_MS=900000      # Rate limit window (15 minutes)
RATE_LIMIT_MAX_REQUESTS=100      # Max requests per window

# Development/Production Mode
NODE_ENV=development
```

### Runs Directory Structure

The server expects AIOpsLab runs to be organized in the configured runs directory:

```
runs/
├── 20250715-57fff059/
│   ├── log.txt
│   ├── copilot.md
│   ├── perplexity.md
│   └── ...
├── 20250714-abcdef123/
│   ├── log.txt
│   └── ...
└── ...
```

**Important**: Run folders must follow the naming pattern `YYYYMMDD-<hash>` to be detected by the system.

### Custom Configuration

```bash
# Custom ports
HTTP_PORT=8080 HTTPS_PORT=8443 npm start

# Bind to specific interface
HOST=127.0.0.1 npm start

# Development with auto-restart
npm run dev
```

## API Endpoints

### 📊 **Database-First Endpoints**

#### GET /api/runs
**Fast database query** - Returns JSON list of runs from database (no filesystem scanning):
```json
[
  {
    "id": "20250715-57fff059",
    "created": "2025-07-15T10:30:00.000Z",
    "modified": "2025-07-15T10:35:00.000Z",
    "hasLogFile": true,
    "evaluationFiles": ["copilot.md", "perplexity.md"],
    "evaluationCount": 2,
    "status": "partial",
    "duration": 368.99,
    "issues": 1,
    "reasoning_judgement": "6/10",
    "detectionAccuracy": "Invalid Format",
    "steps": 20,
    "inputTokens": 75033,
    "outputTokens": 464,
    "reasoningScore": 6,
    "namespace": "test-social-network"
  }
]
```

#### POST /api/runs
**Create new run record** - Creates a new run with auto-generated ID:
```bash
curl -X POST http://localhost:3000/api/runs \
  -H "Content-Type: application/json" \
  -d '{"namespace":"my-test-env"}'
```
Response:
```json
{
  "success": true,
  "runId": "20250716-auo58obr",
  "message": "Run 20250716-auo58obr created successfully",
  "run": {
    "id": "20250716-auo58obr",
    "created_at": "2025-07-16T03:25:31.525Z",
    "status": "created",
    "namespace": "my-test-env",
    "hasLogFile": false,
    "evaluationFiles": [],
    "evaluationCount": 0
  }
}
```

#### POST /api/runs/scan
**Manual filesystem import** - Scans filesystem and imports/updates runs in database:
```bash
curl -X POST http://localhost:3000/api/runs/scan
```
Response:
```json
{
  "success": true,
  "message": "Scanned and imported 3 runs",
  "runs": [
    {"id": "20250715-57fff059", "status": "imported"},
    {"id": "20250714-abcdef12", "status": "imported"}
  ]
}
```

### 📤 **File Upload Endpoints**

#### POST /api/runs/:runId/log
**Upload log file** - Upload log.txt for a specific run:
```bash
curl -X POST http://localhost:3000/api/runs/20250716-auo58obr/log \
  -F "logFile=@session-data.txt"
```
**Note**: File will be automatically saved as `log.txt` regardless of original filename.

Response:
```json
{
  "success": true,
  "message": "Log file uploaded successfully for run 20250716-auo58obr",
  "filename": "log.txt",
  "originalName": "session-data.txt",
  "analysis": {
    "status": "success",
    "duration": 15.5,
    "detectionAccuracy": "High",
    "steps": 12,
    "inputTokens": 150,
    "outputTokens": 300,
    "reasoningScore": 85
  }
}
```

#### POST /api/runs/:runId/evaluation
**Upload evaluation file** - Upload evaluation .md file for a specific run:
```bash
# Upload with evaluator name in filename
curl -X POST http://localhost:3000/api/runs/20250716-auo58obr/evaluation \
  -F "evaluationFile=@github-copilot-analysis.md"

# Upload with specific evaluator name (auto-detected from filename)
curl -X POST http://localhost:3000/api/runs/20250716-auo58obr/evaluation \
  -F "evaluationFile=@gpt-4.md"
```

**Supported Evaluators**: `copilot`, `perplexity`, `claude-sonnet`, `gpt-4`, `gemini`, `claude`, `openai`

**File Naming**: Files are automatically renamed based on detected evaluator:
- `github-copilot-eval.md` → `copilot.md`
- `perplexity-analysis.md` → `perplexity.md`
- `gpt-4.md` → `gpt-4.md`

Response:
```json
{
  "success": true,
  "message": "Evaluation file uploaded successfully for run 20250716-auo58obr",
  "filename": "copilot.md",
  "originalName": "github-copilot-analysis.md",
  "evaluator": "copilot",
  "evaluationFiles": ["copilot.md"],
  "evaluationCount": 1,
  "analysis": {
    "averageReasoningJudgement": "8/10",
    "issueCount": 2
  }
}
```

#### GET /api/stats
**Database statistics** - Returns aggregated run statistics:
```json
{
  "totalRuns": 15,
  "successRuns": 8,
  "partialRuns": 5,
  "failedRuns": 2,
  "avgDuration": 284,
  "avgReasoningScore": 7
}
```

#### DELETE /api/runs/:runId
**Remove from database** - Deletes run from database (filesystem files remain):
```bash
curl -X DELETE http://localhost:3000/api/runs/20250715-57fff059
```

### 🔄 **Analysis Endpoints**

#### POST /api/runs/:runId/reanalyze
**Force re-analysis** - Re-analyzes specific run and updates database:
```bash
curl -X POST http://localhost:3000/api/runs/20250715-57fff059/reanalyze
```

### 📁 **File Serving Endpoints**

#### GET /runs/:runId/:filename
**Direct file access** - Serves individual run files:
- `/runs/20250715-57fff059/log.txt` - Raw log file
- `/runs/20250715-57fff059/copilot.md` - Evaluation file

### 🌐 **Web Interface Endpoints**

#### GET /
Main dashboard with database-driven run list and scan functionality

#### GET /viewer.html
Session viewer with URL parameters:
- `?run=20250715-57fff059` - View specific run

#### GET /admin
Admin panel for database management and system administration

## Intelligent Status Assessment

The server automatically analyzes each run to determine its status based on multiple factors:

### Status Categories

1. **🟢 Success** - Run completed successfully with good results
   - Reasoning score ≥ 7/10
   - Root cause identified or solution provided
   - Minimal API/connectivity errors (< 10)
   - Duration reasonable (< 10 minutes)

2. **🟡 Partial Success** - Run completed but with limitations
   - Reasoning score 5-6/10
   - Some progress made but incomplete resolution
   - Moderate API/connectivity issues (< 20 errors)
   - Session duration reasonable

3. **🔴 Failed** - Run failed to complete or provide useful results
   - Fatal errors or termination
   - Excessive API/connectivity errors (> 50)
   - Very low reasoning score (< 5)
   - No meaningful progress

### Analysis Metrics

The system extracts and analyzes:

- **Duration**: Time to detection (TTD) from logs
- **Error Count**: API failures, connection issues, and other errors
- **Steps Taken**: Number of diagnostic actions performed
- **Token Usage**: Input/output tokens for AI interactions
- **Reasoning Score**: Average score from evaluation files
- **Detection Accuracy**: Whether anomalies were correctly identified
- **Issues Found**: Count of problems identified in evaluations
- **Namespace**: Kubernetes namespace being analyzed

### Data Sources

Status assessment uses:
1. **Log Analysis**: Extracts metrics, errors, and session flow
2. **Evaluation Files**: Averages reasoning judgements from AI service evaluations
3. **File Structure**: Checks for completeness and file availability
4. **Timing Analysis**: Calculates session duration and efficiency

### Reasoning Logic

The status assessment follows this decision tree:

#### 1. **Critical Failure Check**
```
IF log file missing OR fatal errors OR >50 errors
  → Status = "failed"
```

#### 2. **API/Infrastructure Issues**
```
Count API errors (Azure OpenAI 404s, connection refused, etc.)
- Heavy API issues: >20 errors
- Moderate API issues: 10-20 errors  
- Light API issues: <10 errors
```

#### 3. **Success Indicators**
```
Check for resolution evidence:
- Log contains "root cause" OR "solution" OR "resolved"
- Reasoning score ≥ 8/10
- Detection accuracy is valid format
```

#### 4. **Status Decision Matrix**
```
IF (has_resolution AND api_errors < 10 AND reasoning_score ≥ 7)
  → Status = "success"
  
ELSE IF (reasoning_score ≥ 5 OR (duration > 0 AND duration < 600 AND api_errors < 20))
  → Status = "partial"
  
ELSE
  → Status = "failed"
```

#### 5. **Metric Extraction Process**

**From Log Files:**
- **Duration**: Extract TTD (Time to Detection) from results JSON
- **Steps**: Count diagnostic actions (exec_shell, get_logs, etc.)
- **Errors**: Find ERROR:, Connection refused, Failed to patterns
- **Namespace**: Extract from kubectl commands or environment setup
- **Metrics**: Parse Results JSON block for detection accuracy, tokens, etc.

**From Evaluation Files:**
- **Reasoning Score**: Extract from "Reasoning Score: X" or "Overall Score: X"
- **Issues**: Count mentions of "issue", "problem", "error", "weakness"
- **Average Reasoning Judgement**: Calculate mean across all evaluation files

#### 6. **Example Assessment**

For a typical run:
```
Log Analysis:
✓ Duration: 368.99 seconds (from TTD)
✓ Steps: 20 (counted exec_shell, get_logs calls)
⚠ API Errors: 15 (Azure OpenAI 404 errors)
✓ Namespace: test-social-network
⚠ Detection Accuracy: "Invalid Format"

Evaluation Analysis:
✓ Average Reasoning Judgement: 6/10 (from copilot.md, perplexity.md)
⚠ Issues Found: 10 (weakness mentions in evaluations)

Decision Process:
1. Not critical failure ✓
2. Moderate API issues (15 errors)
3. No clear resolution evidence
4. Reasoning score = 6 (≥ 5) ✓
5. Duration < 600 seconds ✓
→ Result: "partial" status
```

This logic ensures consistent, objective assessment based on actual session data rather than manual classification.

### GET /health
Health check endpoint:
```json
{
  "status": "healthy",
  "timestamp": "2025-07-15T10:30:00.000Z",
  "uptime": 123.45
}
```

### GET /api/stats
Database and system statistics:
```json
{
  "database": {
    "totalRuns": 25,
    "successRuns": 8,
    "partialRuns": 12,
    "failedRuns": 5,
    "avgDuration": 287,
    "avgReasoningScore": 6
  },
  "system": {
    "uptime": 3600,
    "memory": {...},
    "timestamp": "2025-07-15T10:30:00.000Z"
  }
}
```

### POST /api/runs/:runId/reanalyze
Force reanalysis of a specific run (ignores cache):
```bash
curl -X POST http://localhost:3000/api/runs/20250715-57fff059/reanalyze
```

### POST /api/cleanup
Clean up old runs from database:
```bash
# Via JSON body
curl -X POST http://localhost:3000/api/cleanup \
  -H "Content-Type: application/json" \
  -d '{"days": 30}'

# Via query parameter  
curl -X POST "http://localhost:3000/api/cleanup?days=30"
```

## Database & Caching

### Embedded SQLite Database

The server uses SQLite for persistent storage of analysis results:

- **File**: `runs.db` (created automatically)
- **Purpose**: Cache computed metrics to avoid reanalysis on every request
- **Schema**: Stores all run metadata, status, reasoning judgements, and analysis results

### Smart Caching Logic

1. **File Change Detection**: Uses SHA-256 hash of file modification times and sizes
2. **Conditional Analysis**: Only reanalyzes when files change or run not in database
3. **Performance**: Reduces API response time from ~500ms to ~50ms for cached runs
4. **Scalability**: Handles thousands of runs efficiently

### Cache Invalidation

Cache is automatically invalidated when:
- Log file is modified
- Any evaluation file (*.md) is modified
- New evaluation files are added
- Files are deleted from run directory

### Database Schema

```sql
CREATE TABLE runs (
  id TEXT PRIMARY KEY,                 -- Run ID (e.g., 20250715-57fff059)
  created_at TEXT NOT NULL,            -- Run creation timestamp
  modified_at TEXT NOT NULL,           -- Last file modification
  file_hash TEXT NOT NULL,             -- Hash for change detection
  status TEXT NOT NULL,                -- success|partial|failed
  duration REAL DEFAULT 0,             -- Time to detection (seconds)
  issues INTEGER DEFAULT 0,            -- Number of issues found
  reasoning_judgement TEXT DEFAULT 'N/A',    -- Overall reasoning judgement (e.g., "6/10")
  detection_accuracy TEXT,             -- Detection accuracy result
  steps INTEGER DEFAULT 0,             -- Diagnostic steps taken
  input_tokens INTEGER DEFAULT 0,      -- AI input tokens
  output_tokens INTEGER DEFAULT 0,     -- AI output tokens
  reasoning_score INTEGER DEFAULT 0,   -- Reasoning quality score
  namespace TEXT DEFAULT 'unknown',   -- Kubernetes namespace
  has_log_file BOOLEAN DEFAULT 0,     -- Log file availability
  evaluation_files TEXT DEFAULT '[]', -- JSON array of eval files
  evaluation_count INTEGER DEFAULT 0, -- Number of evaluations
  errors TEXT DEFAULT '[]',           -- JSON array of errors
  last_analyzed_at TEXT NOT NULL     -- Last analysis timestamp
);
```

## Admin Panel

### Overview

The admin panel provides a web-based interface for database management and system administration. Access it at `/admin` once the server is running.

### Features

#### 📊 **Database Statistics**
- Real-time metrics display (total runs, status breakdown)
- Average duration and reasoning judgements
- System uptime and memory usage
- Refresh functionality for live updates

#### 🧹 **Database Cleanup**
- Remove runs older than specified days
- Configurable retention period (1-365 days)
- Confirmation dialogs for destructive operations
- Real-time feedback on cleanup results

#### 📋 **Run Management**
- Interactive list of all runs with status badges
- Individual run reanalysis (bypass cache)
- Bulk operations for selected runs
- Status-based filtering (success/partial/failed)
- Checkbox selection for batch operations

#### 🔄 **Bulk Operations**
- **Reanalyze All**: Force reanalysis of entire database
- **Reanalyze Selected**: Process only checked runs
- **Export Database**: Download runs as JSON file
- **Smart Progress**: Real-time feedback during bulk operations

### Admin Panel Interface

```
🛠️ AIOpsLab Admin Panel
├── Database Statistics (live metrics)
├── Database Cleanup (retention management)
└── Run Management
    ├── Status filtering
    ├── Individual run actions
    ├── Bulk selection
    └── Export functionality
```

### Security Considerations

The admin panel currently has **no authentication** and should only be used in:
- Local development environments
- Trusted internal networks
- Behind proper authentication proxy in production

For production deployments, consider:
- Adding basic auth or OAuth integration
- Restricting admin routes by IP/network
- Using environment-based feature flags

**Note**: The server includes a Content Security Policy (CSP) that allows inline scripts and event handlers for the admin panel functionality. This is configured via the `scriptSrcAttr: ["'unsafe-inline'"]` directive.

### Usage Examples

#### Clean Up Old Runs
1. Navigate to `/admin`
2. Set retention period (e.g., 30 days)
3. Click "Cleanup Database"
4. Confirm the operation

#### Bulk Reanalysis
1. Go to Run Management section
2. Filter by status if needed
3. Select runs using checkboxes
4. Click "Reanalyze Selected"
5. Monitor progress via alerts

#### Export Data
1. Click "Export Database" button
2. JSON file downloads automatically
3. Contains all run data and metrics

## Security Features

### Content Security Policy (CSP)
- Restricts resource loading to same-origin
- Allows inline styles and scripts (required for the viewer)
- Blocks object and frame embedding

### Security Headers
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`

### Rate Limiting
- 100 requests per 15 minutes per IP
- Configurable via express-rate-limit

### CORS Configuration
- Allows localhost and local network access
- Supports credentials for authenticated requests
- Configurable origin validation

## File Structure

```
aiopslab-runs/
├── server.js              # Main server file
├── database.js            # SQLite database module
├── runs.db                # SQLite database (auto-created)
├── package.json           # Node.js dependencies
├── index.html             # Main dashboard
├── viewer.html            # Session viewer
├── scripts/
│   └── 
├── 
│   ├── server.key         # Private key
│   └── server.crt         # Certificate
└── [run-directories]/     # Session data
    ├── log.txt           # Session logs
    ├── copilot.md        # AI evaluations
    └── ...
```

## Development

### Auto-restart Development Server

```bash
npm run dev
```

Uses nodemon to automatically restart the server when files change.

### Security Audit

```bash
npm run security-check
```

### Adding New Evaluation Files

The server automatically discovers evaluation files in run directories. Supported files:
- `copilot.md`
- `perplexity.md`
- `claude-sonnet.md`
- `gpt-4.md`
- `gemini.md`

## Production Deployment

### Using Kubernetes (Enterprise)

For production Kubernetes deployments with persistent storage and high availability:

#### Quick Start with Helm

```bash
# Clone the repository
git clone <repository-url>
cd aiopslab-runs

# Deploy to Kubernetes
./scripts/k8s-deploy.sh -e prod -t v1.0.0

# Or deploy manually
helm install aiopslab-viewer ./helm/aiopslab-viewer \
  --namespace aiopslab-prod \
  --create-namespace \
  --values ./helm/aiopslab-viewer/values-prod.yaml \
  --set image.tag=v1.0.0
```

#### Features Included

- ✅ **Persistent Volumes**: Separate PVCs for database and runs data
- ✅ **High Availability**: Pod anti-affinity and autoscaling
- ✅ **Security**: Network policies, non-root containers, security contexts
- ✅ **Ingress**: TLS termination with cert-manager integration
- ✅ **Monitoring**: Health checks, readiness/liveness probes
- ✅ **Storage**: Configurable storage classes and sizes

#### Storage Configuration

```yaml
# Production storage example
persistence:
  database:
    enabled: true
    storageClass: "fast-ssd"
    size: 5Gi
  runs:
    enabled: true
    storageClass: "fast-ssd"
    size: 50Gi
```

See [k8s-deployment.md](./k8s-deployment.md) for comprehensive Kubernetes deployment documentation.

### Using Docker (Recommended)

#### Quick Start with Docker Compose

The easiest way to deploy is using Docker Compose:

```bash
# Clone the repository
git clone <repository-url>
cd aiopslab-runs

# Start the application
docker-compose up -d

# View logs
docker-compose logs -f aiopslab-viewer

# Stop the application
docker-compose down
```

The application will be available at:
- **HTTP**: http://localhost:3000
- **HTTPS**: https://localhost:3443

#### Building the Docker Image

```bash
# Build the image
docker build -t aiopslab-viewer .

# Run the container
docker run -d \
  --name aiopslab-viewer \
  -p 3000:3000 \
  -p 3443:3443 \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/runs.db:/app/runs.db \
  -e NODE_ENV=production \
  aiopslab-viewer
```

#### Production Deployment with Nginx

For production, use the included Docker Compose configuration with Nginx:

```bash
# Start with Nginx reverse proxy
docker-compose --profile production up -d

# This starts:
# - AIOpsLab Viewer on internal network
# - Nginx reverse proxy on ports 80/443
# - SSL termination and rate limiting
```

The production setup includes:
- ✅ **Rate limiting** for API and static files
- ✅ **Security headers** (HSTS, CSP, etc.)
- ✅ **Gzip compression** for better performance
- ✅ **Static file caching** for faster load times
- ✅ **Health checks** for container orchestration

#### Environment Variables for Docker

```bash
# Create production environment file
cat > .env.production << EOF
NODE_ENV=production
PORT=3000
HTTPS_PORT=3443
HOST=0.0.0.0
RUNS_PATH=/app/runs
DATABASE_PATH=/app/runs.db
RATE_LIMIT_WINDOW_MS=900000
RATE_LIMIT_MAX_REQUESTS=100
LOG_LEVEL=info
EOF

# Use in Docker Compose
docker-compose --env-file .env.production up -d
```

### Using PM2

For traditional server deployments:

```bash
# Install PM2
npm install -g pm2

# Start server with PM2
pm2 start server.js --name aiopslab-viewer

# Monitor
pm2 monit

# Auto-restart on reboot
pm2 startup
pm2 save
```

### Manual Docker Build

If you prefer to build manually:

```dockerfile
FROM node:18-alpine


WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy application code
COPY . .

# Create directories and generate certificates
RUN mkdir -p /app/runs && \
    chown -R node:node /app

USER node

EXPOSE 3000 3443

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000/health', (res) => { process.exit(res.statusCode === 200 ? 0 : 1) })"

CMD ["npm", "start"]
```

### Reverse Proxy (Manual Nginx)

If setting up Nginx manually:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /path/to/your/certificate.crt;
    ssl_certificate_key /path/to/your/private.key;
    
    # Security headers
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Troubleshooting

### SSL Certificate Issues

If you see SSL errors:

1. Regenerate certificates:
   ```bash
   rm -rf ssl/
   ```

2. Accept the self-signed certificate in your browser
3. For production, use Let's Encrypt or proper CA-signed certificates

### CORS Errors

If you get CORS errors:

1. Check that your origin is in the allowed list
2. Ensure you're accessing via `localhost` or allowed IP ranges
3. For production, update the CORS configuration

### File Serving Issues

If files aren't loading:

1. Check file permissions
2. Verify the file structure matches the expected layout
3. Check the server logs for 404 errors

## License

MIT License - see package.json for details.
