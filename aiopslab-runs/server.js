const express = require('express');
const http = require('http');
const fs = require('fs');
const path = require('path');
const helmet = require('helmet');
const cors = require('cors');
const multer = require('multer');
const RunDatabase = require('./database');

// Load environment variables
require('dotenv').config();

const app = express();

// Configuration
const RUNS_PATH = process.env.RUNS_PATH || './runs';
const PORT = process.env.PORT || 3000;

// Ensure runs directory exists
const runsDir = path.resolve(__dirname, RUNS_PATH);
if (!fs.existsSync(runsDir)) {
    fs.mkdirSync(runsDir, { recursive: true });
    console.log(`📁 Created runs directory: ${runsDir}`);
}

// Initialize database
const runDB = new RunDatabase();

// Helper function to format duration to 2 decimal places
function formatDuration(duration) {
    if (typeof duration === 'number' && !isNaN(duration)) {
        return Math.round(duration * 100) / 100; // Round to 2 decimal places
    }
    return duration;
}

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        const runId = req.params.runId;
        const runPath = path.join(runsDir, runId);
        
        // Ensure run directory exists
        if (!fs.existsSync(runPath)) {
            fs.mkdirSync(runPath, { recursive: true });
        }
        
        cb(null, runPath);
    },
    filename: function (req, file, cb) {
        // Determine final filename based on endpoint
        let finalFilename;
        
        if (req.route.path.includes('/log')) {
            // Force log files to be named log.txt
            finalFilename = 'log.txt';
        } else if (req.route.path.includes('/evaluation')) {
            // For evaluation files, use a temporary name - final naming will be handled in the endpoint
            finalFilename = `temp_${Date.now()}_${file.originalname}`;
        } else {
            finalFilename = file.originalname;
        }
        
        cb(null, finalFilename);
    }
});



const upload = multer({ 
    storage: storage,
    limits: {
        fileSize: 10 * 1024 * 1024, // 10MB limit
        files: 1
    },
    fileFilter: function (req, file, cb) {
        // Allow text files and markdown files
        if (file.mimetype === 'text/plain' || 
            file.mimetype === 'text/markdown' ||
            file.originalname.endsWith('.txt') ||
            file.originalname.endsWith('.md') ||
            file.originalname.endsWith('.log')) {
            cb(null, true);
        } else {
            cb(new Error('Invalid file type. Only .txt, .md, and .log files are allowed.'));
        }
    }
});

// Security middleware
app.use(helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            styleSrc: ["'self'", "'unsafe-inline'"],
            scriptSrc: ["'self'", "'unsafe-inline'"],
            scriptSrcAttr: ["'unsafe-inline'"], // Allow inline event handlers
            imgSrc: ["'self'", "data:", "https:"],
            fontSrc: ["'self'", "https:", "data:"],
            connectSrc: ["'self'"],
            mediaSrc: ["'self'"],
            objectSrc: ["'none'"],
            childSrc: ["'self'"],
            frameSrc: ["'none'"],
            workerSrc: ["'self'"],
            frameAncestors: ["'none'"],
            formAction: ["'self'"],
            upgradeInsecureRequests: []
        }
    },
    crossOriginEmbedderPolicy: false
}));

// CORS configuration
app.use(cors({
    origin: function (origin, callback) {
        // Allow requests with no origin (like mobile apps or curl requests)
        if (!origin) return callback(null, true);
        
        // Allow localhost and local network requests
        const allowedOrigins = [
            /^https?:\/\/localhost(:\d+)?$/,
            /^https?:\/\/127\.0\.0\.1(:\d+)?$/,
            /^https?:\/\/0\.0\.0\.0(:\d+)?$/,
            /^https?:\/\/192\.168\.\d+\.\d+(:\d+)?$/,
            /^https?:\/\/10\.\d+\.\d+\.\d+(:\d+)?$/
        ];
        
        if (allowedOrigins.some(pattern => pattern.test(origin))) {
            return callback(null, true);
        }
        
        return callback(new Error('Not allowed by CORS'));
    },
    credentials: true
}));

// Custom security headers
app.use((req, res, next) => {
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('X-Frame-Options', 'DENY');
    res.setHeader('X-XSS-Protection', '1; mode=block');
    res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
    res.setHeader('Permissions-Policy', 'geolocation=(), microphone=(), camera=()');
    next();
});

// JSON body parsing middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Logging middleware
app.use((req, res, next) => {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] ${req.method} ${req.url} - ${req.ip}`);
    next();
});

// Rate limiting for security
const rateLimit = require('express-rate-limit');
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // limit each IP to 100 requests per windowMs
    message: 'Too many requests from this IP, please try again later.',
    standardHeaders: true,
    legacyHeaders: false,
});
app.use(limiter);

// Serve static files with proper MIME types
app.use(express.static(__dirname, {
    setHeaders: (res, path) => {
        if (path.endsWith('.html')) {
            res.setHeader('Content-Type', 'text/html; charset=utf-8');
        } else if (path.endsWith('.css')) {
            res.setHeader('Content-Type', 'text/css; charset=utf-8');
        } else if (path.endsWith('.js')) {
            res.setHeader('Content-Type', 'application/javascript; charset=utf-8');
        } else if (path.endsWith('.json')) {
            res.setHeader('Content-Type', 'application/json; charset=utf-8');
        } else if (path.endsWith('.md')) {
            res.setHeader('Content-Type', 'text/markdown; charset=utf-8');
        } else if (path.endsWith('.txt')) {
            res.setHeader('Content-Type', 'text/plain; charset=utf-8');
        }
        
        // Cache control for static assets
        if (path.match(/\.(css|js|png|jpg|jpeg|gif|ico|svg)$/)) {
            res.setHeader('Cache-Control', 'public, max-age=86400'); // 1 day
        } else {
            res.setHeader('Cache-Control', 'no-cache');
        }
    }
}));

// Serve run files from the runs directory
app.use('/runs', express.static(runsDir, {
    setHeaders: (res, path) => {
        if (path.endsWith('.md')) {
            res.setHeader('Content-Type', 'text/markdown; charset=utf-8');
        } else if (path.endsWith('.txt')) {
            res.setHeader('Content-Type', 'text/plain; charset=utf-8');
        } else if (path.endsWith('.log')) {
            res.setHeader('Content-Type', 'text/plain; charset=utf-8');
        }
        res.setHeader('Cache-Control', 'no-cache');
    }
}));

// API endpoint to list runs from database (no filesystem scanning)
app.get('/api/runs', async (req, res) => {
    try {
        console.log('📊 Loading runs from database...');
        const runs = await runDB.getAllRuns();
        
        // Transform database format to API format
        const formattedRuns = runs.map(runData => ({
            id: runData.id,
            created: runData.created_at,
            modified: runData.modified_at,
            hasLogFile: runData.has_log_file,
            evaluationFiles: runData.evaluation_files,
            evaluationCount: runData.evaluation_count,
            status: runData.status,
            duration: formatDuration(runData.duration),
            reasoning_judgement: runData.reasoning_judgement,
            detectionAccuracy: runData.detection_accuracy,
            steps: runData.steps,
            inputTokens: runData.input_tokens,
            outputTokens: runData.output_tokens,
            reasoningScore: runData.reasoning_score,
            agentName: runData.agent_name,
            applicationName: runData.application_name,
            lastAnalyzed: runData.last_analyzed_at
        }));
        
        console.log(`💾 Loaded ${formattedRuns.length} runs from database`);
        res.json(formattedRuns);
    } catch (error) {
        console.error('Error loading runs from database:', error);
        res.status(500).json({ error: 'Failed to load runs from database' });
    }
});

// Get individual run data
app.get('/api/runs/:runId', async (req, res) => {
    try {
        const { runId } = req.params;
        console.log(`🔍 Getting run data for: ${runId}`);
        
        const runData = await runDB.getRun(runId);
        if (!runData) {
            return res.status(404).json({ error: 'Run not found' });
        }
        
        // Transform database format to API format
        const formattedRun = {
            id: runData.id,
            created: runData.created_at,
            modified: runData.modified_at,
            hasLogFile: runData.has_log_file,
            evaluationFiles: runData.evaluation_files,
            evaluationCount: runData.evaluation_count,
            status: runData.status,
            duration: formatDuration(runData.duration),
            reasoning_judgement: runData.reasoning_judgement,
            detectionAccuracy: runData.detection_accuracy,
            steps: runData.steps,
            inputTokens: runData.input_tokens,
            outputTokens: runData.output_tokens,
            reasoningScore: runData.reasoning_score,
            agentName: runData.agent_name,
            applicationName: runData.application_name,
            lastAnalyzed: runData.last_analyzed_at,
            evaluation_files: runData.evaluation_files // Also provide snake_case for viewer.html
        };
        
        res.json(formattedRun);
    } catch (error) {
        console.error(`Error getting run data for ${req.params.runId}:`, error);
        res.status(500).json({ error: 'Failed to get run data' });
    }
});

// Analyze run data to determine status and extract metrics
function analyzeRunData(runPath, runId) {
    const analysis = {
        status: 'unknown',
        duration: 0,
        reasoning_judgement: 'N/A',
        detectionAccuracy: 'Unknown',
        steps: 0,
        inputTokens: 0,
        outputTokens: 0,
        reasoningScore: 0,
        agentName: 'unknown',
        applicationName: 'unknown'
    };
    
    try {
        // Analyze log file
        const logPath = path.join(runPath, 'log.txt');
        if (fs.existsSync(logPath)) {
            const logContent = fs.readFileSync(logPath, 'utf8');
            // Extract metrics from log
            const metricsMatch = logContent.match(/Results:\s*(\{[\s\S]*?\})\s*(?=\n==|\n[A-Z]|$)/);
            if (metricsMatch) {
                let metricsStr = metricsMatch[1];
                
                try {
                    // More robust parsing for Python dict format with complex string values
                    const metrics = {};
                    
                    // Extract Detection Accuracy (can be quoted string)
                    const detectionMatch = metricsStr.match(/'Detection[_ ]Accuracy':\s*'([^']+)'/);
                    if (detectionMatch) metrics.Detection_Accuracy = detectionMatch[1];
                    
                    // Extract numeric fields
                    const ttdMatch = metricsStr.match(/'TTD':\s*([\d.]+)/);
                    if (ttdMatch) metrics.TTD = parseFloat(ttdMatch[1]);
                    
                    const stepsMatch = metricsStr.match(/'steps':\s*(\d+)/);
                    if (stepsMatch) metrics.steps = parseInt(stepsMatch[1]);
                    
                    const inTokensMatch = metricsStr.match(/'in_tokens':\s*(\d+)/);
                    if (inTokensMatch) metrics.in_tokens = parseInt(inTokensMatch[1]);
                    
                    const outTokensMatch = metricsStr.match(/'out_tokens':\s*(\d+)/);
                    if (outTokensMatch) metrics.out_tokens = parseInt(outTokensMatch[1]);
                    
                    const reasoningScoreMatch = metricsStr.match(/'reasoning_score':\s*(\d+)/);
                    if (reasoningScoreMatch) metrics.reasoning_score = parseInt(reasoningScoreMatch[1]);
                    
                    // Extract reasoning_judgement (complex string with nested quotes)
                    // Find the start and end of the reasoning_judgement value
                    const reasoningJudgementMatch = metricsStr.match(/'reasoning_judgement':\s*'([\s\S]*?)'(?=,\s*'[^']*':|$)/);
                    if (reasoningJudgementMatch) {
                        // Clean up the extracted string by handling escaped quotes
                        let reasoningText = reasoningJudgementMatch[1];
                        // Unescape single quotes within the string
                        reasoningText = reasoningText.replace(/\\'/g, "'");
                        metrics.reasoning_judgement = reasoningText;
                    }
                    
                    console.log(`Parsed metrics for ${runId}:`, {
                        Detection_Accuracy: metrics.Detection_Accuracy,
                        TTD: metrics.TTD,
                        steps: metrics.steps,
                        in_tokens: metrics.in_tokens,
                        out_tokens: metrics.out_tokens,
                        reasoning_score: metrics.reasoning_score,
                        reasoning_judgement_length: metrics.reasoning_judgement ? metrics.reasoning_judgement.length : 0
                    });

                    analysis.detectionAccuracy = metrics.Detection_Accuracy || 'Unknown';
                    analysis.duration = metrics.TTD || 0;
                    analysis.steps = metrics.steps || 0;
                    analysis.inputTokens = metrics.in_tokens || 0;
                    analysis.outputTokens = metrics.out_tokens || 0;
                    analysis.reasoningScore = metrics.reasoning_score || 0;
                    analysis.reasoning_judgement = metrics.reasoning_judgement || 'N/A';
                    
                } catch (parseError) {
                    console.error(`Failed to parse metrics for ${runId}:`, parseError.message);
                    console.error(`Raw metrics string (first 500 chars):`, metricsStr.substring(0, 500));
                    // Leave metrics with default values when parsing fails
                }
            }
        }
        
        // Determine status based on analysis
        analysis.status = determineRunStatus(analysis, runPath);
        
    } catch (error) {
        console.warn(`Error analyzing run ${runId}:`, error.message);
    }
    
    return analysis;
}


// Generate unique filename for evaluation files with numbering if needed
function generateUniqueEvaluationFilename(runPath, baseName) {
    let filename = `${baseName}.md`;
    let counter = 1;
    
    // Check if file already exists
    while (fs.existsSync(path.join(runPath, filename))) {
        filename = `${baseName}-${counter}.md`;
        counter++;
    }
    
    return filename;
}

// Dynamically discover evaluation files in a run directory
function discoverEvaluationFiles(runPath) {
    const evaluationFiles = [];
    
    try {
        if (fs.existsSync(runPath)) {
            const files = fs.readdirSync(runPath);
            for (const file of files) {
                if (file.endsWith('.md') && file !== 'README.md') {
                    const filePath = path.join(runPath, file);
                    const stats = fs.statSync(filePath);
                    if (stats.isFile()) {
                        evaluationFiles.push(file);
                    }
                }
            }
        }
    } catch (error) {
        console.warn(`Error discovering evaluation files in ${runPath}:`, error.message);
    }
    
    return evaluationFiles;
}


// Determine run status based on analysis
function determineRunStatus(analysis, runPath) {
    const logPath = path.join(runPath, 'log.txt');
    
    // Check if log file exists
    if (!fs.existsSync(logPath)) {
        return 'failed';
    }
    
    const logContent = fs.readFileSync(logPath, 'utf8');
    
    // Check for critical failures
    if (logContent.includes('Fatal error') || 
        logContent.includes('Terminated')) {
        return 'failed';
    }
    
    // Check for successful resolution indicators
    const hasResolution = logContent.includes('root cause') || 
                         logContent.includes('solution') ||
                         logContent.includes('resolved') ||
                         analysis.reasoningScore >= 8;
    
    // Determine status based on multiple factors
    if (hasResolution && analysis.reasoningScore >= 7) {
        return 'success';
    } else if (analysis.reasoningScore >= 5 || 
               (analysis.duration > 0 && analysis.duration < 600)) {
        return 'partial';
    } else {
        return 'failed';
    }
}

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ 
        status: 'healthy', 
        timestamp: new Date().toISOString(),
        uptime: process.uptime()
    });
});

// Database statistics endpoint
app.get('/api/stats', async (req, res) => {
    try {
        const stats = await runDB.getStats();
        res.json(stats);
    } catch (error) {
        console.error('Error getting database statistics:', error);
        res.status(500).json({ error: 'Failed to get database statistics' });
    }
});

// API endpoint to delete a run completely (database record and filesystem files)
app.delete('/api/runs/:runId', async (req, res) => {
    try {
        const { runId } = req.params;
        const { filesOnly = false } = req.query; // Optional parameter to delete only files
        
        console.log(`🗑️ Deleting run: ${runId} (files: ${!filesOnly ? 'yes' : 'no'}, database: ${!filesOnly ? 'yes' : 'no'})`);
        
        const runPath = path.join(runsDir, runId);
        let deletedFiles = [];
        let deletedFromDatabase = false;
        
        // Check if run exists in database
        const existingRun = await runDB.getRun(runId);
        if (!existingRun) {
            return res.status(404).json({ error: 'Run not found in database' });
        }
        
        // Delete filesystem files and directory
        if (fs.existsSync(runPath)) {
            try {
                // Get list of files before deletion for reporting
                const files = fs.readdirSync(runPath);
                deletedFiles = files;
                
                // Remove all files in the directory
                for (const file of files) {
                    const filePath = path.join(runPath, file);
                    fs.unlinkSync(filePath);
                    console.log(`🗂️ Deleted file: ${file}`);
                }
                
                // Remove the directory itself
                fs.rmdirSync(runPath);
                console.log(`📁 Deleted directory: ${runPath}`);
            } catch (fileError) {
                console.error(`Error deleting files for run ${runId}:`, fileError);
                return res.status(500).json({ 
                    error: 'Failed to delete run files',
                    details: fileError.message 
                });
            }
        }
        
        // Delete from database (unless filesOnly is true)
        if (!filesOnly) {
            try {
                await runDB.deleteRun(runId);
                deletedFromDatabase = true;
                console.log(`💾 Deleted run from database: ${runId}`);
            } catch (dbError) {
                console.error(`Error deleting run from database:`, dbError);
                return res.status(500).json({ 
                    error: 'Files deleted but failed to remove database record',
                    details: dbError.message 
                });
            }
        }
        
        res.json({ 
            success: true,
            message: `Run ${runId} deleted successfully`,
            deleted: {
                files: deletedFiles,
                filesCount: deletedFiles.length,
                directory: !fs.existsSync(runPath),
                database: deletedFromDatabase
            }
        });
    } catch (error) {
        console.error('Error deleting run:', error);
        res.status(500).json({ error: 'Failed to delete run' });
    }
});

// Force reanalysis of a specific run
app.post('/api/runs/:runId/reanalyze', async (req, res) => {
    try {
        const { runId } = req.params;
        const runPath = path.join(runsDir, runId);
        
        if (!fs.existsSync(runPath)) {
            return res.status(404).json({ error: 'Run not found' });
        }
        
        console.log(`🔄 Force reanalyzing run: ${runId}`);
        
        const stats = fs.statSync(runPath);
        const hasLogFile = fs.existsSync(path.join(runPath, 'log.txt'));
        const evaluationFiles = discoverEvaluationFiles(runPath);
        
        const runAnalysis = analyzeRunData(runPath, runId);
        
        // Get existing run data to preserve manually set fields
        const existingRun = await runDB.getRun(runId);
        
        const runData = {
            id: runId,
            created_at: stats.birthtime.toISOString(),
            modified_at: stats.mtime.toISOString(),
            file_hash: runDB.calculateRunHash(runPath),
            has_log_file: hasLogFile,
            evaluation_files: evaluationFiles,
            evaluation_count: evaluationFiles.length,
            // Map camelCase to snake_case for database
            status: runAnalysis.status,
            duration: runAnalysis.duration,
            issues: runAnalysis.issues,
            reasoning_judgement: runAnalysis.reasoning_judgement,
            detection_accuracy: runAnalysis.detectionAccuracy,
            steps: runAnalysis.steps,
            input_tokens: runAnalysis.inputTokens,
            output_tokens: runAnalysis.outputTokens,
            reasoning_score: runAnalysis.reasoningScore,
            // Preserve existing agent_name and application_name if they exist and are not 'unknown'
            agent_name: (existingRun?.agent_name && existingRun.agent_name !== 'unknown') 
                ? existingRun.agent_name 
                : runAnalysis.agentName,
            application_name: (existingRun?.application_name && existingRun.application_name !== 'unknown') 
                ? existingRun.application_name 
                : runAnalysis.applicationName,
            namespace: runAnalysis.namespace,
            errors: runAnalysis.errors
        };
        
        await runDB.upsertRun(runData);
        
        res.json({ 
            message: 'Run reanalyzed successfully',
            runData: {
                id: runData.id,
                status: runData.status,
                duration: formatDuration(runData.duration),
                reasoning_judgement: runData.reasoning_judgement,
                lastAnalyzed: new Date().toISOString()
            }
        });
    } catch (error) {
        console.error('Error reanalyzing run:', error);
        res.status(500).json({ error: 'Failed to reanalyze run' });
    }
});

// Database cleanup endpoint
app.post('/api/cleanup', async (req, res) => {
    try {
        const daysOld = parseInt(req.query.days || req.body.days || 30);
        const deletedCount = await runDB.cleanupOldRuns(daysOld);
        res.json({ 
            message: `Cleaned up ${deletedCount} old runs`,
            deletedCount
        });
    } catch (error) {
        console.error('Error cleaning up database:', error);
        res.status(500).json({ error: 'Failed to cleanup database' });
    }
});

// Delete all records endpoint (dangerous operation)
app.post('/api/delete-all', async (req, res) => {
    try {
        const { confirm } = req.body;
        
        if (confirm !== 'DELETE_ALL_RECORDS') {
            return res.status(400).json({ 
                error: 'Missing confirmation',
                message: 'You must send {"confirm": "DELETE_ALL_RECORDS"} to perform this operation'
            });
        }
        
        console.log('🗑️ DANGER: Deleting ALL database records');
        
        // Delete all records from the database
        const deletedCount = await new Promise((resolve, reject) => {
            runDB.db.run('DELETE FROM runs', function(err) {
                if (err) {
                    reject(err);
                } else {
                    console.log(`🧹 Deleted ALL ${this.changes} records from database`);
                    resolve(this.changes);
                }
            });
        });
        
        res.json({ 
            message: `Deleted all ${deletedCount} records from database`,
            deletedCount
        });
    } catch (error) {
        console.error('Error deleting all records:', error);
        res.status(500).json({ error: 'Failed to delete all records' });
    }
});

// API endpoint to scan filesystem and import/update runs in database
app.post('/api/runs/scan', async (req, res) => {
    try {
        console.log('🔍 Scanning filesystem for runs...');
        const importedRuns = [];
        const entries = fs.readdirSync(runsDir, { withFileTypes: true });
        
        for (const entry of entries) {
            if (entry.isDirectory() && entry.name.match(/^\d{8}-[a-f0-9]+$/)) {
                const runPath = path.join(runsDir, entry.name);
                const stats = fs.statSync(runPath);
                
                console.log(`📁 Processing run: ${entry.name}`);
                
                // Check for required files
                const hasLogFile = fs.existsSync(path.join(runPath, 'log.txt'));
                const evaluationFiles = discoverEvaluationFiles(runPath);
                
                // Analyze run data for intelligent status assessment
                const runAnalysis = analyzeRunData(runPath, entry.name);
                
                // Get existing run data to preserve manually set fields
                const existingRun = await runDB.getRun(entry.name);
                
                // Prepare data for database
                const runData = {
                    id: entry.name,
                    created_at: stats.birthtime.toISOString(),
                    modified_at: stats.mtime.toISOString(),
                    file_hash: runDB.calculateRunHash(runPath),
                    has_log_file: hasLogFile,
                    evaluation_files: evaluationFiles,
                    evaluation_count: evaluationFiles.length,
                    status: runAnalysis.status,
                    duration: runAnalysis.duration,
                    issues: runAnalysis.issues,
                    reasoning_judgement: runAnalysis.reasoning_judgement,
                    detection_accuracy: runAnalysis.detectionAccuracy,
                    steps: runAnalysis.steps,
                    input_tokens: runAnalysis.inputTokens,
                    output_tokens: runAnalysis.outputTokens,
                    reasoning_score: runAnalysis.reasoningScore,
                    // Preserve existing agent_name and application_name if they exist and are not 'unknown'
                    agent_name: (existingRun?.agent_name && existingRun.agent_name !== 'unknown') 
                        ? existingRun.agent_name 
                        : runAnalysis.agentName,
                    application_name: (existingRun?.application_name && existingRun.application_name !== 'unknown') 
                        ? existingRun.application_name 
                        : runAnalysis.applicationName,
                    namespace: runAnalysis.namespace,
                    errors: runAnalysis.errors
                };
                
                // Save to database
                await runDB.upsertRun(runData);
                importedRuns.push({
                    id: entry.name,
                    status: 'imported'
                });
                
                console.log(`✅ Imported run: ${entry.name}`);
            }
        }
        
        console.log(`🎉 Scan complete. Imported ${importedRuns.length} runs.`);
        res.json({
            success: true,
            message: `Scanned and imported ${importedRuns.length} runs`,
            runs: importedRuns
        });
    } catch (error) {
        console.error('Error scanning and importing runs:', error);
        res.status(500).json({ error: 'Failed to scan and import runs' });
    }
});

// API endpoint to create a new run record
app.post('/api/runs', async (req, res) => {
    try {
        const { runId, agentName, applicationName, status = 'unknown', duration = 0, score = 'N/A' } = req.body;
        
        if (!runId) {
            return res.status(400).json({ error: 'runId is required' });
        }
        
        if (!agentName || !applicationName) {
            return res.status(400).json({ error: 'agentName and applicationName are required' });
        }
        
        // Validate runId format (should be like 20250715-57fff059)
        if (!runId.match(/^\d{8}-[a-f0-9]+$/)) {
            return res.status(400).json({ error: 'Invalid runId format. Expected format: YYYYMMDD-hash' });
        }
        
        const runPath = path.join(runsDir, runId);
        
        // Create run directory if it doesn't exist
        if (!fs.existsSync(runPath)) {
            fs.mkdirSync(runPath, { recursive: true });
        }
        
        const stats = fs.existsSync(runPath) ? fs.statSync(runPath) : null;
        const hasLogFile = fs.existsSync(path.join(runPath, 'log.txt'));
        
        const runData = {
            id: runId,
            created_at: stats ? stats.birthtime.toISOString() : new Date().toISOString(),
            modified_at: new Date().toISOString(),
            file_hash: runDB.calculateRunHash(runPath),
            has_log_file: hasLogFile,
            evaluation_files: [],
            evaluation_count: 0,
            status,
            duration,
            issues: 0,
            score,
            detection_accuracy: 'Unknown',
            steps: 0,
            input_tokens: 0,
            output_tokens: 0,
            reasoning_score: 0,
            namespace: applicationName, // Use applicationName as namespace
            agent_name: agentName, // Store agent name
            application_name: applicationName, // Store application name
            errors: []
        };
        
        await runDB.upsertRun(runData);
        
        console.log(`📝 Created new run record: ${runId}`);
        res.json({
            success: true,
            message: `Run ${runId} created successfully`,
            runData: {
                id: runData.id,
                status: runData.status,
                created: runData.created_at
            }
        });
    } catch (error) {
        console.error('Error creating run record:', error);
        res.status(500).json({ error: 'Failed to create run record' });
    }
});

// API endpoint to delete multiple runs at once
app.post('/api/runs/delete-batch', async (req, res) => {
    try {
        const { runIds, deleteFiles = true } = req.body;
        
        if (!Array.isArray(runIds) || runIds.length === 0) {
            return res.status(400).json({ error: 'runIds must be a non-empty array' });
        }
        
        console.log(`🗑️ Batch deleting ${runIds.length} runs (files: ${deleteFiles})`);
        
        const results = {
            successful: [],
            failed: [],
            totalDeleted: 0,
            totalFailed: 0
        };
        
        for (const runId of runIds) {
            try {
                const runPath = path.join(runsDir, runId);
                let deletedFiles = [];
                
                // Check if run exists in database
                const existingRun = await runDB.getRun(runId);
                if (!existingRun) {
                    results.failed.push({
                        runId,
                        error: 'Run not found in database'
                    });
                    continue;
                }
                
                // Delete filesystem files if requested
                if (deleteFiles && fs.existsSync(runPath)) {
                    const files = fs.readdirSync(runPath);
                    deletedFiles = files;
                    
                    // Remove all files in the directory
                    for (const file of files) {
                        const filePath = path.join(runPath, file);
                        fs.unlinkSync(filePath);
                    }
                    
                    // Remove the directory itself
                    fs.rmdirSync(runPath);
                }
                
                // Delete from database
                await runDB.deleteRun(runId);
                
                results.successful.push({
                    runId,
                    deletedFiles: deletedFiles.length,
                    deletedFromDatabase: true
                });
                
                results.totalDeleted++;
                console.log(`✅ Deleted run: ${runId}`);
                
            } catch (error) {
                console.error(`Error deleting run ${runId}:`, error);
                results.failed.push({
                    runId,
                    error: error.message
                });
                results.totalFailed++;
            }
        }
        
        console.log(`🎉 Batch deletion complete. Success: ${results.totalDeleted}, Failed: ${results.totalFailed}`);
        
        res.json({
            success: true,
            message: `Batch deletion completed. ${results.totalDeleted} runs deleted, ${results.totalFailed} failed.`,
            results
        });
        
    } catch (error) {
        console.error('Error in batch deletion:', error);
        res.status(500).json({ error: 'Failed to perform batch deletion' });
    }
});

// API endpoint to upload log file for a run
app.post('/api/runs/:runId/log', upload.single('logFile'), async (req, res) => {
    try {
        const runId = req.params.runId;
        
        if (!req.file) {
            return res.status(400).json({ error: 'No log file provided' });
        }
        
        const runPath = path.join(runsDir, runId);
        
        // Verify run exists in database
        const existingRun = await runDB.getRun(runId);
        if (!existingRun) {
            return res.status(404).json({ error: 'Run not found' });
        }
        
        // Analyze the uploaded log file
        const runAnalysis = analyzeRunData(runPath, runId);
        
        // Update run data with log file analysis
        const updatedData = {
            ...existingRun,
            modified_at: new Date().toISOString(),
            file_hash: runDB.calculateRunHash(runPath),
            has_log_file: true,
            status: runAnalysis.status,
            duration: runAnalysis.duration,
            issues: runAnalysis.issues,
            reasoning_judgement: runAnalysis.reasoning_judgement,
            detection_accuracy: runAnalysis.detectionAccuracy,
            steps: runAnalysis.steps,
            input_tokens: runAnalysis.inputTokens,
            output_tokens: runAnalysis.outputTokens,
            reasoning_score: runAnalysis.reasoningScore,
            // Preserve existing agent_name and application_name if they exist and are not 'unknown'
            agent_name: (existingRun.agent_name && existingRun.agent_name !== 'unknown') 
                ? existingRun.agent_name 
                : runAnalysis.agentName,
            application_name: (existingRun.application_name && existingRun.application_name !== 'unknown') 
                ? existingRun.application_name 
                : runAnalysis.applicationName,
            namespace: runAnalysis.namespace,
            errors: runAnalysis.errors
        };
        
        await runDB.upsertRun(updatedData);
        
        console.log(`📄 Log file uploaded for run: ${runId} (saved as log.txt)`);
        res.json({
            success: true,
            message: `Log file uploaded successfully for run ${runId}`,
            filename: 'log.txt',
            originalName: req.file.originalname,
            analysis: runAnalysis
        });
    } catch (error) {
        console.error('Error uploading log file:', error);
        res.status(500).json({ error: 'Failed to upload log file' });
    }
});

// API endpoint to upload evaluation file for a run
app.post('/api/runs/:runId/evaluation', upload.single('evaluationFile'), async (req, res) => {
    try {
        const runId = req.params.runId;
        const targetName = req.body.targetName || 'Eval'; // Default to 'Eval' if not provided
        
        if (!req.file) {
            return res.status(400).json({ error: 'No evaluation file provided' });
        }
        
        if (!req.file.originalname.endsWith('.md')) {
            return res.status(400).json({ error: 'Evaluation file must be a .md file' });
        }
        
        const runPath = path.join(runsDir, runId);
        
        // Verify run exists in database
        const existingRun = await runDB.getRun(runId);
        if (!existingRun) {
            return res.status(404).json({ error: 'Run not found' });
        }
        
        // Determine the final filename with numbering if needed
        const finalFilename = generateUniqueEvaluationFilename(runPath, targetName);
        
        // Move the uploaded file to the correct location with the new name
        const tempPath = req.file.path;
        const finalPath = path.join(runPath, finalFilename);
        
        try {
            fs.renameSync(tempPath, finalPath);
        } catch (moveError) {
            console.error('Error moving uploaded file:', moveError);
            return res.status(500).json({ error: 'Failed to save evaluation file' });
        }
        
        // Update evaluation files list
        const currentEvaluationFiles = Array.isArray(existingRun.evaluation_files) 
            ? existingRun.evaluation_files 
            : JSON.parse(existingRun.evaluation_files || '[]');
            
        if (!currentEvaluationFiles.includes(finalFilename)) {
            currentEvaluationFiles.push(finalFilename);
        }        
        
        // Update run data
        const updatedData = {
            ...existingRun,
            modified_at: new Date().toISOString(),
            evaluation_files: currentEvaluationFiles,
            evaluation_count: currentEvaluationFiles.length
        };
        
        // Update status if needed
        if (updatedData.has_log_file && updatedData.evaluation_count > 0) {
            updatedData.status = determineRunStatus(updatedData, runPath);
        }
        
        await runDB.upsertRun(updatedData);
        
        console.log(`📝 Evaluation file uploaded for run: ${runId} (saved as ${finalFilename})`);
        res.json({
            success: true,
            message: `Evaluation file uploaded successfully for run ${runId}`,
            filename: finalFilename,
            originalName: req.file.originalname,
            targetName: targetName,
            evaluationFiles: currentEvaluationFiles,
            evaluationCount: currentEvaluationFiles.length
        });
    } catch (error) {
        console.error('Error uploading evaluation file:', error);
        res.status(500).json({ error: 'Failed to upload evaluation file' });
    }
});

// API endpoint to update a run record
app.put('/api/runs/:runId', async (req, res) => {
    try {
        const { runId } = req.params;
        const updateData = req.body;
        
        console.log(`📝 Updating run ${runId}`);
        
        // Get existing run to preserve unchanged fields
        const existingRun = await runDB.getRun(runId);
        if (!existingRun) {
            return res.status(404).json({ error: 'Run not found' });
        }
        
        // Merge existing data with updates
        const updatedRun = {
            ...existingRun,
            ...updateData,
            id: runId, // Ensure ID cannot be changed
            modified_at: new Date().toISOString(),
            last_analyzed_at: new Date().toISOString()
        };
        
        // Update in database
        await runDB.upsertRun(updatedRun);
        
        res.json({
            message: 'Run updated successfully',
            run: updatedRun
        });
        
    } catch (error) {
        console.error('Error updating run:', error);
        res.status(500).json({ error: 'Failed to update run' });
    }
});

// API endpoint for bulk reanalysis
app.post('/api/runs/bulk-reanalyze', async (req, res) => {
    try {
        const { runIds, reanalyzeAll = false } = req.body;
        
        let targetRunIds = [];
        
        if (reanalyzeAll) {
            // Get all run IDs from database
            const allRuns = await runDB.getAllRuns();
            targetRunIds = allRuns.map(run => run.id);
        } else {
            if (!Array.isArray(runIds) || runIds.length === 0) {
                return res.status(400).json({ error: 'runIds must be a non-empty array when reanalyzeAll is false' });
            }
            targetRunIds = runIds;
        }
        
        console.log(`🔄 Starting bulk reanalysis of ${targetRunIds.length} runs`);
        
        const results = {
            successful: [],
            failed: [],
            totalProcessed: targetRunIds.length,
            successCount: 0,
            failedCount: 0
        };
        
        // Process each run
        for (const runId of targetRunIds) {
            try {
                const runPath = path.join(runsDir, runId);
                
                // Check if run directory exists
                if (!fs.existsSync(runPath)) {
                    results.failed.push({
                        runId,
                        error: 'Run directory not found'
                    });
                    results.failedCount++;
                    continue;
                }
                
                // Trigger reanalysis
                const analysisResult = await analyzeRun(runId, runPath);
                
                if (analysisResult) {
                    results.successful.push(runId);
                    results.successCount++;
                } else {
                    results.failed.push({
                        runId,
                        error: 'Analysis returned no result'
                    });
                    results.failedCount++;
                }
                
            } catch (error) {
                results.failed.push({
                    runId,
                    error: error.message
                });
                results.failedCount++;
            }
        }
        
        console.log(`✅ Bulk reanalysis complete: ${results.successCount} successful, ${results.failedCount} failed`);
        
        res.json({
            message: `Bulk reanalysis complete`,
            results
        });
        
    } catch (error) {
        console.error('Error in bulk reanalysis:', error);
        res.status(500).json({ error: 'Failed to perform bulk reanalysis' });
    }
});

// Admin route
app.get('/admin', (req, res) => {
    res.sendFile(path.join(__dirname, 'admin.html'));
});

// Main viewer route
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Start the server
const server = app.listen(PORT, () => {
    console.log(`🚀 AIOpsLab server running on port ${PORT}`);
    console.log(`📊 Dashboard: http://localhost:${PORT}`);
    console.log(`🔧 Admin Panel: http://localhost:${PORT}/admin`);
    console.log(`👁️ Viewer: http://localhost:${PORT}/viewer.html`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('SIGTERM signal received: closing HTTP server');
    server.close(() => {
        console.log('HTTP server closed');
        if (runDB) {
            runDB.close();
        }
    });
});

process.on('SIGINT', () => {
    console.log('SIGINT signal received: closing HTTP server');
    server.close(() => {
        console.log('HTTP server closed');
        if (runDB) {
            runDB.close();
        }
    });
});


