const sqlite3 = require('sqlite3').verbose();
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

class RunDatabase {
    constructor(dbPath = './runs.db') {
        this.dbPath = dbPath;
        this.db = null;
        this.init();
    }

    init() {
        this.db = new sqlite3.Database(this.dbPath, sqlite3.OPEN_READWRITE | sqlite3.OPEN_CREATE, (err) => {
            if (err) {
                console.error('Error opening database:', err.message);
            } else {
                console.log('📊 Connected to SQLite database');
                this.createTables();
            }
        });
    }

    createTables() {
        // First, create the table with current schema
        const schema = `
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                modified_at TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                duration REAL DEFAULT 0,
                reasoning_judgement TEXT DEFAULT 'N/A',
                detection_accuracy TEXT DEFAULT 'Unknown',
                steps INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                reasoning_score INTEGER DEFAULT 0,
                agent_name TEXT DEFAULT 'unknown',
                application_name TEXT DEFAULT 'unknown',
                has_log_file BOOLEAN DEFAULT 0,
                evaluation_files TEXT DEFAULT '[]',
                evaluation_count INTEGER DEFAULT 0,
                last_analyzed_at TEXT NOT NULL,
                UNIQUE(id)
            );

            CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
            CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
            CREATE INDEX IF NOT EXISTS idx_runs_modified_at ON runs(modified_at);
        `;

        this.db.exec(schema, (err) => {
            if (err) {
                console.error('Error creating tables:', err.message);
            } else {
                console.log('✅ Database tables initialized');
                
                // Ensure reasoning_judgement column exists for existing databases
                this.db.run("ALTER TABLE runs ADD COLUMN reasoning_judgement TEXT DEFAULT 'N/A'", (alterErr) => {
                    if (alterErr && !alterErr.message.includes('duplicate column name')) {
                        console.error('Error adding reasoning_judgement column:', alterErr.message);
                    } else if (!alterErr) {
                        console.log('✅ Added reasoning_judgement column to existing database');
                    }
                });
            }
        });
    }

    // Calculate hash of run directory files for change detection
    calculateRunHash(runPath) {
        const hash = crypto.createHash('sha256');
        
        try {
            // Always include log.txt
            const logFile = path.join(runPath, 'log.txt');
            if (fs.existsSync(logFile)) {
                const stats = fs.statSync(logFile);
                hash.update(`log.txt:${stats.mtime.toISOString()}:${stats.size}`);
            }
            
            // Dynamically discover all .md files in the run directory
            const files = fs.readdirSync(runPath).filter(file => file.endsWith('.md'));
            
            // Sort files for consistent hashing
            files.sort();
            
            for (const file of files) {
                const filePath = path.join(runPath, file);
                if (fs.existsSync(filePath)) {
                    const stats = fs.statSync(filePath);
                    hash.update(`${file}:${stats.mtime.toISOString()}:${stats.size}`);
                }
            }
            
            return hash.digest('hex');
        } catch (error) {
            console.warn('Error calculating hash for', runPath, ':', error.message);
            return crypto.randomBytes(16).toString('hex');
        }
    }

    // Get run from database
    async getRun(runId) {
        return new Promise((resolve, reject) => {
            const query = 'SELECT * FROM runs WHERE id = ?';
            this.db.get(query, [runId], (err, row) => {
                if (err) {
                    reject(err);
                } else if (!row) {
                    resolve(null);
                } else {
                    // Parse JSON fields
                    const parsedRow = {
                        ...row,
                        evaluation_files: JSON.parse(row.evaluation_files || '[]'),
                        has_log_file: Boolean(row.has_log_file)
                    };
                    resolve(parsedRow);
                }
            });
        });
    }

    // Insert or update run data
    async upsertRun(runData) {
        return new Promise((resolve, reject) => {
            const query = `
                INSERT OR REPLACE INTO runs (
                    id, created_at, modified_at, file_hash, status, duration, reasoning_judgement,
                    detection_accuracy, steps, input_tokens, output_tokens, reasoning_score,
                    agent_name, application_name, has_log_file, evaluation_files, evaluation_count,
                    last_analyzed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            `;

            const params = [
                runData.id,
                runData.created_at,
                runData.modified_at,
                runData.file_hash,
                runData.status,
                runData.duration,
                runData.reasoning_judgement,
                runData.detection_accuracy,
                runData.steps,
                runData.input_tokens,
                runData.output_tokens,
                runData.reasoning_score,
                runData.agent_name || 'unknown',
                runData.application_name || 'unknown',
                runData.has_log_file ? 1 : 0,
                JSON.stringify(runData.evaluation_files || []),
                runData.evaluation_count || 0,
                new Date().toISOString()
            ];

            this.db.run(query, params, function(err) {
                if (err) {
                    reject(err);
                } else {
                    resolve({ changes: this.changes, lastID: this.lastID });
                }
            });
        });
    }

    // Get all runs ordered by creation date
    async getAllRuns() {
        return new Promise((resolve, reject) => {
            const query = 'SELECT * FROM runs ORDER BY created_at DESC';
            this.db.all(query, [], (err, rows) => {
                if (err) {
                    reject(err);
                } else {
                    // Parse JSON fields for all rows
                    const parsedRows = rows.map(row => ({
                        ...row,
                        evaluation_files: JSON.parse(row.evaluation_files || '[]'),
                        has_log_file: Boolean(row.has_log_file)
                    }));
                    resolve(parsedRows);
                }
            });
        });
    }

    // Check if run needs reanalysis (file changes or doesn't exist in DB)
    async needsReanalysis(runId, runPath) {
        try {
            const existingRun = await this.getRun(runId);
            if (!existingRun) {
                return true; // Run not in database
            }

            const currentHash = this.calculateRunHash(runPath);
            return existingRun.file_hash !== currentHash;
        } catch (error) {
            console.warn('Error checking if run needs reanalysis:', error.message);
            return true; // Default to reanalysis on error
        }
    }

    // Get database statistics
    async getStats() {
        return new Promise((resolve, reject) => {
            const queries = [
                'SELECT COUNT(*) as total_runs FROM runs',
                'SELECT COUNT(*) as success_runs FROM runs WHERE status = "success"',
                'SELECT COUNT(*) as partial_runs FROM runs WHERE status = "partial"',
                'SELECT COUNT(*) as failed_runs FROM runs WHERE status = "failed"',
                'SELECT AVG(duration) as avg_duration FROM runs WHERE duration > 0',
                'SELECT AVG(reasoning_score) as avg_reasoning_score FROM runs WHERE reasoning_score > 0'
            ];

            Promise.all(queries.map(query => 
                new Promise((res, rej) => {
                    this.db.get(query, [], (err, row) => {
                        if (err) rej(err);
                        else res(row);
                    });
                })
            )).then(results => {
                resolve({
                    totalRuns: results[0].total_runs,
                    successRuns: results[1].success_runs,
                    partialRuns: results[2].partial_runs,
                    failedRuns: results[3].failed_runs,
                    avgDuration: Math.round(results[4].avg_duration || 0),
                    avgReasoningScore: Math.round(results[5].avg_reasoning_score || 0)
                });
            }).catch(reject);
        });
    }

    // Clean up old runs (optional)
    async cleanupOldRuns(daysOld = 30) {
        return new Promise((resolve, reject) => {
            const cutoffDate = new Date();
            cutoffDate.setDate(cutoffDate.getDate() - daysOld);
            const cutoffISO = cutoffDate.toISOString();

            const query = 'DELETE FROM runs WHERE created_at < ?';
            this.db.run(query, [cutoffISO], function(err) {
                if (err) {
                    reject(err);
                } else {
                    console.log(`🧹 Cleaned up ${this.changes} old runs`);
                    resolve(this.changes);
                }
            });
        });
    }

    // Delete a run from the database
    async deleteRun(runId) {
        return new Promise((resolve, reject) => {
            const query = 'DELETE FROM runs WHERE id = ?';
            this.db.run(query, [runId], function(err) {
                if (err) {
                    reject(err);
                } else {
                    console.log(`🗑️ Deleted run ${runId} from database (${this.changes} rows affected)`);
                    resolve(this.changes);
                }
            });
        });
    }

    // Close database connection
    close() {
        if (this.db) {
            this.db.close((err) => {
                if (err) {
                    console.error('Error closing database:', err.message);
                } else {
                    console.log('📊 Database connection closed');
                }
            });
        }
    }
}

module.exports = RunDatabase;
