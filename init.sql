-- Initial database setup for AIOpsLab Task Execution API

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create custom types
CREATE TYPE task_status AS ENUM ('pending', 'running', 'completed', 'failed', 'timeout');
CREATE TYPE worker_status AS ENUM ('idle', 'busy', 'offline');
CREATE TYPE log_level AS ENUM ('debug', 'info', 'warning', 'error', 'critical');

-- Add comments
COMMENT ON TYPE task_status IS 'Status of task execution';
COMMENT ON TYPE worker_status IS 'Current status of worker';
COMMENT ON TYPE log_level IS 'Log severity levels';

-- Create indexes for JSONB queries
CREATE INDEX IF NOT EXISTS idx_gin_task_parameters ON tasks USING gin (parameters);
CREATE INDEX IF NOT EXISTS idx_gin_task_result ON tasks USING gin (result);
CREATE INDEX IF NOT EXISTS idx_gin_worker_capabilities ON workers USING gin (capabilities);
CREATE INDEX IF NOT EXISTS idx_gin_worker_metadata ON workers USING gin (metadata);

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE aiopslab_tasks TO aiopslab;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO aiopslab;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO aiopslab;