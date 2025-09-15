"""Initial database tables

Revision ID: 001
Revises:
Create Date: 2025-09-14 15:31:47

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial tables."""

    # Create enum types
    op.execute("CREATE TYPE taskstatus AS ENUM ('pending', 'running', 'completed', 'failed', 'timeout', 'cancelled')")
    op.execute("CREATE TYPE workerstatus AS ENUM ('idle', 'busy', 'offline')")
    op.execute("CREATE TYPE loglevel AS ENUM ('debug', 'info', 'warning', 'error', 'critical')")

    # Create tasks table
    op.create_table('tasks',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('problem_id', sa.String(length=255), nullable=False),
        sa.Column('status', postgresql.ENUM('pending', 'running', 'completed', 'failed', 'timeout', 'cancelled',
                                            name='taskstatus', create_type=False),
                  nullable=False, server_default='pending'),
        sa.Column('parameters', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('worker_id', sa.String(length=255), nullable=True),
        sa.Column('result', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('error_details', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint("status != 'running' OR worker_id IS NOT NULL", name='check_running_has_worker'),
        sa.CheckConstraint("status NOT IN ('completed', 'failed', 'timeout') OR completed_at IS NOT NULL",
                          name='check_terminal_has_completed_at'),
        sa.CheckConstraint("started_at IS NULL OR created_at <= started_at", name='check_started_after_created'),
        sa.CheckConstraint("completed_at IS NULL OR started_at <= completed_at", name='check_completed_after_started'),
    )

    # Create indexes for tasks
    op.create_index('idx_tasks_problem_id', 'tasks', ['problem_id'])
    op.create_index('idx_tasks_status', 'tasks', ['status'])
    op.create_index('idx_tasks_worker_id', 'tasks', ['worker_id'])
    op.create_index('idx_tasks_created_at', 'tasks', ['created_at'])
    op.create_index('idx_tasks_status_created', 'tasks', ['status', 'created_at'])
    op.create_index('idx_tasks_worker_status', 'tasks', ['worker_id', 'status'])
    op.create_index('idx_tasks_problem_id_status', 'tasks', ['problem_id', 'status'])
    op.create_index('idx_tasks_parameters', 'tasks', ['parameters'], postgresql_using='gin')
    op.create_index('idx_tasks_result', 'tasks', ['result'], postgresql_using='gin')

    # Create workers table
    op.create_table('workers',
        sa.Column('id', sa.String(length=255), nullable=False),
        sa.Column('backend_type', sa.String(length=100), nullable=False),
        sa.Column('status', postgresql.ENUM('idle', 'busy', 'offline', name='workerstatus', create_type=False),
                  nullable=False, server_default='idle'),
        sa.Column('last_heartbeat', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('current_task_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('capabilities', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('worker_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('tasks_completed', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('tasks_failed', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('registered_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint("id ~ '^worker-[0-9]{3}-kind$'", name='check_worker_id_pattern'),
        sa.CheckConstraint("status != 'busy' OR current_task_id IS NOT NULL", name='check_busy_has_task'),
        sa.CheckConstraint("status = 'busy' OR current_task_id IS NULL", name='check_not_busy_no_task'),
        sa.CheckConstraint("tasks_completed >= 0", name='check_tasks_completed_positive'),
        sa.CheckConstraint("tasks_failed >= 0", name='check_tasks_failed_positive'),
    )

    # Create indexes for workers
    op.create_index('idx_workers_backend_type', 'workers', ['backend_type'])
    op.create_index('idx_workers_status', 'workers', ['status'])
    op.create_index('idx_workers_last_heartbeat', 'workers', ['last_heartbeat'])
    op.create_index('idx_workers_status_heartbeat', 'workers', ['status', 'last_heartbeat'])
    op.create_index('idx_workers_backend_status', 'workers', ['backend_type', 'status'])
    op.create_index('idx_workers_capabilities', 'workers', ['capabilities'], postgresql_using='gin')
    op.create_index('idx_workers_worker_metadata', 'workers', ['worker_metadata'], postgresql_using='gin')

    # Create task_logs table
    op.create_table('task_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('task_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('level', postgresql.ENUM('debug', 'info', 'warning', 'error', 'critical',
                                          name='loglevel', create_type=False),
                  nullable=False, server_default='info'),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('context', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.ForeignKeyConstraint(['task_id'], ['tasks.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for task_logs
    op.create_index('idx_task_logs_task_id', 'task_logs', ['task_id'])
    op.create_index('idx_task_logs_timestamp', 'task_logs', ['timestamp'])
    op.create_index('idx_task_logs_level', 'task_logs', ['level'])
    op.create_index('idx_task_logs_task_timestamp', 'task_logs', ['task_id', 'timestamp'])
    op.create_index('idx_task_logs_task_level', 'task_logs', ['task_id', 'level'])
    op.create_index('idx_task_logs_level_timestamp', 'task_logs', ['level', 'timestamp'])
    op.create_index('idx_task_logs_context', 'task_logs', ['context'], postgresql_using='gin')


def downgrade() -> None:
    """Drop all tables and types."""
    op.drop_table('task_logs')
    op.drop_table('workers')
    op.drop_table('tasks')

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS taskstatus")
    op.execute("DROP TYPE IF EXISTS workerstatus")
    op.execute("DROP TYPE IF EXISTS loglevel")