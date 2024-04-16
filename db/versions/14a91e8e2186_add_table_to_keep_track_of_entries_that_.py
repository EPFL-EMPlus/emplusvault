"""Add table to keep track of entries that need to be added to the queue

Revision ID: 14a91e8e2186
Revises: 9a8b297d6759
Create Date: 2024-02-27 13:13:44.376025

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '14a91e8e2186'
down_revision = '9a8b297d6759'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'entries_to_queue',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('job_type', sa.String, nullable=False),
        sa.Column('media_id', sa.String, nullable=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime, server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False)
    )


def downgrade() -> None:
    op.drop_table('entries_to_queue')
