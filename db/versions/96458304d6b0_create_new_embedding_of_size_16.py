"""Create new embedding of size 16

Revision ID: 96458304d6b0
Revises: eb699a8e36eb
Create Date: 2024-11-14 15:39:01.438693

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '96458304d6b0'
down_revision = 'eb699a8e36eb'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE feature
        ADD COLUMN embedding_16 vector (16)
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE feature
        DROP COLUMN embedding_16
    """)
