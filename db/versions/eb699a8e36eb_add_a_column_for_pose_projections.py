"""Add a column for pose projections

Revision ID: eb699a8e36eb
Revises: 83f4e38fabef
Create Date: 2024-06-14 11:08:25.404876

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'eb699a8e36eb'
down_revision = '83f4e38fabef'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE feature
        ADD COLUMN embedding_33 vector (33)
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE feature
        DROP COLUMN embedding_33
    """)
