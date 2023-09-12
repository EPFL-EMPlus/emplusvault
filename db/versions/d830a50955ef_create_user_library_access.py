"""create user library access

Revision ID: d830a50955ef
Revises: 
Create Date: 2023-09-05 14:10:28.496872

"""
from alembic import op
import sqlalchemy as sa

revision = 'd830a50955ef'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create the user_library_access table
    op.create_table(
        'user_library_access',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('user_id', sa.Integer, sa.ForeignKey(
            'users.user_id'), nullable=False),
        sa.Column('library_id', sa.Integer, sa.ForeignKey(
            'library.library_id'), nullable=False),

        # Unique constraint to ensure a user cannot have multiple access entries for the same library
        sa.UniqueConstraint('user_id', 'library_id',
                            name='uq_user_library_access')
    )


def downgrade():
    # Drop the user_library_access table
    op.drop_table('user_library_access')
