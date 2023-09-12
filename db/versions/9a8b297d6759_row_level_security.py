"""row level security

Revision ID: 9a8b297d6759
Revises: d830a50955ef
Create Date: 2023-09-05 16:33:49.137076

"""
from alembic import op
import sqlalchemy as sa

revision = '9a8b297d6759'
down_revision = 'd830a50955ef'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
    -- Enable RLS
    ALTER TABLE media ENABLE ROW LEVEL SECURITY;
    ALTER TABLE feature ENABLE ROW LEVEL SECURITY;

    -- Remove owner rights to bypass the RLS
    ALTER TABLE media FORCE ROW LEVEL SECURITY;
    ALTER TABLE feature FORCE ROW LEVEL SECURITY;

    -- Create the policy for the media table
    CREATE POLICY user_media_access_policy
        ON media
        FOR ALL
        USING (
            library_id IN (
                SELECT library_id FROM user_library_access WHERE user_id = current_setting('emplusvault.current_user_id')::int
            )
        );
    
    CREATE POLICY user_feature_access_policy
        ON feature
        FOR ALL
        USING (
            EXISTS (
                SELECT 1 FROM user_library_access JOIN media ON user_library_access.library_id = media.library_id WHERE user_id = current_setting('emplusvault.current_user_id')::integer AND media.media_id = feature.media_id
            )
        );
               
    """)


def downgrade():
    op.execute("""
    DROP POLICY IF EXISTS user_media_access_policy ON media;
    ALTER TABLE media DISABLE ROW LEVEL SECURITY;
               
    DROP POLICY IF EXISTS user_feature_access_policy ON feature;
    ALTER TABLE feature DISABLE ROW LEVEL SECURITY;
    """)
