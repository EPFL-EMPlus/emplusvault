CREATE TABLE IF NOT EXISTS media (
    media_id SERIAL PRIMARY KEY,
    media_path VARCHAR(500) NOT NULL,
    original_path VARCHAR(500) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    media_type VARCHAR(50) NOT NULL,
    sub_type VARCHAR(50) NOT NULL,
    size INTEGER NOT NULL,
    archive_name VARCHAR (50) NOT NULL,
    archive_id VARCHAR (50) NOT NULL,
    metadata JSONB NOT NULL
);

COMMENT ON COLUMN media.media_id IS 'Unique identifier for the media';
COMMENT ON COLUMN media.media_path IS 'Path to the media file';
COMMENT ON COLUMN media.original_path IS 'Path to the original media file';
COMMENT ON COLUMN media.created_at IS 'Timestamp when the media was added to the database';
COMMENT ON COLUMN media.media_type IS 'Type of the media (image, video, audio)';
COMMENT ON COLUMN media.sub_type IS 'Subtype of the media (thumbnail, ...)';
COMMENT ON COLUMN media.size IS 'Size of the media file in bytes';
COMMENT ON COLUMN media.archive_name IS 'Name of the archive the media belongs to (rts, ioc, mjf,...)';
COMMENT ON COLUMN media.archive_id IS 'Unique identifier of the archive the media belongs to';
COMMENT ON COLUMN media.metadata IS 'Metadata of the media';


CREATE TABLE IF NOT EXISTS features (
    feature_id SERIAL PRIMARY KEY,
    feature_type VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    model_name VARCHAR(200) NOT NULL,
    model_params JSONB NOT NULL,
    data JSONB NOT NULL,

    embedding_size INTEGER,
    embedding_1024 vector (1024),
    embedding_2048 vector (2048),

    media_id INTEGER,

    CONSTRAINT FK_features_media_id FOREIGN KEY (media_id) 
        REFERENCES media (media_id)
);

COMMENT ON COLUMN features.feature_id IS 'Unique identifier for the feature';
COMMENT ON COLUMN features.feature_type IS 'Type of the feature (image, video, audio)';
COMMENT ON COLUMN features.version IS 'Version of the feature';
COMMENT ON COLUMN features.created_at IS 'Timestamp when the feature was added to the database';
COMMENT ON COLUMN features.model_name IS 'Name of the model used to extract the feature';
COMMENT ON COLUMN features.model_params IS 'Parameters of the model used to extract the feature';
COMMENT ON COLUMN features.data IS 'Other data related to the feature';
COMMENT ON COLUMN features.embedding_size IS 'Size of the embedding';
COMMENT ON COLUMN features.embedding_1024 IS '1024 dimensional embedding';
COMMENT ON COLUMN features.embedding_2048 IS '2048 dimensional embedding';
COMMENT ON COLUMN features.media_id IS 'Unique identifier of the media the feature belongs to';

CREATE TABLE tasks (
   task_id SERIAL PRIMARY KEY,
   task_name VARCHAR(200) NOT NULL,
   task_type VARCHAR(200) NOT NULL,
   task_params JSONB NOT NULL,
   created_at TIMESTAMP DEFAULT NOW(),
   updated_at TIMESTAMP DEFAULT NOW(),
   status VARCHAR(200) NOT NULL,
   task_result JSONB NOT NULL
);

CREATE INDEX features_data_jsonb_idx ON features USING GIN (data);
CREATE INDEX media_media_path_idx ON media (media_path);
