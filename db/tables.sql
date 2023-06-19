DROP TABLE IF EXISTS map_projection_feature;
DROP TABLE IF EXISTS atlas;
DROP TABLE IF EXISTS projection;
DROP TABLE IF EXISTS feature;
DROP TABLE IF EXISTS media;
DROP TABLE IF EXISTS library;

CREATE TABLE IF NOT EXISTS library (
    library_id SERIAL PRIMARY KEY,
    library_name VARCHAR(50) UNIQUE,
    version VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    data JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS projection (
    projection_id SERIAL PRIMARY KEY,
    version VARCHAR(20) NOT NULL,
    library_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    model_name VARCHAR(200) NOT NULL,
    model_params JSONB NOT NULL,
    data JSONB NOT NULL,
    dimension INTEGER NOT NULL,
    atlas_folder_path VARCHAR(500) NOT NULL,
    atlas_width INTEGER NOT NULL,
    tile_size INTEGER NOT NULL,
    atlas_count INTEGER NOT NULL,
    total_tiles INTEGER NOT NULL,
    tiles_per_atlas INTEGER NOT NULL,

    CONSTRAINT FK_projection_library_id FOREIGN KEY (library_id)
        REFERENCES library (library_id)
);

CREATE TABLE IF NOT EXISTS atlas (
    atlas_id SERIAL PRIMARY KEY,
    projection_id INTEGER NOT NULL,
    atlas_order INTEGER NOT NULL,
    atlas_path VARCHAR(500) NOT NULL,
    atlas_size Vector (2) NOT NULL,
    tile_size Vector (2) NOT NULL,
    tile_count INTEGER NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    tiles_per_atlas INTEGER NOT NULL,

    CONSTRAINT FK_atlas_projection_id FOREIGN KEY (projection_id)
        REFERENCES projection (projection_id)
);

CREATE TABLE IF NOT EXISTS media (
    media_id SERIAL PRIMARY KEY,
    media_path VARCHAR(500) UNIQUE,
    original_path VARCHAR(500) NOT NULL,
    original_id VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    media_type VARCHAR(50) NOT NULL,
    file_id VARCHAR(50) NOT NULL,
    sub_type VARCHAR(50) NOT NULL,
    size INTEGER NOT NULL,
    metadata JSONB NOT NULL,
    library_id INTEGER NOT NULL,
    hash VARCHAR(50) UNIQUE,
    parent_id INTEGER,
    start_ts FLOAT,
    end_ts FLOAT,
    start_frame INTEGER,
    end_frame INTEGER,
    frame_rate FLOAT,

    CONSTRAINT FK_media_library_id FOREIGN KEY (library_id)
        REFERENCES library (library_id)
);

COMMENT ON COLUMN media.original_id IS 'The original id (ex. ZE004015 for rts) to identify the media file in the original archive.';

CREATE TABLE IF NOT EXISTS feature (
    feature_id SERIAL PRIMARY KEY,
    feature_type VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    model_name VARCHAR(200) NOT NULL,
    model_params JSONB NOT NULL,
    data JSONB NOT NULL,

    embedding_size INTEGER,
    embedding_1024 vector (1024),
    embedding_1536 vector (1536),
    embedding_2048 vector (2048),

    media_id INTEGER,

    CONSTRAINT FK_feature_media_id FOREIGN KEY (media_id) 
        REFERENCES media (media_id)
);


CREATE TABLE IF NOT EXISTS map_projection_feature (
    map_projection_feature_id SERIAL PRIMARY KEY,
    projection_id INTEGER NOT NULL,
    feature_id INTEGER,
    media_id INTEGER,
    atlas_order INTEGER NOT NULL,
    coordinates GEOMETRY(PointZ) NOT NULL,
    index_in_atlas INTEGER NOT NULL,

    CONSTRAINT FK_map_projection_feature_projection_id FOREIGN KEY (projection_id)
        REFERENCES projection (projection_id),
    CONSTRAINT FK_map_projection_feature_media_id FOREIGN KEY (media_id)
        REFERENCES media (media_id)
);
