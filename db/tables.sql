CREATE TABLE media (
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

CREATE TABLE features (
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

   generic_embedding vector (2048),

   embedding_resnet vector (2048),
   embedding_openai vector (1536), 
   

-- Create vector tables for each vector feature
-- type of vectors: text, image, audio, video
CREATE TABLE embedding_text (
   feature_id INTEGER PRIMARY KEY,
   embedding vector (1536),
   CONSTRAINT FK_embedding_text_feature_id FOREIGN KEY (feature_id) 
    REFERENCES features (feature_id)
);

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

-- create index for vector column


SELECT 
    media_id,
    CASE
        WHEN embedding_size = 1024 THEN embedding_1024
        WHEN embedding_size = 2048 THEN embedding_2048
        ELSE NULL
    END AS embedding_vector
FROM 
    features
WHERE 
    media_id = <your_media_id>;