#!/bin/sh

git clone --branch v0.4.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
make install

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<EOF
create extension pg_trgm;
CREATE EXTENSION vector;
select * FROM pg_extension;
EOF