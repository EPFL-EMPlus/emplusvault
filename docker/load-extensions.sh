#!/bin/sh
set -e
set -x

# # Ensure running as root to handle permissions correctly
# if [ "$(id -u)" != "0" ]; then
#     echo "This script must be run as root" 1>&2
#     exit 1
# fi

cd /home/postgres

# Check if pgvector is already cloned and remove if it is
if [ -d "pgvector" ]; then
    echo "pgvector directory exists, removing..."
    rm -rf pgvector
fi

echo "Cloning pgvector..."
git clone --branch v0.7.0 https://github.com/pgvector/pgvector.git
cd pgvector

echo "Running make..."
make -j 2 -Wunknown-attributes

echo "Running make install v2..."
make install

# Check if vector extension already exists
EXT_EXISTS=$(psql --username=postgres -tAc "SELECT 1 FROM pg_extension WHERE extname='vector'")
if [ "$EXT_EXISTS" = '1' ]; then
    echo "Vector extension already exists, skipping creation..."
else
    echo "Creating vector extension..."
    psql --username=postgres -c "CREATE EXTENSION vector;"
fi
