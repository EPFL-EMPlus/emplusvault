#!/bin/sh
set -e
set -x

cd /home/postgres
echo "Cloning pgvector..."
git clone --branch v0.4.1 https://github.com/pgvector/pgvector.git
cd pgvector

echo "Running make..."
make

echo "Running make install..."
make install

echo "Creating vector extension..."
psql --username=postgres -c "CREATE EXTENSION vector;"