#!/bin/sh
cd /home/postgres
git clone --branch v0.4.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
make install

psql --username=postgres -c "CREATE EXTENSION vector;"
