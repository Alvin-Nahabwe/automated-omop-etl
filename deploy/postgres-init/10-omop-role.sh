#!/bin/bash
# Runs once, when the database volume is first created (see deploy/compose.yaml).
# Creates the application role and database: the app owns its database but is not a
# PostgreSQL superuser. pg_trgm (fast fuzzy concept search) is installed by the superuser.
set -euo pipefail
: "${OMOP_DB_PASSWORD:?set OMOP_DB_PASSWORD in .env}"
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres -v pw="$OMOP_DB_PASSWORD" <<'EOSQL'
CREATE ROLE omop LOGIN PASSWORD :'pw';
CREATE DATABASE omop OWNER omop;
\connect omop
CREATE EXTENSION IF NOT EXISTS pg_trgm;
EOSQL
