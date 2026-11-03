#!/bin/sh
set -eu
# Cloud Run sets PORT. Workers default to 1: the measured baseline is in docs/deploy.md.
exec gunicorn \
  --bind "0.0.0.0:${PORT:-8080}" \
  --workers "${SPROUT_WORKERS:-1}" \
  --threads "${SPROUT_THREADS:-4}" \
  --timeout "${SPROUT_TIMEOUT:-60}" \
  --chdir /app/deploy \
  wsgi:app
