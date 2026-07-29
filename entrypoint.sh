#!/bin/sh
set -e

varlock run -- uvicorn transcribo_backend.app:app --host 0.0.0.0 --port "${PORT:-8090}" --no-access-log
