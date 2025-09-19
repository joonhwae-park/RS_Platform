#!/usr/bin/env bash
set -euo pipefail

: "${PORT:=8080}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/hf}"
export HF_HOME="${HF_HOME:-/tmp/hf}"

exec micromamba run -n p5 uvicorn app:app --host 0.0.0.0 --port "$PORT"
