#!/bin/bash
set -eo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export CI=true
export MPLBACKEND=Agg

cd /workspace/yolov5

rm -rf .pytest_cache

pytest tests/ -v --tb=short --no-cov -p no:cacheprovider

