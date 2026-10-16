#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -f data/METABRIC_RNA_Mutation.csv ]]; then
  echo "Missing data/METABRIC_RNA_Mutation.csv; download from Kaggle." >&2
  exit 1
fi
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/analyze.py --input data/METABRIC_RNA_Mutation.csv \
  --output results/figures --bootstrap 40 --seed 0
