#!/bin/bash
# KANCHAN-AI Demo Launch Script
# Run from the project root: bash scripts/run_demo.sh

set -e

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║           KANCHAN-AI — Demo Mode                 ║"
echo "║  Spurious Gold Intelligence System               ║"
echo "║  Canara Bank / SuRaksha Hackathon 2.0            ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# data/density_log.csv ships Benford-conformant seeds; if it's missing,
# the records monitor simply reports "insufficient data" until 30 real
# readings accumulate — do NOT regenerate synthetic data here.

# Copy .env.example to .env if not present
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "Created .env from .env.example (add API keys for LLM verdict generation)"
fi

# Build frontend if dist doesn't exist
if [ ! -d "frontend/dist" ]; then
  echo "Building frontend..."
  cd frontend && npm install && npm run build && cd ..
  echo "Frontend built."
fi

# Start backend
echo "Starting KANCHAN-AI server on http://localhost:8000"
echo "Press Ctrl+C to stop."
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
SERVER_PID=$!

sleep 2

# Open browser (macOS)
if command -v open &> /dev/null; then
  open http://localhost:8000
elif command -v xdg-open &> /dev/null; then
  xdg-open http://localhost:8000
fi

echo ""
echo "Demo ready at http://localhost:8000"
echo "Demo files for presentation: data/demo/"
echo ""
echo "Scenario 1 — Genuine Gold:      weight_dry=20.00, weight_sub=18.88, karat=22"
echo "Scenario 2 — Tungsten-Core Fake: weight_dry=50.00, weight_sub=47.41, karat=24"
echo ""

wait $SERVER_PID
