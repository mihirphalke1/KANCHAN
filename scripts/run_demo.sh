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

# Seed demo data if density_log.csv doesn't exist
if [ ! -f "data/density_log.csv" ]; then
  echo "Seeding demo data..."
  python scripts/seed_demo_data.py
fi

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
echo "Scenario 1 — Genuine Gold:      weight_dry=15.2, weight_sub=12.8, karat=22"
echo "Scenario 2 — Tungsten-Core Fake: weight_dry=18.9, weight_sub=15.9, karat=24"
echo ""

wait $SERVER_PID
