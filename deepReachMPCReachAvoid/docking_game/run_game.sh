#!/bin/bash
# Run the 13D Spacecraft Docking Game

cd "$(dirname "$0")"

# Check if virtual environment exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

python game.py
