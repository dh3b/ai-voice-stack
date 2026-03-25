#!/bin/bash
set -e

echo "Generating confirmation audio files..."
python3 -c "
import asyncio
from app.config import Config
from app.pipeline import generate_confirmations

asyncio.run(generate_confirmations())
" || echo "Warning: Confirmation generation failed"

exec "$@"
