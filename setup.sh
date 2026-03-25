#!/bin/bash

echo "Setting up AI Voice Assistant Stack..."

# Create model directories
echo "Creating model directories..."
mkdir -p models/{llm,stt,tts,wakeword}

# Create assets directory
mkdir -p orchestrator/app/assets/confirmations

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "Please edit .env file with your configuration"
fi

# Make entrypoint executable
chmod +x orchestrator/entrypoint.sh

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download models to models/ directories"
echo "2. Edit .env file with your configuration"
echo "3. Run: docker-compose up --build"
echo ""
echo "Model download examples:"
echo "  LLM: Download GGUF model to models/llm/"
echo "  TTS: Download Piper voice to models/tts/"
echo "  STT & Wakeword: Auto-download on first run"
