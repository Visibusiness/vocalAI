#!/bin/bash

# Start Ollama server
ollama serve &

sleep 5

# Create model if it doesn't exist
if ! ollama list | grep -q "visi-ro"; then
    echo "Creating visi-ro model..."
    ollama create visi-ro -f /app/Modelfile
fi

echo "Starting FastAPI..."
uvicorn main:app --host 0.0.0.0 --port 8000