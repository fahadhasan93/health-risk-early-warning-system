#!/bin/bash

echo "🚑 Starting Health Risk Early Warning System (HREWS)..."
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "hrews_env" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    echo "Run: python3 -m venv hrews_env && source hrews_env/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check if model exists
if [ ! -f "hrews_model.pkl" ]; then
    echo "⚠️  Model not found. Training model first..."
    source hrews_env/bin/activate
    python hrews_model.py
    echo "✅ Model trained successfully!"
fi

# Activate virtual environment and start app
echo "🚀 Launching Streamlit application..."
source hrews_env/bin/activate
streamlit run app.py --server.port 8501 --server.headless true

echo "✅ Application started! Open http://localhost:8501 in your browser"
