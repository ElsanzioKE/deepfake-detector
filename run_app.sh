#!/bin/bash
# Startup script for Deepfake Detector

echo "=========================================="
echo "CNN Deepfake Detector"
echo "=========================================="
echo ""

# Activate virtual environment
source venv/bin/activate

# Check if model exists
if [ ! -f "deepfake_model.h5" ]; then
    echo "⚠ Model file not found!"
    echo "Please train the model first: python model_train.py"
    exit 1
fi

# Check if data directory exists
if [ ! -d "data/real" ] || [ ! -d "data/fake" ]; then
    echo "⚠ Data directories not found!"
    echo "Please ensure data/real/ and data/fake/ directories exist with images"
    exit 1
fi

echo "✓ Environment ready"
echo "✓ Model loaded"
echo "✓ Starting Streamlit app..."
echo ""
echo "The app will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

# Run Streamlit
streamlit run app.py
