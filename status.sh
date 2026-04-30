#!/bin/bash
# Project Status Checker

echo "═══════════════════════════════════════════════════════"
echo "  CNN DEEPFAKE DETECTOR - PROJECT STATUS"
echo "═══════════════════════════════════════════════════════"
echo ""

# Check Model
if [ -f "deepfake_model.h5" ]; then
    size=$(du -h deepfake_model.h5 | cut -f1)
    echo "✓ Model: deepfake_model.h5 ($size)"
else
    echo "✗ Model: Not found (run model_train.py)"
fi

# Check Training Log
if [ -f "training_log.csv" ]; then
    epochs=$(tail -n +2 training_log.csv | wc -l)
    echo "✓ Training Log: $epochs epochs recorded"
else
    echo "⚠ Training Log: Not found"
fi

# Check Data
if [ -d "data/real" ]; then
    real_count=$(ls data/real/*.{jpg,png} 2>/dev/null | wc -l)
    echo "✓ Real Images: $real_count files"
else
    echo "✗ Real Images: Directory not found"
fi

if [ -d "data/fake" ]; then
    fake_count=$(ls data/fake/*.{jpg,png} 2>/dev/null | wc -l)
    echo "✓ Fake Images: $fake_count files"
else
    echo "✗ Fake Images: Directory not found"
fi

# Check Virtual Environment
if [ -d "venv" ]; then
    echo "✓ Virtual Environment: Active"
else
    echo "✗ Virtual Environment: Not found"
fi

# Check App File
if [ -f "app.py" ]; then
    lines=$(wc -l < app.py)
    echo "✓ Application: app.py ($lines lines)"
else
    echo "✗ Application: Not found"
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  FEATURES AVAILABLE"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  📊 Network Flow Visualization"
echo "  🗺️  Feature Maps Extraction"
echo "  📈 Training Curves Display"
echo "  🎨 Weight Heatmaps"
echo "  🧠 FC Layer Analysis"
echo "  📁 Demo Dataset Loading (3 modes)"
echo "  🖼️  Frame-by-frame Navigation"
echo "  📝 Complete Documentation"
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  QUICK COMMANDS"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  Test Backend:    ./test_backend.py"
echo "  Run App:         ./run_app.sh"
echo "  Train Model:     venv/bin/python model_train.py"
echo "  Quick Start:     cat QUICKSTART.md"
echo ""
