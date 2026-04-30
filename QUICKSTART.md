# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Test Backend
```bash
source venv/bin/activate
python test_backend.py
```

Expected output:
```
✓ Model loaded successfully
✓ real directory exists with 50 images
✓ fake directory exists with 50 images
✓ All critical backend components are working!
```

### Step 2: Launch Application
```bash
./run_app.sh
```
Or manually:
```bash
source venv/bin/activate
streamlit run app.py
```

### Step 3: Use the Interface
1. Open browser at `http://localhost:8501`
2. Click **"▶ LOAD REAL FACES (12 FRAMES)"** in sidebar
3. Explore the tabs:
   - **NETWORK FLOW**: See input and preprocessed images
   - **FEATURE MAPS**: View learned features
   - **TRAINING CURVES**: Check training performance
   - **FC LAYER**: Get predictions with confidence

## 📋 Common Commands

### Run the app
```bash
./run_app.sh
```

### Test backend
```bash
venv/bin/python test_backend.py
```

### Train new model
```bash
venv/bin/python model_train.py
```

### Check data
```bash
ls data/real/ | wc -l   # Count real images
ls data/fake/ | wc -l   # Count fake images
```

## 🎯 Key Features to Try

1. **Load Demo Dataset**
   - Sidebar → "02 — DEMO DATASET" → Click any load button

2. **View Network Flow**
   - Network Flow tab → See input vs preprocessed image
   - Check normalization parameters

3. **Explore Feature Maps**
   - Feature Maps tab → Select convolutional layer
   - See what patterns the CNN learned

4. **Check Training**
   - Training Curves tab → View accuracy/loss graphs
   - See validation performance

5. **Get Predictions**
   - FC Layer tab → See REAL/FAKE prediction
   - Check confidence scores

## 🔧 Troubleshooting

### App won't start?
```bash
# Check if port is in use
lsof -i :8501

# Use different port
streamlit run app.py --server.port 8502
```

### Model not found?
```bash
# Check if model exists
ls -lh deepfake_model.h5

# If missing, train it
venv/bin/python model_train.py
```

### No images loading?
```bash
# Verify data directories
ls data/real/ | head
ls data/fake/ | head

# Should show .jpg or .png files
```

## 📊 Interface Overview

```
┌─────────────────────────────────────────────────┐
│  Sidebar                  │  Main Content       │
│  ┌──────────────────┐    │  ┌──────────────┐  │
│  │ 01 - INPUT       │    │  │ TABS:        │  │
│  │ Upload file      │    │  │ • Network    │  │
│  │                  │    │  │ • Features   │  │
│  │ 02 - DEMO        │    │  │ • Gradients  │  │
│  │ • Real faces     │    │  │ • Weights    │  │
│  │ • Deepfakes      │    │  │ • Training   │  │
│  │ • Mixed          │    │  │ • FC Layer   │  │
│  │                  │    │  │ • Docs       │  │
│  │ 03 - SELECTION   │    │  └──────────────┘  │
│  │ Frame: [1/12]    │    │                     │
│  └──────────────────┘    │  [Visualizations]   │
└─────────────────────────────────────────────────┘
```

## 🎓 Learning Path

1. **Start**: Load demo dataset
2. **Understand**: Check Network Flow tab to see preprocessing
3. **Explore**: View Feature Maps to see what CNN learns
4. **Analyze**: Check Training Curves for model performance
5. **Predict**: Use FC Layer for real-time predictions

## 💡 Pro Tips

- Use **Mixed Dataset** to compare real vs fake predictions
- Check **Weight Heatmaps** to understand filter patterns
- Monitor **Training Curves** to identify overfitting
- Compare **Feature Maps** across different layers
- Switch frames to see how predictions vary

## 📞 Need Help?

- Check `README.md` for detailed documentation
- Run `test_backend.py` to diagnose issues
- Ensure virtual environment is activated
- Verify all dependencies are installed
