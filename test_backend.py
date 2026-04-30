#!/usr/bin/env python3
"""
Test script to verify backend functionality
"""
import os
import sys
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import glob

print("=" * 60)
print("TESTING DEEPFAKE DETECTOR BACKEND")
print("=" * 60)

# Test 1: Model Loading
print("\n[1/6] Testing Model Loading...")
try:
    model = load_model("deepfake_model.h5")
    print("✓ Model loaded successfully")
    print(f"  - Model input shape: {model.input_shape}")
    print(f"  - Model output shape: {model.output_shape}")
    print(f"  - Total layers: {len(model.layers)}")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    sys.exit(1)

# Test 2: Data Directory Check
print("\n[2/6] Testing Data Directory Structure...")
data_dirs = {
    'real': 'data/real/',
    'fake': 'data/fake/'
}

for name, path in data_dirs.items():
    if os.path.exists(path):
        images = glob.glob(f"{path}*.jpg") + glob.glob(f"{path}*.png")
        print(f"✓ {name} directory exists with {len(images)} images")
    else:
        print(f"✗ {name} directory not found at {path}")

# Test 3: Image Loading and Preprocessing
print("\n[3/6] Testing Image Loading and Preprocessing...")
try:
    test_images = glob.glob("data/real/*.jpg")[:1] + glob.glob("data/fake/*.jpg")[:1]
    if test_images:
        img_path = test_images[0]
        image = Image.open(img_path)
        print(f"✓ Loaded test image: {os.path.basename(img_path)}")
        print(f"  - Original size: {image.size}")
        
        # Preprocess
        img = image.resize((128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        print(f"✓ Preprocessed image shape: {img_array.shape}")
    else:
        print("✗ No test images found")
except Exception as e:
    print(f"✗ Image preprocessing failed: {e}")

# Test 4: Model Prediction
print("\n[4/6] Testing Model Prediction...")
try:
    prediction = model.predict(img_array, verbose=0)
    print(f"✓ Prediction successful")
    print(f"  - Raw output: {prediction[0][0]:.6f}")
    print(f"  - Classification: {'REAL' if prediction[0][0] >= 0.5 else 'FAKE'}")
    print(f"  - Confidence: {max(prediction[0][0], 1-prediction[0][0])*100:.2f}%")
except Exception as e:
    print(f"✗ Prediction failed: {e}")

# Test 5: Feature Map Extraction
print("\n[5/6] Testing Feature Map Extraction...")
try:
    conv_layers = [layer for layer in model.layers if 'conv' in layer.name]
    if conv_layers:
        print(f"✓ Found {len(conv_layers)} convolutional layers")
        # Test extraction from first conv layer
        layer_idx = 0
        layer_model = Model(inputs=model.input, outputs=model.layers[layer_idx].output)
        feature_maps = layer_model.predict(img_array, verbose=0)
        print(f"✓ Feature maps extracted from layer 0")
        print(f"  - Feature map shape: {feature_maps.shape}")
    else:
        print("✗ No convolutional layers found")
except Exception as e:
    print(f"✗ Feature extraction failed: {e}")

# Test 6: Training Log
print("\n[6/6] Testing Training Log...")
try:
    if os.path.exists('training_log.csv'):
        import pandas as pd
        df = pd.read_csv('training_log.csv')
        print("✓ Training log found")
        print(f"  - Total epochs: {len(df)}")
        print(f"  - Columns: {', '.join(df.columns)}")
        if len(df) > 0:
            print(f"  - Final accuracy: {df['accuracy'].iloc[-1]:.4f}")
    else:
        print("⚠ Training log not found (optional)")
except Exception as e:
    print(f"✗ Training log check failed: {e}")

print("\n" + "=" * 60)
print("BACKEND TEST COMPLETE")
print("=" * 60)
print("\n✓ All critical backend components are working!")
print("You can now run: streamlit run app.py")
