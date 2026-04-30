# CNN Deepfake Detector

An interactive neural network visualization tool for deepfake detection using Convolutional Neural Networks (CNN). This project provides a comprehensive deepfake detection system with advanced visualization capabilities for understanding how the model makes predictions.

## ✨ Features

### Core Functionality
* **Binary Classification**: Real vs. fake face detection using CNN
* **Interactive Web Interface**: Modern Streamlit-based dashboard
* **Demo Datasets**: Pre-loaded real faces, deepfake frames, and mixed datasets
* **Batch Processing**: Process multiple images with frame selection

### Advanced Visualizations
* **Network Flow**: Visualize signal flow through the CNN architecture with input/preprocessed frame comparison
* **Feature Maps**: Interactive visualization of learned features in convolutional layers
* **Gradient Flow**: Analysis of gradient propagation during training
* **Weight Heatmaps**: Visual representation of learned weights across all layers
* **Training Curves**: Real-time accuracy and loss monitoring with detailed metrics
* **FC Layer Analysis**: Fully connected layer activation visualization and prediction breakdown
* **Model Architecture**: Complete layer-by-layer architecture inspection

---

## 📁 Project Structure

```
deepfake-detector/
├── app.py                  # Enhanced Streamlit dashboard with visualizations
├── model_train.py          # CNN training script with callbacks
├── test_backend.py         # Backend functionality test script
├── run_app.sh              # Quick launch script
├── deepfake_model.h5       # Trained model (generated after training)
├── training_log.csv        # Training metrics log (generated after training)
└── data/
    ├── real/               # Real face images dataset
    └── fake/               # Deepfake/synthetic face images dataset
```

## 🏗️ Model Architecture

```
Input (128x128x3)
    ↓
Conv2D (32 filters, 3x3, ReLU)
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3, ReLU)
    ↓
MaxPooling2D (2x2)
    ↓
Flatten
    ↓
Dense (128 units, ReLU)
    ↓
Dropout (50%)
    ↓
Dense (1 unit, Sigmoid)
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/dantusaikamal/Deepfake-detection.git
cd deepfake-detector
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate     # On Windows
# source venv/bin/activate  # On Linux/macOS
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

Or

```bash
pip install tensorflow opencv-python streamlit pillow pandas matplotlib
```

---

## Preparing the Dataset

Create a folder named `data/` in the root directory with the following structure:

```
data/
├── real/    # Add real face images here
└── fake/    # Add deepfake or synthetic face images here
```

You can use datasets such as:

* [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

Start with a small sample (e.g., 500–1000 images per class) to test.

---

## Training the Model

Run the training script:

```bash
python model_train.py
```

This will:

* Train the CNN model
* Save the best model as `deepfake_model.h5`
* Save training logs in `training_log.csv`

---

## 🚀 Running the Application

### Option 1: Quick Start (Recommended)

```bash
./run_app.sh
```

### Option 2: Manual Start

```bash
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate   # On Windows
streamlit run app.py
```

Visit `http://localhost:8501` in your browser to access the dashboard.

## 🧪 Testing Backend

Before running the app, you can test if all backend components are working:

```bash
source venv/bin/activate
python test_backend.py
```

This will verify:
- Model loading
- Data directory structure
- Image preprocessing
- Model predictions
- Feature map extraction
- Training log availability

## 📱 Using the Application

### 1. **Load Data**
   - Upload your own image (JPG, PNG, MP4, AVI)
   - Or use demo datasets:
     - **Real Faces**: 12 authentic face images
     - **Deepfake Frames**: 12 synthetic/manipulated faces
     - **Mixed Dataset**: 24 combined images

### 2. **Explore Visualizations**

#### Network Flow Tab
- View input frame and preprocessed tensor side-by-side
- See normalization parameters (μ and σ values)
- Inspect complete model architecture
- Color-coded layer type legend

#### Feature Maps Tab
- Select any convolutional layer
- Visualize learned features as heatmaps
- See what patterns the network detects

#### Weight Heatmaps Tab
- Explore learned weights in each layer
- Visualize convolutional filters
- Inspect dense layer weight matrices

#### Training Curves Tab
- View accuracy and loss over epochs
- Compare training vs validation metrics
- Analyze training statistics
- Review complete training history

#### FC Layer Tab
- Get real-time predictions (REAL/FAKE)
- See confidence scores and raw probabilities
- Visualize neuron activations in dense layers
- Understand decision-making process

#### Documentation Tab
- Complete project documentation
- Usage instructions
- Model architecture details

### 3. **Frame Selection**
   - Use the sidebar dropdown to navigate through loaded images
   - Each frame is analyzed independently
   - Switch between frames to compare predictions

---

## 🎨 Interface Features

- **Wide Layout**: Maximized screen space for visualizations
- **Multi-Tab Navigation**: Organized interface with 7 specialized tabs
- **Interactive Controls**: Real-time updates and selections
- **Responsive Design**: Works on different screen sizes
- **Professional Styling**: Clean, modern UI with color-coded elements

## 📊 Visualization Capabilities

### Color Legend
- 🔵 **Blue**: Input/Convolutional layers
- 🟢 **Green**: Activation functions
- 🟦 **Teal**: Pooling layers
- 🟠 **Orange**: Fully connected layers
- 🟣 **Pink**: Output layer

### Supported Visualizations
1. **Pre-processing Pipeline**: Input → Normalized Tensor
2. **Layer Activations**: Feature maps at each convolutional layer
3. **Weight Matrices**: Kernel/filter visualizations
4. **Training Metrics**: Accuracy, loss, validation curves
5. **Neuron Activations**: Dense layer activation patterns

## ⚙️ Technical Details

### Dependencies
- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web application framework
- **NumPy**: Numerical computations
- **Pillow**: Image processing
- **Matplotlib**: Plotting and visualizations
- **OpenCV**: Computer vision operations
- **Pandas**: Data analysis and CSV handling

### Model Training
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Callbacks**: 
  - Early Stopping (patience=3)
  - Model Checkpoint (save best model)
  - CSV Logger (training metrics)

### Image Preprocessing
- Resize to 128×128 pixels
- Normalize to [0, 1] range
- Optional ImageNet normalization parameters displayed
- RGB color space

## 🐛 Troubleshooting

### Model Not Found
```bash
# Train the model first
python model_train.py
```

### No Images in Dataset
```bash
# Check data directory structure
ls data/real/
ls data/fake/
# Add images to these directories
```

### Module Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate
# Reinstall dependencies
pip install -r requirements.txt
```

### Streamlit Port Already in Use
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

## 📝 Notes

* Ensure `deepfake_model.h5` exists before running the app
* The model expects face images; results may be unreliable for non-face inputs
* GPU acceleration is automatically used if available via TensorFlow
* Training logs are cached for performance
* Feature extraction happens in real-time for selected images

## 🎓 Educational Use

This project is ideal for:
- Understanding CNN architectures
- Learning about deepfake detection
- Visualizing neural network internals
- Final year engineering projects
- Machine learning education
- Computer vision research

## 🤝 Contributing

This is a final year EEE project demonstrating interactive neural network visualization for deepfake detection. Feel free to fork and extend with additional features.

---
