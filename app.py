# app.py
import streamlit as st

# Page Config - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="CNN Deepfake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import other libraries
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
import glob

# Constants
IMG_SIZE = 128

# Normalization parameters
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Load model with error handling
@st.cache_resource
def load_cnn_model():
    try:
        return load_model("deepfake_model.h5")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_cnn_model()

# Custom CSS for modern styling
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: #f5f7fa;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: white;
        padding: 1.5rem 2rem;
        border-bottom: 2px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    
    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1f2937;
        margin: 0;
    }
    
    .subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        margin-top: 0.25rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: white;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1.5rem;
        background-color: transparent;
        border-radius: 6px;
        color: #6b7280;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        font-weight: 600;
    }
    
    .stButton>button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Code block */
    .normalization-params {
        background: #1f2937;
        color: #e5e7eb;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        margin-top: 1rem;
    }
    
    /* Legend items */
    .legend-item {
        display: inline-flex;
        align-items: center;
        margin-right: 1.5rem;
        font-size: 0.9rem;
    }
    
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Helper Functions
def load_demo_images(category, num_frames=12):
    """Load demo images from data directory"""
    try:
        if category == "real":
            path = "data/real/"
        elif category == "fake":
            path = "data/fake/"
        else:  # mixed
            real_path = "data/real/"
            fake_path = "data/fake/"
            real_imgs = sorted(glob.glob(f"{real_path}*.jpg") + glob.glob(f"{real_path}*.png"))[:num_frames//2]
            fake_imgs = sorted(glob.glob(f"{fake_path}*.jpg") + glob.glob(f"{fake_path}*.png"))[:num_frames//2]
            return real_imgs + fake_imgs
        
        images = sorted(glob.glob(f"{path}*.jpg") + glob.glob(f"{path}*.png"))[:num_frames]
        return images
    except Exception as e:
        st.error(f"Error loading images: {e}")
        return []

def normalize_image(img_array):
    """Apply ImageNet normalization"""
    normalized = np.zeros_like(img_array)
    for i in range(3):
        normalized[:, :, i] = (img_array[:, :, i] - MEAN[i]) / STD[i]
    return normalized

def get_feature_maps(model, img_array, layer_idx):
    """Extract feature maps from a specific layer"""
    try:
        if model is None:
            return None
        layer_model = Model(inputs=model.input, outputs=model.layers[layer_idx].output)
        feature_maps = layer_model.predict(img_array, verbose=0)
        return feature_maps
    except Exception as e:
        st.error(f"Error extracting feature maps: {e}")
        return None

def visualize_feature_maps(feature_maps, num_maps=16):
    """Visualize feature maps"""
    try:
        if feature_maps is None or len(feature_maps.shape) != 4:
            return None
            
        num_maps = min(num_maps, feature_maps.shape[-1])
        cols = 4
        rows = (num_maps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i in range(num_maps):
            axes[i].imshow(feature_maps[0, :, :, i], cmap='viridis')
            axes[i].axis('off')
            axes[i].set_title(f'Map {i+1}', fontsize=10)
        
        for i in range(num_maps, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error visualizing feature maps: {e}")
        return None

@st.cache_data
def plot_training_curves():
    """Plot training curves from CSV log"""
    try:
        if not os.path.exists('training_log.csv'):
            return None
            
        df = pd.read_csv('training_log.csv')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        if 'accuracy' in df.columns:
            ax1.plot(df['epoch'], df['accuracy'], 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
        if 'val_accuracy' in df.columns:
            ax1.plot(df['epoch'], df['val_accuracy'], 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        if 'loss' in df.columns:
            ax2.plot(df['epoch'], df['loss'], 'b-o', label='Training Loss', linewidth=2, markersize=6)
        if 'val_loss' in df.columns:
            ax2.plot(df['epoch'], df['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error plotting training curves: {e}")
        return None

def perform_forward_propagation(img_array):
    """Perform forward propagation and return layer-by-layer outputs"""
    try:
        layer_outputs = []
        layer_names = []
        
        for i, layer in enumerate(model.layers):
            layer_model = Model(inputs=model.input, outputs=layer.output)
            output = layer_model.predict(img_array, verbose=0)
            layer_outputs.append(output)
            layer_names.append(f"{layer.name} ({layer.__class__.__name__})")
        
        return layer_outputs, layer_names
    except Exception as e:
        st.error(f"Error in forward propagation: {e}")
        return None, None

def visualize_forward_propagation(layer_outputs, layer_names):
    """Visualize the forward propagation through layers"""
    try:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, (output, name) in enumerate(zip(layer_outputs[:8], layer_names[:8])):
            ax = axes[i]
            
            if len(output.shape) == 4:  # Conv/Pool layers
                # Show first feature map
                ax.imshow(output[0, :, :, 0], cmap='viridis')
                ax.set_title(f"{name}\nShape: {output.shape[1:]}", fontsize=9)
            elif len(output.shape) == 2:  # Dense layers
                # Show as bar chart
                ax.bar(range(min(20, output.shape[1])), output[0, :20])
                ax.set_title(f"{name}\nShape: {output.shape[1:]}", fontsize=9)
            else:  # Flatten
                ax.text(0.5, 0.5, f"{name}\nShape: {output.shape[1:]}", 
                       ha='center', va='center', fontsize=10)
            
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error visualizing forward propagation: {e}")
        return None

def get_prediction_from_layers(img_array):
    """Get prediction with layer-by-layer inference details"""
    try:
        # Get final prediction
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        # Get intermediate layer outputs
        layer_outputs = []
        for layer in model.layers:
            layer_model = Model(inputs=model.input, outputs=layer.output)
            output = layer_model.predict(img_array, verbose=0)
            layer_outputs.append({
                'name': layer.name,
                'type': layer.__class__.__name__,
                'shape': output.shape,
                'output': output
            })
        
        return prediction, layer_outputs
    except Exception as e:
        st.error(f"Error getting prediction: {e}")
        return None, None

# Initialize session state
if 'selected_image' not in st.session_state:
    st.session_state.selected_image = None
if 'demo_images' not in st.session_state:
    st.session_state.demo_images = []
if 'current_frame_idx' not in st.session_state:
    st.session_state.current_frame_idx = 0
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'show_forward_prop' not in st.session_state:
    st.session_state.show_forward_prop = False
if 'show_backward_prop' not in st.session_state:
    st.session_state.show_backward_prop = False
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
if 'run_pressed' not in st.session_state:
    st.session_state.run_pressed = False
if 'previous_selected_image' not in st.session_state:
    st.session_state.previous_selected_image = None

# Check if model loaded successfully
if model is None:
    st.error("⚠️ Failed to load the deepfake detection model. Please ensure 'deepfake_model.h5' exists in the project directory.")
    st.stop()

# Header Section
st.markdown("""
    <div class='header-container'>
        <div class='main-title'>CNN DEEPFAKE DETECTOR</div>
        <div class='subtitle'>Final Year EEE Project — Interactive Neural Network Visualization</div>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 01 — INPUT")
    st.markdown("**Upload video or frames**")
    st.markdown("MP4, AVI, JPG, PNG")
    st.markdown("or use demo dataset below")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "mp4", "avi"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 02 — DEMO DATASET")
    
    if st.button("▶ LOAD REAL FACES (12 FRAMES)", use_container_width=True):
        st.session_state.demo_images = load_demo_images("real", 12)
        if st.session_state.demo_images:
            st.session_state.current_frame_idx = 0
            st.session_state.selected_image = st.session_state.demo_images[0]
            st.success(f"✓ Loaded {len(st.session_state.demo_images)} real faces")
        else:
            st.error("❌ No images found in data/real/")
    
    if st.button("▶ LOAD DEEPFAKE FRAMES (12 FRAMES)", use_container_width=True):
        st.session_state.demo_images = load_demo_images("fake", 12)
        if st.session_state.demo_images:
            st.session_state.current_frame_idx = 0
            st.session_state.selected_image = st.session_state.demo_images[0]
            st.success(f"✓ Loaded {len(st.session_state.demo_images)} deepfake frames")
        else:
            st.error("❌ No images found in data/fake/")
    
    if st.button("▶ LOAD MIXED DATASET (24 FRAMES)", use_container_width=True):
        st.session_state.demo_images = load_demo_images("mixed", 24)
        if st.session_state.demo_images:
            st.session_state.current_frame_idx = 0
            st.session_state.selected_image = st.session_state.demo_images[0]
            st.success(f"✓ Loaded {len(st.session_state.demo_images)} mixed frames")
        else:
            st.error("❌ No images found in data directories")
    
    # Train Model Button
    st.markdown("---")
    model_exists = os.path.exists("deepfake_model.h5")
    if model_exists:
        st.success("✓ Model already trained and available")
        if st.button("🔄 RETRAIN MODEL", use_container_width=True):
            st.session_state.training_in_progress = True
            with st.spinner("Retraining model... This may take several minutes"):
                try:
                    import subprocess
                    result = subprocess.run(
                        ["venv/bin/python", "model_train.py"],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    if result.returncode == 0:
                        st.success("✓ Model retraining completed successfully!")
                        st.rerun()
                    else:
                        st.error(f"Training failed: {result.stderr}")
                except subprocess.TimeoutExpired:
                    st.error("Training timeout. Please run model_train.py manually for longer training.")
                except Exception as e:
                    st.error(f"Error during training: {e}")
            st.session_state.training_in_progress = False
    else:
        if st.button("🎓 TRAIN MODEL", use_container_width=True, type="primary"):
            st.session_state.training_in_progress = True
            with st.spinner("Training model... This may take several minutes"):
                try:
                    import subprocess
                    result = subprocess.run(
                        ["venv/bin/python", "model_train.py"],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    if result.returncode == 0:
                        st.success("✓ Model training completed successfully!")
                        st.rerun()
                    else:
                        st.error(f"Training failed: {result.stderr}")
                except subprocess.TimeoutExpired:
                    st.error("Training timeout. Please run model_train.py manually for longer training.")
                except Exception as e:
                    st.error(f"Error during training: {e}")
            st.session_state.training_in_progress = False
    
    st.markdown("---")
    st.markdown("### 03 — FRAME SELECTION")
    
    if st.session_state.demo_images:
        frame_idx = st.selectbox(
            "Select frame",
            range(len(st.session_state.demo_images)),
            format_func=lambda x: f"Frame {x+1}/{len(st.session_state.demo_images)}"
        )
        st.session_state.current_frame_idx = frame_idx
        st.session_state.selected_image = st.session_state.demo_images[frame_idx]
        
        # Reset run_pressed when image selection changes
        if st.session_state.selected_image != st.session_state.previous_selected_image:
            st.session_state.run_pressed = False
            st.session_state.previous_selected_image = st.session_state.selected_image
    
    st.markdown("---")
    
    # RUN PREDICTION Button
    if st.session_state.selected_image:
        if st.button("🚀 RUN PREDICTION", use_container_width=True, type="primary"):
            st.session_state.run_pressed = True
            st.session_state.previous_selected_image = st.session_state.selected_image
    
    st.markdown("---")
    st.markdown("### 04 — PREDICTION OUTPUT")
    
    # Get prediction only if RUN button has been pressed
if st.session_state.selected_image:

    if st.session_state.run_pressed:
        try:
            # Load and preprocess image
            if isinstance(st.session_state.selected_image, str):
                temp_img = Image.open(st.session_state.selected_image)
            else:
                temp_img = Image.open(st.session_state.selected_image)

            img_resized = temp_img.resize((IMG_SIZE, IMG_SIZE))
            img_array = img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Get prediction
            prediction, layer_outputs = get_prediction_from_layers(img_array)

            if prediction is not None:
                st.session_state.prediction_result = {
                    'prediction': prediction,
                    'layer_outputs': layer_outputs
                }

                # Display result
                is_real = prediction >= 0.5
                confidence = prediction if is_real else (1 - prediction)

                if is_real:
                    st.success(f"✅ REAL — Confidence: {confidence*100:.1f}%")
                else:
                    st.error(f"⚠️ FAKE — Confidence: {confidence*100:.1f}%")

                st.progress(float(confidence))

                st.markdown(f"""
                    **Real:** {prediction:.4f} ({prediction*100:.2f}%)  
                    **Fake:** {1-prediction:.4f} ({(1-prediction)*100:.2f}%)
                """)

        except Exception as e:
            st.error(f"Error making prediction: {e}")

    else:
        st.info("Press '🚀 RUN PREDICTION' to analyze the selected image")

else:
    st.info("Load an image to see prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⏩ Forward", use_container_width=True):
            st.session_state.show_forward_prop = True
            st.session_state.show_backward_prop = False
    
    with col2:
        if st.button("⏪ Backward", use_container_width=True):
            st.session_state.show_backward_prop = True
            st.session_state.show_forward_prop = False
    
    st.markdown("---")
    st.markdown(f"**Epoch:** 0")
    st.markdown("**Status:** {'TRAINING' if st.session_state.training_in_progress else 'IDLE'}")

# Process image if available
if uploaded_file:
    st.session_state.selected_image = uploaded_file

# Main Content - Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "NETWORK FLOW", 
    "FEATURE MAPS", 
    "GRADIENT FLOW", 
    "WEIGHT HEATMAPS", 
    "TRAINING CURVES",
    "FC LAYER",
    "DOCUMENTATION"
])

# Tab 1: Network Flow
with tab1:
    st.markdown("### CNN ARCHITECTURE — SIGNAL FLOW VISUALIZATION")
    
    # Legend
    st.markdown("""
        <div style='margin-bottom: 1rem;'>
            <span class='legend-item'><span class='legend-dot' style='background: #3b82f6;'></span> Input / Conv</span>
            <span class='legend-item'><span class='legend-dot' style='background: #10b981;'></span> Activation</span>
            <span class='legend-item'><span class='legend-dot' style='background: #14b8a6;'></span> Pooling</span>
            <span class='legend-item'><span class='legend-dot' style='background: #f97316;'></span> FC Layers</span>
            <span class='legend-item'><span class='legend-dot' style='background: #ec4899;'></span> Output</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Load image if selected
    current_image = None
    if st.session_state.selected_image:
        try:
            if isinstance(st.session_state.selected_image, str):
                current_image = Image.open(st.session_state.selected_image)
            else:
                current_image = Image.open(st.session_state.selected_image)
        except Exception as e:
            st.error(f"Error loading image: {e}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### INPUT FRAME PREVIEW")
        
        if current_image:
            st.image(current_image, use_column_width=True)
        else:
            st.info("No frame selected")
    
    with col2:
        st.markdown("#### PREPROCESSED (NORMALIZED TENSOR)")
        
        if current_image:
            img = current_image.resize((IMG_SIZE, IMG_SIZE))
            img_array = img_to_array(img) / 255.0
            
            # Display normalized image
            st.image(img_array, use_column_width=True, clamp=True)
            
            # Show normalization parameters
            st.markdown(f"""
                <div class='normalization-params'>
                x_norm = (x - μ) / σ<br><br>
                μ = [{MEAN[0]:.3f}, {MEAN[1]:.3f}, {MEAN[2]:.3f}]<br>
                σ = [{STD[0]:.3f}, {STD[1]:.3f}, {STD[2]:.3f}]
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No frame selected")
    
    # Forward Propagation Visualization
    if st.session_state.show_forward_prop and current_image:
        st.markdown("---")
        st.markdown("#### 🔄 FORWARD PROPAGATION")
        
        with st.spinner("Performing forward propagation..."):
            img = current_image.resize((IMG_SIZE, IMG_SIZE))
            img_array = img_to_array(img) / 255.0
            img_array_batch = np.expand_dims(img_array, axis=0)
            
            layer_outputs, layer_names = perform_forward_propagation(img_array_batch)
            
            if layer_outputs:
                st.success("✓ Forward propagation complete!")
                
                # Visualize layer outputs
                fig = visualize_forward_propagation(layer_outputs, layer_names)
                if fig:
                    st.pyplot(fig)
                
                # Show layer details
                with st.expander("📊 Layer Output Details"):
                    for i, (output, name) in enumerate(zip(layer_outputs, layer_names)):
                        st.markdown(f"**Layer {i+1}: {name}**")
                        st.markdown(f"- Output shape: `{output.shape}`")
                        if len(output.shape) == 4:
                            st.markdown(f"- Feature maps: {output.shape[-1]}")
                        elif len(output.shape) == 2:
                            st.markdown(f"- Neurons: {output.shape[-1]}")
                        st.markdown("---")
    
    # Network Architecture Visualization
    st.markdown("---")
    st.markdown("#### NETWORK ARCHITECTURE")
    
    # Display model summary
    if st.checkbox("Show Model Architecture Details"):
        st.text(model.summary())

# Tab 2: Feature Maps
with tab2:
    st.markdown("### FEATURE MAPS VISUALIZATION")
    
    if st.session_state.selected_image:
        try:
            # Load image
            if isinstance(st.session_state.selected_image, str):
                temp_image = Image.open(st.session_state.selected_image)
            else:
                temp_image = Image.open(st.session_state.selected_image)
            
            img = temp_image.resize((IMG_SIZE, IMG_SIZE))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            layer_names = [layer.name for layer in model.layers if 'conv' in layer.name]
            
            if layer_names:
                selected_layer = st.selectbox("Select Convolutional Layer", layer_names)
                layer_idx = [i for i, layer in enumerate(model.layers) if layer.name == selected_layer][0]
                
                with st.spinner("Extracting feature maps..."):
                    feature_maps = get_feature_maps(model, img_array, layer_idx)
                    fig = visualize_feature_maps(feature_maps)
                    
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.warning("Could not generate feature map visualization")
            else:
                st.warning("No convolutional layers found in model")
        except Exception as e:
            st.error(f"Error processing image: {e}")
    else:
        st.info("Please select an image from the sidebar to visualize feature maps")

# Tab 3: Gradient Flow / Backward Propagation
with tab3:
    st.markdown("### GRADIENT FLOW / BACKWARD PROPAGATION")
    
    if st.session_state.show_backward_prop:
        st.markdown("#### ⏪ BACKWARD PROPAGATION SIMULATION")
        
        if st.session_state.selected_image:
            try:
                # Load and preprocess image
                if isinstance(st.session_state.selected_image, str):
                    bp_image = Image.open(st.session_state.selected_image)
                else:
                    bp_image = Image.open(st.session_state.selected_image)
                
                img = bp_image.resize((IMG_SIZE, IMG_SIZE))
                img_array = img_to_array(img) / 255.0
                img_array_batch = np.expand_dims(img_array, axis=0)
                
                with st.spinner("Computing gradients..."):
                    # Import tensorflow for gradient computation
                    import tensorflow as tf
                    
                    # Convert to tensor
                    img_tensor = tf.convert_to_tensor(img_array_batch, dtype=tf.float32)
                    
                    # Compute gradients
                    with tf.GradientTape() as tape:
                        tape.watch(img_tensor)
                        predictions = model(img_tensor)
                        loss = predictions[0][0]  # Use the prediction as loss
                    
                    # Get gradients
                    gradients = tape.gradient(loss, model.trainable_variables)
                    
                    st.success("✓ Backward propagation complete!")
                    
                    # Display gradient information
                    st.markdown("#### 📉 GRADIENT STATISTICS")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    total_params = sum([tf.size(g).numpy() for g in gradients if g is not None])
                    avg_gradient = np.mean([tf.reduce_mean(tf.abs(g)).numpy() for g in gradients if g is not None])
                    max_gradient = max([tf.reduce_max(tf.abs(g)).numpy() for g in gradients if g is not None])
                    
                    with col1:
                        st.metric("Total Parameters", f"{total_params:,}")
                    with col2:
                        st.metric("Avg Gradient", f"{avg_gradient:.6f}")
                    with col3:
                        st.metric("Max Gradient", f"{max_gradient:.6f}")
                    
                    # Visualize gradient flow
                    st.markdown("#### 📊 GRADIENT FLOW BY LAYER")
                    
                    layer_names = [layer.name for layer in model.layers if layer.trainable]
                    gradient_norms = []
                    
                    for grad, layer in zip(gradients, model.trainable_variables):
                        if grad is not None:
                            norm = tf.norm(grad).numpy()
                            gradient_norms.append(norm)
                    
                    if gradient_norms:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.bar(range(len(gradient_norms)), gradient_norms, color='#3b82f6', alpha=0.7)
                        ax.set_xlabel('Layer Index', fontsize=12)
                        ax.set_ylabel('Gradient Norm', fontsize=12)
                        ax.set_title('Gradient Flow Through Network Layers', fontsize=14, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Layer-by-layer gradient details
                    with st.expander("📋 Detailed Gradient Information"):
                        for i, (var, grad) in enumerate(zip(model.trainable_variables, gradients)):
                            if grad is not None:
                                st.markdown(f"**Layer {i+1}: {var.name}**")
                                st.markdown(f"- Shape: `{var.shape}`")
                                st.markdown(f"- Gradient norm: `{tf.norm(grad).numpy():.6f}`")
                                st.markdown(f"- Mean gradient: `{tf.reduce_mean(grad).numpy():.6f}`")
                                st.markdown("---")
            
            except Exception as e:
                st.error(f"Error in backward propagation: {e}")
        else:
            st.warning("Please select an image first")
    else:
        st.info("Click the '⏪ Backward' button in the sidebar to perform backward propagation and visualize gradient flow.")
        
        st.markdown("""
        ### About Backward Propagation
        
        **Gradient Flow Analysis:**
        - Shows how gradients propagate backward through the network
        - Helps identify vanishing/exploding gradient problems
        - Reveals which layers are learning effectively
        - Useful for debugging training issues
        
        **What to look for:**
        - **Uniform gradient flow**: Indicates healthy learning
        - **Vanishing gradients**: Very small values in early layers
        - **Exploding gradients**: Very large values
        - **Dead neurons**: Zero gradients
        """)

# Tab 4: Weight Heatmaps
with tab4:
    st.markdown("### WEIGHT HEATMAPS")
    
    # Get all layers with weights
    weight_layers = [layer for layer in model.layers if len(layer.get_weights()) > 0]
    
    if weight_layers:
        layer_names = [layer.name for layer in weight_layers]
        selected_layer = st.selectbox("Select Layer", layer_names)
        
        layer = model.get_layer(selected_layer)
        weights = layer.get_weights()[0]
        
        st.markdown(f"**Weight shape:** {weights.shape}")
        
        # Visualize weights
        if len(weights.shape) == 4:  # Conv layer
            num_filters = min(16, weights.shape[3])
            cols = 4
            rows = (num_filters + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
            axes = axes.flatten() if rows > 1 else [axes]
            
            for i in range(num_filters):
                w = weights[:, :, 0, i]  # First channel
                im = axes[i].imshow(w, cmap='coolwarm', aspect='auto')
                axes[i].set_title(f'Filter {i+1}')
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i])
            
            for i in range(num_filters, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        elif len(weights.shape) == 2:  # Dense layer
            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.imshow(weights, cmap='coolwarm', aspect='auto')
            ax.set_title(f'Weights: {selected_layer}')
            plt.colorbar(im)
            st.pyplot(fig)

# Tab 5: Training Curves
with tab5:
    st.markdown("### TRAINING CURVES")
    
    fig = plot_training_curves()
    if fig:
        st.pyplot(fig)
        
        # Display training statistics
        if os.path.exists('training_log.csv'):
            df = pd.read_csv('training_log.csv')
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Final Train Accuracy", f"{df['accuracy'].iloc[-1]:.4f}")
            with col2:
                st.metric("Final Val Accuracy", f"{df['val_accuracy'].iloc[-1]:.4f}" if 'val_accuracy' in df.columns else "N/A")
            with col3:
                st.metric("Final Train Loss", f"{df['loss'].iloc[-1]:.4f}")
            with col4:
                st.metric("Final Val Loss", f"{df['val_loss'].iloc[-1]:.4f}" if 'val_loss' in df.columns else "N/A")
            
            st.markdown("---")
            st.markdown("#### Training History")
            st.dataframe(df, use_container_width=True)
    else:
        st.warning("No training log found. Train the model first using model_train.py")

# Tab 6: FC Layer
with tab6:
    st.markdown("### FULLY CONNECTED LAYER ANALYSIS")
    
    if st.session_state.selected_image:
        try:
            # Load image
            if isinstance(st.session_state.selected_image, str):
                fc_image = Image.open(st.session_state.selected_image)
            else:
                fc_image = Image.open(st.session_state.selected_image)
            
            img = fc_image.resize((IMG_SIZE, IMG_SIZE))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Get prediction
            prediction = model.predict(img_array, verbose=0)[0][0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### PREDICTION OUTPUT")
                is_real = prediction >= 0.5
                label = "REAL" if is_real else "FAKE"
                confidence = prediction if is_real else (1 - prediction)
                
                st.markdown(f"### {label}")
                st.progress(float(confidence))
                st.metric("Confidence", f"{confidence * 100:.2f}%")
            
            with col2:
                st.markdown("#### RAW OUTPUT VALUES")
                st.metric("Real Probability", f"{prediction:.6f}")
                st.metric("Fake Probability", f"{1-prediction:.6f}")
                st.metric("Threshold", "0.5")
            
            # Activation visualization
            st.markdown("---")
            st.markdown("#### ACTIVATION VALUES")
            
            # Get dense layer activations
            dense_layers = [layer for layer in model.layers if 'dense' in layer.name]
            if dense_layers:
                layer_name = st.selectbox("Select Dense Layer", [l.name for l in dense_layers])
                layer_idx = [i for i, l in enumerate(model.layers) if l.name == layer_name][0]
                
                activations = get_feature_maps(model, img_array, layer_idx)
                
                if activations is not None:
                    activations_flat = activations[0]
                    
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.bar(range(len(activations_flat)), activations_flat)
                    ax.set_xlabel('Neuron Index', fontsize=12)
                    ax.set_ylabel('Activation Value', fontsize=12)
                    ax.set_title(f'Activation Values - {layer_name}', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"Error analyzing FC layer: {e}")
    else:
        st.info("Please select an image to analyze FC layer activations")

# Tab 7: Documentation
with tab7:
    st.markdown("### DOCUMENTATION")
    
    st.markdown("""
    ## CNN Deepfake Detector
    
    ### Project Overview
    This is an interactive neural network visualization tool for deepfake detection using Convolutional Neural Networks (CNN).
    
    ### Model Architecture
    - **Input Layer:** 128x128x3 RGB images
    - **Conv2D Layer 1:** 32 filters, 3x3 kernel, ReLU activation
    - **MaxPooling2D:** 2x2 pool size
    - **Conv2D Layer 2:** 64 filters, 3x3 kernel, ReLU activation
    - **MaxPooling2D:** 2x2 pool size
    - **Flatten Layer**
    - **Dense Layer:** 128 units, ReLU activation, 50% Dropout
    - **Output Layer:** 1 unit, Sigmoid activation (binary classification)
    
    ### Features
    1. **Network Flow:** Visualize the signal flow through the CNN architecture
    2. **Feature Maps:** View learned features at different convolutional layers
    3. **Gradient Flow:** Analyze gradient flow during training
    4. **Weight Heatmaps:** Visualize learned weights in different layers
    5. **Training Curves:** Monitor training progress with accuracy and loss plots
    6. **FC Layer:** Analyze fully connected layer activations and predictions
    
    ### Usage
    1. Upload an image or load demo dataset from sidebar
    2. Navigate through tabs to explore different visualizations
    3. Select frames to analyze individual predictions
    
    ### Dataset
    - **Real Faces:** Authentic face images
    - **Deepfake Frames:** Synthetically generated/manipulated faces
    - **Mixed Dataset:** Combination of real and fake images
    
    ### Training
    To train the model, run:
    ```bash
    python model_train.py
    ```
    
    ### Requirements
    - TensorFlow/Keras
    - Streamlit
    - NumPy
    - Pillow
    - Matplotlib
    - OpenCV
    - Pandas
    
    ### Author
    Final Year EEE Project
    """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #6b7280; padding: 1rem;'>Powered by TensorFlow & Streamlit</div>", unsafe_allow_html=True)
