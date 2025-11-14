import io
import os
from typing import Dict, Any, Optional
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Add the project root to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.audio.feature_extraction import (
        load_audio_file,
        compute_mel_spectrogram,
        extract_frame_features,
        build_feature_matrix,
    )
    from src.plots.plots import plot_mel_spectrogram
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure you're running from the project root directory")
    st.stop()


class MIMIIPredictor:
    """Predictor class for using trained MIMII models."""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.is_loaded = False
    
    def load_models(self, model_dir: str) -> bool:
        """Load trained models and preprocessing objects."""
        try:
            model_path = Path(model_dir)
            
            if not model_path.exists():
                st.warning(f"Model directory not found: {model_path}")
                st.info("💡 Run train_model.py first to create models")
                return False
            
            scaler_path = model_path / "feature_scaler.pkl"
            if not scaler_path.exists():
                st.warning(f"Feature scaler not found: {scaler_path}")
                st.info("Expected files: feature_scaler.pkl, best_model.pkl, model_metadata.pkl")
                return False
            
            self.scaler = joblib.load(scaler_path)
            
            metadata_path = model_path / "model_metadata.pkl"
            if metadata_path.exists():
                metadata = joblib.load(metadata_path)
                st.success(f"Models trained: {metadata.get('timestamp', 'unknown')}")
            
            self.models = {}
            skip_files = ["feature_scaler.pkl", "model_metadata.pkl"]
            
            for model_file in model_path.glob("*.pkl"):
                if model_file.name not in skip_files:
                    model_name = model_file.stem.replace('_', ' ').title()
                    
                    try:
                        model = joblib.load(model_file)
                        self.models[model_name] = model
                    except Exception as e:
                        st.warning(f"Could not load {model_file.name}: {e}")
            
            self.is_loaded = len(self.models) > 0
            
            if self.is_loaded:
                st.success(f"✓ Loaded {len(self.models)} model(s)")
            else:
                st.error("No models found in directory")
            
            return self.is_loaded
            
        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.code(traceback.format_exc())
            return False

    def extract_features_from_audio(self, y: np.ndarray, sr: int, **kwargs) -> Optional[np.ndarray]:
        """Extract features from audio data (matches training pipeline)."""
        try:
            features_dict = extract_frame_features(
                y, sr,
                frame_length=kwargs.get('frame_length', 2048),
                hop_length=kwargs.get('hop_length', 512),
                n_mfcc=kwargs.get('n_mfcc', 13)
            )
            
            feature_matrix, feature_names, times = build_feature_matrix(features_dict)
            
            features_mean = np.mean(feature_matrix, axis=0)
            features_std = np.std(feature_matrix, axis=0)
            features_max = np.max(feature_matrix, axis=0)
            features_min = np.min(feature_matrix, axis=0)
            features_median = np.median(feature_matrix, axis=0)
            
            combined_features = np.concatenate([
                features_mean, features_std, features_max, features_min, features_median
            ])
            
            return combined_features.reshape(1, -1)
            
        except Exception as e:
            st.error(f"Error extracting features: {e}")
            st.code(traceback.format_exc())
            return None

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Make predictions using loaded models."""
        if not self.is_loaded or self.scaler is None:
            return {}
        
        results = {}
        
        try:
            features_scaled = self.scaler.transform(features)
            
            for model_name, model in self.models.items():
                try:
                    prediction = model.predict(features_scaled)[0]
                    
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(features_scaled)[0]
                        prob_normal = probabilities[0]
                        prob_abnormal = probabilities[1]
                        confidence = max(prob_normal, prob_abnormal)
                        
                        results[model_name] = {
                            'prediction': 'normal' if prediction == 0 else 'abnormal',
                            'confidence': confidence,
                            'probabilities': {
                                'normal': prob_normal,
                                'abnormal': prob_abnormal
                            },
                            'raw_prediction': int(prediction)
                        }
                    else:
                        results[model_name] = {
                            'prediction': 'normal' if prediction == 0 else 'abnormal',
                            'confidence': None,
                            'probabilities': None,
                            'raw_prediction': int(prediction)
                        }
                
                except Exception as e:
                    st.warning(f"Error predicting with {model_name}: {e}")
            
            return results
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.code(traceback.format_exc())
            return {}


@st.cache_resource
def load_predictor():
    return MIMIIPredictor()


# Custom CSS for dark minimal design
st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Dark background */
    .main {
        background: #0a0a0a;
        color: #e5e5e5;
    }
    
    /* Clean container */
    .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Dark sidebar */
    section[data-testid="stSidebar"] {
        background: #111111;
        border-right: 1px solid #1f1f1f;
    }
    
    section[data-testid="stSidebar"] > div {
        padding: 2rem 1.5rem;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #a3a3a3;
    }
    
    /* Hide defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Typography */
    h1 {
        font-weight: 700;
        font-size: 2.25rem !important;
        letter-spacing: -0.025em;
        color: #ffffff !important;
        margin-bottom: 0.5rem !important;
    }
    
    h2 {
        font-weight: 600;
        font-size: 1.5rem !important;
        color: #f5f5f5 !important;
        margin-top: 3rem !important;
        margin-bottom: 1.5rem !important;
    }
    
    h3 {
        font-weight: 600;
        font-size: 1.125rem !important;
        color: #e5e5e5 !important;
    }
    
    p {
        color: #a3a3a3;
        line-height: 1.6;
    }
    
    /* Status cards - minimal dark */
    .status-card {
        padding: 2rem;
        border-radius: 12px;
        margin: 2rem 0;
        border: 1px solid;
        background: #141414;
    }
    
    .status-normal {
        border-color: #22c55e;
        background: linear-gradient(135deg, #0a1f0f 0%, #141414 100%);
    }
    
    .status-abnormal {
        border-color: #ef4444;
        background: linear-gradient(135deg, #1f0a0a 0%, #141414 100%);
    }
    
    .status-uncertain {
        border-color: #f59e0b;
        background: linear-gradient(135deg, #1f1508 0%, #141414 100%);
    }
    
    /* Metrics - dark theme */
    [data-testid="stMetricValue"] {
        font-size: 1.875rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem !important;
        color: #737373 !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Buttons - minimal dark */
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.625rem 1.5rem;
        border: 1px solid #262626;
        background: #171717;
        color: #e5e5e5;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background: #1f1f1f;
        border-color: #404040;
    }
    
    .stButton button[kind="primary"] {
        background: #2563eb;
        border-color: #2563eb;
        color: #ffffff;
    }
    
    .stButton button[kind="primary"]:hover {
        background: #1d4ed8;
        border-color: #1d4ed8;
    }
    
    /* File uploader - dark */
    [data-testid="stFileUploader"] {
        border: 1px dashed #262626;
        border-radius: 12px;
        padding: 2rem;
        background: #0f0f0f;
    }
    
    [data-testid="stFileUploader"] label {
        color: #e5e5e5 !important;
    }
    
    [data-testid="stFileUploader"] section {
        border-color: #262626 !important;
    }
    
    [data-testid="stFileUploader"] section > div {
        color: #737373;
    }
    
    /* Inputs - dark */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: #171717;
        border: 1px solid #262626;
        border-radius: 8px;
        color: #e5e5e5;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #404040;
        box-shadow: 0 0 0 1px #404040;
    }
    
    .stTextInput label,
    .stNumberInput label {
        color: #a3a3a3 !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }
    
    /* Expander - dark */
    .streamlit-expanderHeader {
        background: #141414;
        border: 1px solid #1f1f1f;
        border-radius: 8px;
        color: #e5e5e5 !important;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        background: #171717;
    }
    
    .streamlit-expanderContent {
        background: #0f0f0f;
        border: 1px solid #1f1f1f;
        border-top: none;
    }
    
    /* Progress bar - minimal */
    .stProgress > div > div > div {
        background: #2563eb;
    }
    
    .stProgress > div > div {
        background: #1f1f1f;
    }
    
    /* Info/warning boxes - dark */
    .stAlert {
        background: #141414;
        border: 1px solid #262626;
        color: #e5e5e5;
    }
    
    [data-baseweb="notification"] {
        background: #141414;
        border: 1px solid #262626;
    }
    
    /* Success/Error/Warning - dark variants */
    .stSuccess {
        background: #0a1f0f !important;
        border-color: #166534 !important;
    }
    
    .stError {
        background: #1f0a0a !important;
        border-color: #991b1b !important;
    }
    
    .stWarning {
        background: #1f1508 !important;
        border-color: #92400e !important;
    }
    
    .stInfo {
        background: #0a1220 !important;
        border-color: #1e40af !important;
    }
    
    /* Dataframe - dark */
    [data-testid="stDataFrame"] {
        background: #0f0f0f;
    }
    
    /* Audio player - dark */
    audio {
        filter: invert(1) hue-rotate(180deg);
    }
    
    /* Divider */
    hr {
        margin: 2.5rem 0;
        border: none;
        height: 1px;
        background: #1f1f1f;
    }
    
    /* Caption text */
    .caption, [data-testid="stCaptionContainer"] {
        color: #737373 !important;
        font-size: 0.875rem;
    }
    
    /* Code blocks */
    code {
        background: #171717;
        border: 1px solid #262626;
        color: #e5e5e5;
        padding: 0.125rem 0.375rem;
        border-radius: 4px;
    }
    
    pre {
        background: #0f0f0f;
        border: 1px solid #1f1f1f;
        border-radius: 8px;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0a0a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #262626;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #404040;
    }
</style>
""", unsafe_allow_html=True)

# App Configuration
st.set_page_config(
    page_title="Anomaly Detector",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("Anomaly Detector")
st.caption("Machine condition monitoring through audio analysis")
st.markdown("<br>", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("## Settings")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model Settings
    st.markdown("**Model Configuration**")
    model_dir = st.text_input(
        "Directory", 
        value="models",
        placeholder="models"
    )
    
    if st.button("Load Models", type="primary", use_container_width=True):
        predictor = load_predictor()
        with st.spinner("Loading..."):
            models_loaded = predictor.load_models(model_dir)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Audio Processing Settings
    st.markdown("**Audio Processing**")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        target_sr = st.number_input(
            "Sample Rate", 
            min_value=8000, 
            max_value=48000, 
            value=16000, 
            step=1000
        )
        hop_length = st.number_input(
            "Hop Length", 
            min_value=128, 
            max_value=2048, 
            value=512, 
            step=64
        )
        n_mfcc = st.number_input(
            "MFCCs", 
            min_value=8, 
            max_value=40, 
            value=13, 
            step=1
        )
    
    with col2:
        n_fft = st.number_input(
            "FFT Size", 
            min_value=512, 
            max_value=4096, 
            value=2048, 
            step=256
        )
        n_mels = st.number_input(
            "Mel Bands", 
            min_value=32, 
            max_value=256, 
            value=128, 
            step=16
        )

# Initialize predictor
predictor = load_predictor()

# Check if models are loaded
if not predictor.is_loaded:
    st.info("Load your trained models to get started")
    

    st.stop()

# File Upload Section
st.markdown("<br>", unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Upload audio file (.wav)", 
    type=["wav"],
    help="Upload machine audio for analysis"
)
st.markdown("<br>", unsafe_allow_html=True)

if uploaded is not None:
    try:
        # Progress tracking
        progress_bar = st.progress(0, "Initializing...")
        
        # Load audio
        progress_bar.progress(20, "Loading audio file...")
        y, sr = load_audio_file(uploaded, target_sr=target_sr)
        
        # Audio info
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.audio(uploaded, format="audio/wav")
        with col2:
            st.metric("Sample Rate", f"{sr}Hz")
        with col3:
            st.metric("Duration", f"{len(y)/sr:.1f}s")
        with col4:
            st.metric("Samples", f"{len(y):,}")
        
        # Feature extraction
        progress_bar.progress(40, "Extracting audio features...")
        
        features_dict = extract_frame_features(
            y, sr, 
            frame_length=n_fft, 
            hop_length=hop_length, 
            n_mfcc=int(n_mfcc)
        )
        feature_matrix, feature_names, times = build_feature_matrix(features_dict)
        
        # Compute mel spectrogram
        progress_bar.progress(60, "Computing mel-spectrogram...")
        
        S_db, mel_times = compute_mel_spectrogram(
            y, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        
        # Extract features for prediction
        progress_bar.progress(70, "Preparing features for models...")
        
        prediction_features = predictor.extract_features_from_audio(
            y, sr, 
            frame_length=n_fft, 
            hop_length=hop_length, 
            n_mfcc=n_mfcc
        )
        
        # Make predictions
        progress_bar.progress(85, "Running AI models...")
        
        if prediction_features is not None:
            predictions = predictor.predict(prediction_features)
        else:
            st.error("Failed to extract features for prediction")
            st.stop()
        
        progress_bar.progress(100, "Analysis complete!")
        progress_bar.empty()
        
        # Results Section
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("## Results")
        st.markdown("<br>", unsafe_allow_html=True)
        
        if predictions:
            # Calculate consensus
            predictions_list = [pred['prediction'] for pred in predictions.values()]
            abnormal_count = predictions_list.count('abnormal')
            normal_count = predictions_list.count('normal')
            total_models = len(predictions_list)
            
            # Overall status display
            if abnormal_count > normal_count:
                st.markdown(f"""
                <div class="status-card status-abnormal">
                    <h2 style="color: #ef4444; margin: 0; font-weight: 700; font-size: 1.75rem;">⚠ Anomaly Detected</h2>
                    <p style="font-size: 1rem; margin-top: 0.75rem; margin-bottom: 0; color: #a3a3a3;">
                        {abnormal_count} of {total_models} models detected abnormal behavior
                    </p>
                </div>
                """, unsafe_allow_html=True)
            elif normal_count > abnormal_count:
                st.markdown(f"""
                <div class="status-card status-normal">
                    <h2 style="color: #22c55e; margin: 0; font-weight: 700; font-size: 1.75rem;">✓ Normal Operation</h2>
                    <p style="font-size: 1rem; margin-top: 0.75rem; margin-bottom: 0; color: #a3a3a3;">
                        {normal_count} of {total_models} models indicate normal operation
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-card status-uncertain">
                    <h2 style="color: #f59e0b; margin: 0; font-weight: 700; font-size: 1.75rem;">⚡ Uncertain</h2>
                    <p style="font-size: 1rem; margin-top: 0.75rem; margin-bottom: 0; color: #a3a3a3;">
                        Models split evenly — manual inspection recommended
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Individual model predictions - clean grid
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Model Predictions**")
            st.markdown("<br>", unsafe_allow_html=True)
            
            cols = st.columns(3)
            
            for i, (model_name, pred_data) in enumerate(predictions.items()):
                with cols[i % 3]:
                    prediction = pred_data['prediction']
                    confidence = pred_data.get('confidence')
                    
                    status_emoji = "⚠" if prediction == 'abnormal' else "✓"
                    border_color = "#ef4444" if prediction == 'abnormal' else "#22c55e"
                    text_color = "#ef4444" if prediction == 'abnormal' else "#22c55e"
                    
                    confidence_text = f"{confidence*100:.0f}%" if confidence else "—"
                    
                    st.markdown(f"""
                    <div style="background: #141414; padding: 1.5rem; border-radius: 10px; 
                                border: 1px solid {border_color}; margin-bottom: 1rem; height: 140px;
                                display: flex; flex-direction: column; justify-content: space-between;">
                        <div>
                            <div style="font-size: 0.75rem; color: #737373; font-weight: 600; 
                                        margin-bottom: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;">
                                {model_name}
                            </div>
                            <div style="color: {text_color}; font-size: 1.25rem; font-weight: 700; margin-bottom: 0.5rem;">
                                {status_emoji} {prediction.upper()}
                            </div>
                        </div>
                        <div style="font-size: 0.875rem; color: #737373;">
                            Confidence: <span style="color: #e5e5e5; font-weight: 600;">{confidence_text}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("No predictions available")
        
        # Visualizations
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("## Frequency Analysis")
        st.markdown("<br>", unsafe_allow_html=True)
        
        fig_spec = plot_mel_spectrogram(S_db, sr, hop_length, title="Mel-Spectrogram")
        st.pyplot(fig_spec, clear_figure=True, use_container_width=True)
        
        # Feature Data Export
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Export Feature Data"):
            df = pd.DataFrame(feature_matrix, columns=feature_names)
            df.insert(0, "time_s", times)
            
            st.dataframe(df.head(100), use_container_width=True, height=300)
            
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV", 
                data=csv, 
                file_name="audio_features.csv", 
                mime="text/csv",
                use_container_width=True
            )
        
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        with st.expander("View Error Details"):
            st.code(traceback.format_exc())

else:
    st.info("Upload a .wav file to analyze")