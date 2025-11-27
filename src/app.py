"""
Streamlit Web UI for Urban Sound Classification
Provides interface for prediction, visualization, upload, and retraining
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import os
from datetime import datetime
import time
from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Urban Sound Classification",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoints
PREDICTION_API = os.getenv("PREDICTION_API", "http://localhost:8000")
RETRAIN_API = os.getenv("RETRAIN_API", "http://localhost:8001")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health(api_url):
    """Check if API is healthy"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def load_audio_file(file):
    """Load audio file and return audio data"""
    try:
        audio, sr = librosa.load(file, sr=22050, duration=4.0)
        return audio, sr
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return None, None


def plot_waveform(audio, sr):
    """Plot audio waveform"""
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    return fig


def plot_spectrogram(audio, sr):
    """Plot mel-spectrogram"""
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Mel-Spectrogram')
    return fig


def main():
    """Main application"""
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/1f77b4/ffffff?text=Urban+Sound+ML", use_container_width=True)
        st.title("Navigation")
        page = st.radio(
            "Select Page",
            ["🏠 Home", "🎯 Prediction", "📊 Visualizations", "📤 Upload & Retrain", "📈 Model Monitoring"]
        )
        
        st.divider()
        
        # API Status
        st.subheader("API Status")
        pred_status = check_api_health(PREDICTION_API)
        retrain_status = check_api_health(RETRAIN_API)
        
        st.write("Prediction API:", "✅ Online" if pred_status else "❌ Offline")
        st.write("Retrain API:", "✅ Online" if retrain_status else "❌ Offline")
    
    # Main content
    if page == "🏠 Home":
        show_home()
    elif page == "🎯 Prediction":
        show_prediction()
    elif page == "📊 Visualizations":
        show_visualizations()
    elif page == "📤 Upload & Retrain":
        show_upload_retrain()
    elif page == "📈 Model Monitoring":
        show_monitoring()


def show_home():
    """Home page"""
    st.markdown('<h1 class="main-header">🔊 Urban Sound Classification System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Welcome to the Urban Sound Classification Platform
    
    This system uses deep learning to classify urban sounds for smart city safety applications.
    
    ### 🎯 Features:
    - **Real-time Prediction**: Upload audio files and get instant classifications
    - **Interactive Visualizations**: Explore dataset insights and model performance
    - **Batch Processing**: Upload multiple files for bulk predictions
    - **Model Retraining**: Upload new data and retrain the model
    - **Performance Monitoring**: Track model uptime, latency, and predictions
    
    ### 🚀 Use Cases:
    - Emergency response (sirens, gunshots detection)
    - Traffic management (car horns, engine sounds)
    - Public safety monitoring (dog barks, street music)
    - Environmental noise analysis
    
    ### 📊 Model Information:
    """)
    
    try:
        response = requests.get(f"{PREDICTION_API}/classes")
        if response.status_code == 200:
            data = response.json()
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Number of Classes", data['num_classes'])
            
            with col2:
                st.metric("Model Status", "Active ✅")
            
            st.subheader("Available Sound Classes:")
            classes = data['classes']
            cols = st.columns(3)
            for i, cls in enumerate(classes):
                with cols[i % 3]:
                    st.info(f"🔹 {cls}")
    except:
        st.warning("Unable to fetch model information. Please check API connection.")
    
    st.divider()
    st.markdown("""
    ### 🎓 Getting Started:
    1. Navigate to **Prediction** to classify audio files
    2. Check **Visualizations** for dataset insights
    3. Use **Upload & Retrain** to improve the model with new data
    4. Monitor performance in **Model Monitoring**
    """)


def show_prediction():
    """Prediction page"""
    st.title("🎯 Audio Classification Prediction")
    
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
    
    with tab1:
        st.subheader("Upload Audio File for Classification")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file (WAV format recommended)",
            type=['wav', 'mp3', 'ogg', 'flac'],
            key="single"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.audio(uploaded_file, format='audio/wav')
                
                # Load and visualize
                audio, sr = load_audio_file(uploaded_file)
                if audio is not None:
                    st.pyplot(plot_waveform(audio, sr))
            
            with col2:
                if audio is not None:
                    st.pyplot(plot_spectrogram(audio, sr))
            
            if st.button("🔍 Classify", type="primary", use_container_width=True):
                with st.spinner("Classifying..."):
                    try:
                        # Reset file pointer
                        uploaded_file.seek(0)
                        
                        files = {'file': (uploaded_file.name, uploaded_file, 'audio/wav')}
                        response = requests.post(f"{PREDICTION_API}/predict", files=files)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            st.success("Classification Complete!")
                            
                            # Display results
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Predicted Class", result['predicted_class'])
                            
                            with col2:
                                st.metric("Confidence", f"{result['confidence']*100:.2f}%")
                            
                            with col3:
                                st.metric("Timestamp", result['timestamp'].split('T')[1][:8])
                            
                            # Probability distribution
                            st.subheader("Probability Distribution")
                            probs_df = pd.DataFrame([
                                {"Class": k, "Probability": v*100}
                                for k, v in result['all_probabilities'].items()
                            ]).sort_values('Probability', ascending=False)
                            
                            fig = px.bar(probs_df, x='Class', y='Probability',
                                       title='Class Probabilities',
                                       color='Probability',
                                       color_continuous_scale='viridis')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error(f"Prediction failed: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    with tab2:
        st.subheader("Batch Prediction")
        
        uploaded_files = st.file_uploader(
            "Upload multiple audio files",
            type=['wav', 'mp3', 'ogg', 'flac'],
            accept_multiple_files=True,
            key="batch"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} files selected")
            
            if st.button("🔍 Classify All", type="primary", use_container_width=True):
                with st.spinner(f"Classifying {len(uploaded_files)} files..."):
                    try:
                        files = [('files', (f.name, f, 'audio/wav')) for f in uploaded_files]
                        response = requests.post(f"{PREDICTION_API}/batch-predict", files=files)
                        
                        if response.status_code == 200:
                            results = response.json()['predictions']
                            
                            # Create results dataframe
                            results_df = pd.DataFrame(results)
                            
                            st.success(f"✅ Classified {len(results)} files")
                            
                            # Display results table
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Class distribution
                            if 'predicted_class' in results_df.columns:
                                class_counts = results_df['predicted_class'].value_counts()
                                
                                fig = px.pie(values=class_counts.values,
                                           names=class_counts.index,
                                           title='Predicted Class Distribution')
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Results (CSV)",
                                csv,
                                "batch_predictions.csv",
                                "text/csv"
                            )
                        else:
                            st.error(f"Batch prediction failed: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {e}")


def show_visualizations():
    """Visualizations page"""
    st.title("📊 Data Visualizations & Insights")
    
    st.info("📌 This section shows insights from your trained model and dataset.")
    
    # Mock data for demonstration (replace with actual data loading)
    tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Model Performance", "Feature Analysis"])
    
    with tab1:
        st.subheader("Dataset Distribution")
        
        # Example visualization
        sample_data = {
            'Class': ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
                     'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'],
            'Count': [1000, 429, 1000, 1000, 1000, 1000, 374, 1000, 929, 1000]
        }
        df = pd.DataFrame(sample_data)
        
        fig = px.bar(df, x='Class', y='Count', title='Training Samples per Class',
                    color='Count', color_continuous_scale='blues')
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", df['Count'].sum())
        with col2:
            st.metric("Number of Classes", len(df))
        with col3:
            st.metric("Avg Samples/Class", int(df['Count'].mean()))
    
    with tab2:
        st.subheader("Model Performance Metrics")
        
        # Training metrics visualization
        epochs = list(range(1, 31))
        train_acc = [0.3 + 0.023*i for i in epochs]
        val_acc = [0.28 + 0.022*i for i in epochs]
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Training History', 'Final Metrics'))
        
        fig.add_trace(go.Scatter(x=epochs, y=train_acc, name='Train Accuracy', mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=val_acc, name='Val Accuracy', mode='lines'), row=1, col=1)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [0.92, 0.91, 0.90, 0.91]
        
        fig.add_trace(go.Bar(x=metrics, y=values, name='Metrics'), row=1, col=2)
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", "92.0%")
        with col2:
            st.metric("Precision", "91.0%")
        with col3:
            st.metric("Recall", "90.0%")
        with col4:
            st.metric("F1-Score", "91.0%")
    
    with tab3:
        st.subheader("Feature Analysis")
        
        st.write("**Mel-Spectrogram Features**")
        st.write("""
        The model uses mel-spectrograms as input features:
        - **Sample Rate**: 22050 Hz
        - **Duration**: 4 seconds
        - **Mel Bands**: 128
        - **FFT Window**: 2048
        - **Hop Length**: 512
        """)
        
        # Example feature importance (mock data)
        feature_importance = pd.DataFrame({
            'Feature': ['Low Frequency Energy', 'Mid Frequency Energy', 'High Frequency Energy',
                       'Spectral Centroid', 'Zero Crossing Rate', 'Temporal Features'],
            'Importance': [0.25, 0.30, 0.15, 0.12, 0.10, 0.08]
        }).sort_values('Importance', ascending=True)
        
        fig = px.barh(feature_importance, x='Importance', y='Feature',
                     title='Feature Importance Analysis',
                     color='Importance', color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)


def show_upload_retrain():
    """Upload and retrain page"""
    st.title("📤 Upload Data & Retrain Model")
    
    st.info("🎯 Upload new audio samples and trigger model retraining to improve performance.")
    
    tab1, tab2 = st.tabs(["Upload Data", "Trigger Retraining"])
    
    with tab1:
        st.subheader("Upload Training Data")
        
        # Get available classes
        try:
            response = requests.get(f"{PREDICTION_API}/classes")
            if response.status_code == 200:
                available_classes = response.json()['classes']
            else:
                available_classes = []
        except:
            available_classes = []
        
        class_label = st.selectbox("Select Class Label", available_classes + ["Add New Class"])
        
        if class_label == "Add New Class":
            class_label = st.text_input("Enter New Class Name")
        
        uploaded_files = st.file_uploader(
            "Upload audio files for this class",
            type=['wav', 'mp3', 'ogg', 'flac'],
            accept_multiple_files=True
        )
        
        if uploaded_files and class_label:
            st.success(f"📁 {len(uploaded_files)} files selected for class '{class_label}'")
            
            if st.button("📤 Upload Files", type="primary", use_container_width=True):
                with st.spinner("Uploading files..."):
                    try:
                        files = [('files', (f.name, f, 'audio/wav')) for f in uploaded_files]
                        data = {'class_label': class_label}
                        
                        response = requests.post(
                            f"{RETRAIN_API}/upload",
                            files=files,
                            data=data
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ Successfully uploaded {result['saved_count']} files!")
                            
                            if result['failed_count'] > 0:
                                st.warning(f"⚠️ Failed to upload {result['failed_count']} files")
                        else:
                            st.error(f"Upload failed: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    with tab2:
        st.subheader("Trigger Model Retraining")
        
        st.write("""
        Retraining will:
        1. Process uploaded audio files
        2. Extract features (mel-spectrograms)
        3. Combine with existing training data
        4. Train the model with new data
        5. Save the updated model
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            augment = st.checkbox("Apply Data Augmentation", value=True,
                                 help="Apply noise, time shift, and other augmentations")
            use_pretrained = st.checkbox("Use Current Model as Starting Point", value=True,
                                        help="Fine-tune existing model instead of training from scratch")
        
        with col2:
            epochs = st.slider("Training Epochs", 10, 100, 30,
                              help="Number of training iterations")
        
        st.warning("⚠️ Retraining may take several minutes depending on data size.")
        
        if st.button("🚀 Start Retraining", type="primary", use_container_width=True):
            with st.spinner("Retraining model... This may take a while."):
                try:
                    data = {
                        'augment': augment,
                        'use_pretrained': use_pretrained,
                        'epochs': epochs
                    }
                    
                    response = requests.post(f"{RETRAIN_API}/retrain", data=data, timeout=600)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if result.get('success'):
                            st.success("✅ Model retraining completed successfully!")
                            
                            info = result.get('training_info', {})
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Training Samples", info.get('training_samples', 'N/A'))
                            with col2:
                                st.metric("Validation Accuracy",
                                         f"{info.get('val_accuracy', 0)*100:.2f}%")
                            with col3:
                                st.metric("Epochs Trained", info.get('epochs', epochs))
                            
                            st.json(info)
                        else:
                            st.error(f"Retraining failed: {result.get('error', 'Unknown error')}")
                    else:
                        st.error(f"Retraining failed: {response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")


def show_monitoring():
    """Monitoring page"""
    st.title("📈 Model Monitoring & Performance")
    
    st.info("📊 Real-time monitoring of model performance and system health")
    
    # Get metrics
    try:
        response = requests.get(f"{PREDICTION_API}/metrics")
        if response.status_code == 200:
            metrics = response.json()
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model Uptime", f"{metrics['uptime_seconds']/3600:.1f} hrs")
            
            with col2:
                st.metric("Total Predictions", metrics['total_predictions'])
            
            with col3:
                st.metric("Model Status", "✅ Active" if metrics['model_loaded'] else "❌ Inactive")
            
            with col4:
                avg_time = metrics['uptime_seconds'] / max(metrics['total_predictions'], 1)
                st.metric("Avg Response Time", f"{avg_time:.2f}s")
            
            # Predictions by class
            if metrics['predictions_by_class']:
                st.subheader("Predictions by Class")
                
                class_df = pd.DataFrame([
                    {"Class": k, "Count": v}
                    for k, v in metrics['predictions_by_class'].items()
                ])
                
                fig = px.pie(class_df, values='Count', names='Class',
                           title='Distribution of Predictions')
                st.plotly_chart(fig, use_container_width=True)
            
            # Training history
            st.subheader("Latest Training Information")
            
            try:
                response = requests.get(f"{RETRAIN_API}/training-history")
                if response.status_code == 200:
                    history = response.json()
                    
                    if 'message' not in history:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.json(history)
                        
                        with col2:
                            st.write("**Training Summary:**")
                            st.write(f"- Timestamp: {history.get('timestamp', 'N/A')}")
                            st.write(f"- Training Samples: {history.get('training_samples', 'N/A')}")
                            st.write(f"- Validation Accuracy: {history.get('val_accuracy', 0)*100:.2f}%")
                            st.write(f"- Epochs: {history.get('epochs', 'N/A')}")
                    else:
                        st.info(history['message'])
            except:
                st.warning("Unable to fetch training history")
                
        else:
            st.error("Unable to fetch metrics")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
    
    # Refresh button
    if st.button("🔄 Refresh Metrics"):
        st.rerun()


if __name__ == "__main__":
    main()
