"""
FastAPI Prediction Service for Urban Sound Classification
Provides REST API endpoints for audio classification
"""

import os
import sys
import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import uvicorn
from datetime import datetime
import logging

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import AudioPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Urban Sound Classification API",
    description="API for classifying urban sounds using deep learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
preprocessor = None
class_names = []
MODEL_PATH = "models/urbansound_cnn.keras"
CLASS_NAMES_PATH = "models/class_names.json"

# Metrics storage
prediction_metrics = {
    "total_predictions": 0,
    "predictions_by_class": {},
    "start_time": datetime.now().isoformat()
}


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    timestamp: str


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    uptime_seconds: float
    total_predictions: int


def _load_resources():
    """Helper to load model, class names, and preprocessor"""
    global model, preprocessor, class_names
    
    # Initialize preprocessor every time to ensure fresh state
    preprocessor = AudioPreprocessor(
        sample_rate=22050,
        duration=4.0,
        n_mels=128
    )
    logger.info("Preprocessor initialized")
    
    # Load class names first
    if os.path.exists(CLASS_NAMES_PATH):
        import json
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        logger.info(f"Loaded {len(class_names)} class names")
    else:
        class_names = []
        logger.warning(f"Class names file not found at {CLASS_NAMES_PATH}")
    
    # Load model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    else:
        model = None
        logger.warning(f"Model not found at {MODEL_PATH}")


@app.on_event("startup")
async def load_model_on_startup():
    """Load resources when the API starts"""
    try:
        _load_resources()
    except Exception as e:
        logger.error(f"Error during startup: {e}")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Urban Sound Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "metrics": "/metrics"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = (datetime.now() - datetime.fromisoformat(prediction_metrics["start_time"])).total_seconds()
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        uptime_seconds=uptime,
        total_predictions=prediction_metrics["total_predictions"]
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)):
    """
    Predict class for a single audio file
    
    Args:
        file: Audio file (WAV format recommended)
        
    Returns:
        Prediction with confidence scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not class_names:
        raise HTTPException(status_code=503, detail="Class names not loaded")
    
    try:
        # Read file content
        contents = await file.read()
        
        # Save temporarily
        temp_path = f"temp_{datetime.now().timestamp()}.wav"
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Preprocess audio
        audio = preprocessor.load_audio(temp_path)
        if audio is None:
            raise HTTPException(status_code=400, detail="Failed to load audio file")
        
        mel_spec = preprocessor.extract_mel_spectrogram(audio)
        mel_spec = mel_spec[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
        
        # Make prediction
        predictions = model.predict(mel_spec, verbose=0)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx])
        
        # Ensure class name list matches model output
        num_outputs = predictions.shape[1]
        if len(class_names) < num_outputs:
            logger.warning(
                "Class name count (%d) is less than model outputs (%d). Filling placeholders.",
                len(class_names), num_outputs
            )
            extended_names = class_names + [f"class_{i}" for i in range(len(class_names), num_outputs)]
        else:
            extended_names = class_names[:num_outputs]
        
        predicted_class = extended_names[predicted_idx]
        
        # Create probability dictionary aligned with extended names
        all_probs = {
            extended_names[i]: float(predictions[0][i])
            for i in range(num_outputs)
        }
        
        # Update metrics
        prediction_metrics["total_predictions"] += 1
        if predicted_class not in prediction_metrics["predictions_by_class"]:
            prediction_metrics["predictions_by_class"][predicted_class] = 0
        prediction_metrics["predictions_by_class"][predicted_class] += 1
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            all_probabilities=all_probs,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch-predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    """
    Predict classes for multiple audio files
    
    Args:
        files: List of audio files
        
    Returns:
        List of predictions
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not class_names:
        raise HTTPException(status_code=503, detail="Class names not loaded")
    
    results = []
    
    for file in files:
        try:
            # Read file
            contents = await file.read()
            temp_path = f"temp_{datetime.now().timestamp()}_{file.filename}"
            
            with open(temp_path, "wb") as f:
                f.write(contents)
            
            # Preprocess
            audio = preprocessor.load_audio(temp_path)
            if audio is None:
                results.append({
                    "filename": file.filename,
                    "error": "Failed to load audio"
                })
                continue
            
            mel_spec = preprocessor.extract_mel_spectrogram(audio)
            mel_spec = mel_spec[np.newaxis, ..., np.newaxis]
            
            # Predict
            predictions = model.predict(mel_spec, verbose=0)
            predicted_idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_idx])

            num_outputs = predictions.shape[1]
            if len(class_names) < num_outputs:
                logger.warning(
                    "Class name count (%d) is less than model outputs (%d). Filling placeholders.",
                    len(class_names), num_outputs
                )
                extended_names = class_names + [f"class_{i}" for i in range(len(class_names), num_outputs)]
            else:
                extended_names = class_names[:num_outputs]

            predicted_class = extended_names[predicted_idx]
            
            results.append({
                "filename": file.filename,
                "predicted_class": predicted_class,
                "confidence": confidence
            })
            
            # Update metrics
            prediction_metrics["total_predictions"] += 1
            if predicted_class not in prediction_metrics["predictions_by_class"]:
                prediction_metrics["predictions_by_class"][predicted_class] = 0
            prediction_metrics["predictions_by_class"][predicted_class] += 1
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse(content={"predictions": results})


@app.post("/reload")
async def reload_resources():
    """Reload the model and class names (use after retraining)"""
    try:
        _load_resources()
        return {"status": "success", "message": "Resources reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")


@app.get("/metrics")
async def get_metrics():
    """Get prediction metrics"""
    uptime = (datetime.now() - datetime.fromisoformat(prediction_metrics["start_time"])).total_seconds()
    
    return {
        "uptime_seconds": uptime,
        "total_predictions": prediction_metrics["total_predictions"],
        "predictions_by_class": prediction_metrics["predictions_by_class"],
        "model_path": MODEL_PATH,
        "model_loaded": model is not None
    }


@app.get("/classes")
async def get_classes():
    """Get list of available classes"""
    return {
        "classes": class_names,
        "num_classes": len(class_names)
    }


if __name__ == "__main__":
    uvicorn.run(
        "prediction:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )