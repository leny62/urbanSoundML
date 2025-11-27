"""
Utility functions for the Urban Sound Classification project
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_directory_structure():
    """Create necessary directory structure for the project"""
    directories = [
        "data/train",
        "data/test",
        "data/uploaded",
        "data/processed",
        "models",
        "models/backups",
        "results",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def save_json(data: dict, filepath: str):
    """Save dictionary to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath: str) -> Optional[dict]:
    """Load JSON file"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def get_model_info(model_path: str = "models/best_model.h5") -> Dict:
    """Get model information"""
    import tensorflow as tf
    
    if not os.path.exists(model_path):
        return {"error": "Model not found"}
    
    model = tf.keras.models.load_model(model_path)
    
    return {
        "total_params": model.count_params(),
        "trainable_params": sum([tf.size(w).numpy() for w in model.trainable_weights]),
        "layers": len(model.layers),
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "file_size_mb": os.path.getsize(model_path) / (1024 * 1024)
    }


def calculate_metrics_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate comprehensive metrics summary"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, confusion_matrix
    )
    
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average='macro')),
        "precision_weighted": float(precision_score(y_true, y_pred, average='weighted')),
        "recall_macro": float(recall_score(y_true, y_pred, average='macro')),
        "recall_weighted": float(recall_score(y_true, y_pred, average='weighted')),
        "f1_macro": float(f1_score(y_true, y_pred, average='macro')),
        "f1_weighted": float(f1_score(y_true, y_pred, average='weighted')),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }


def log_prediction(
    filename: str,
    predicted_class: str,
    confidence: float,
    log_file: str = "logs/predictions.log"
):
    """Log prediction to file"""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "filename": filename,
        "predicted_class": predicted_class,
        "confidence": confidence
    }
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


def get_system_info() -> Dict:
    """Get system information"""
    import platform
    import psutil
    
    return {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "memory_available_gb": psutil.virtual_memory().available / (1024**3)
    }


if __name__ == "__main__":
    # Create directory structure
    create_directory_structure()
    
    # Print system info
    print("System Information:")
    print(json.dumps(get_system_info(), indent=2))
