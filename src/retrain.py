"""
Model Retraining Pipeline
Handles data upload, preprocessing, and model retraining
"""

import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
import json
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score
import logging

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import AudioPreprocessor, prepare_train_test_split
from model import UrbanSoundCNN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrainingPipeline:
    """Handles the complete retraining pipeline"""
    
    def __init__(self,
                 data_dir: str = "data/uploaded",
                 processed_dir: str = "data/processed",
                 model_dir: str = "models",
                 backup_dir: str = "models/backups"):
        """
        Initialize retraining pipeline
        
        Args:
            data_dir: Directory for uploaded audio files
            processed_dir: Directory for processed data
            model_dir: Directory for model files
            backup_dir: Directory for model backups
        """
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.model_dir = model_dir
        self.backup_dir = backup_dir
        
        # Create directories
        for directory in [data_dir, processed_dir, model_dir, backup_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.preprocessor = AudioPreprocessor()
        self.class_names = self._load_class_names()
        
    def _load_class_names(self) -> List[str]:
        """Load class names from file"""
        # Try JSON format first (new format)
        class_names_json = os.path.join(self.model_dir, 'class_names.json')
        if os.path.exists(class_names_json):
            with open(class_names_json, 'r') as f:
                return json.load(f)
        
        # Fallback to txt format (old format)
        class_names_txt = os.path.join(self.processed_dir, 'class_names.txt')
        if os.path.exists(class_names_txt):
            with open(class_names_txt, 'r') as f:
                return [line.strip() for line in f.readlines()]
        
        return []
    
    def save_uploaded_files(self, 
                           files: List[tuple],
                           class_label: str) -> dict:
        """
        Save uploaded audio files to disk
        
        Args:
            files: List of (filename, file_content) tuples
            class_label: Class label for the files
            
        Returns:
            Dictionary with save status
        """
        class_dir = os.path.join(self.data_dir, class_label)
        os.makedirs(class_dir, exist_ok=True)
        
        saved_files = []
        failed_files = []
        
        for filename, content in files:
            try:
                filepath = os.path.join(class_dir, filename)
                with open(filepath, 'wb') as f:
                    f.write(content)
                saved_files.append(filename)
            except Exception as e:
                logger.error(f"Failed to save {filename}: {e}")
                failed_files.append(filename)
        
        return {
            "saved_count": len(saved_files),
            "failed_count": len(failed_files),
            "saved_files": saved_files,
            "failed_files": failed_files,
            "class_label": class_label
        }
    
    def preprocess_uploaded_data(self, augment: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess uploaded data for retraining
        
        Args:
            augment: Whether to apply data augmentation
            
        Returns:
            Tuple of (features, labels)
        """
        logger.info("Starting preprocessing of uploaded data...")
        
        # Process uploaded data
        X_new, y_new, new_classes = self.preprocessor.process_dataset(
            data_dir=self.data_dir,
            augment=augment
        )
        
        logger.info(f"Processed {len(X_new)} samples from uploaded data")
        
        # Update class names if new classes found
        for cls in new_classes:
            if cls not in self.class_names:
                self.class_names.append(cls)
        
        # Save updated class names
        self._save_class_names()
        
        return X_new, y_new
    
    def _save_class_names(self):
        """Save class names to file"""
        # Save in JSON format (new format)
        class_names_json = os.path.join(self.model_dir, 'class_names.json')
        with open(class_names_json, 'w') as f:
            json.dump(self.class_names, f)
        
        # Also save in TXT format for backward compatibility
        class_names_txt = os.path.join(self.processed_dir, 'class_names.txt')
        with open(class_names_txt, 'w') as f:
            f.write('\n'.join(self.class_names))
        
        logger.info(f"Saved {len(self.class_names)} class names")
    
    def load_existing_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load existing preprocessed training data
        
        Returns:
            Tuple of (X_train, y_train)
        """
        X_train_path = os.path.join(self.processed_dir, 'X_train.npy')
        y_train_path = os.path.join(self.processed_dir, 'y_train.npy')
        
        if os.path.exists(X_train_path) and os.path.exists(y_train_path):
            X_train = np.load(X_train_path)
            y_train = np.load(y_train_path)
            # If data was saved with channel dimension, squeeze it to match new data
            if X_train.ndim == 4 and X_train.shape[-1] == 1:
                X_train = np.squeeze(X_train, axis=-1)
            logger.info(f"Loaded existing data: {X_train.shape[0]} samples")
            return X_train, y_train
        
        return None, None
    
    def combine_datasets(self,
                        X_existing: Optional[np.ndarray],
                        y_existing: Optional[np.ndarray],
                        X_new: np.ndarray,
                        y_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine existing and new datasets
        
        Args:
            X_existing: Existing features
            y_existing: Existing labels
            X_new: New features
            y_new: New labels
            
        Returns:
            Combined (X, y)
        """
        if X_existing is not None and y_existing is not None:
            X_combined = np.concatenate([X_existing, X_new], axis=0)
            y_combined = np.concatenate([y_existing, y_new], axis=0)
            logger.info(f"Combined dataset size: {X_combined.shape[0]} samples")
        else:
            X_combined = X_new
            y_combined = y_new
            logger.info(f"Using new dataset: {X_combined.shape[0]} samples")
        
        return X_combined, y_combined
    
    def backup_current_model(self):
        """Create backup of current model before retraining"""
        # Try .keras format first (new format)
        model_path = os.path.join(self.model_dir, 'urbansound_cnn.keras')
        if not os.path.exists(model_path):
            # Fallback to .h5 format (old format)
            model_path = os.path.join(self.model_dir, 'best_model.h5')
        
        if os.path.exists(model_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = os.path.splitext(model_path)[1]
            backup_path = os.path.join(self.backup_dir, f'model_backup_{timestamp}{extension}')
            shutil.copy(model_path, backup_path)
            logger.info(f"Model backed up to {backup_path}")
            return backup_path
        
        return None
    
    def load_pretrained_model(self) -> Optional[tf.keras.Model]:
        """Load existing pretrained model"""
        # Try .keras format first (new format)
        model_path = os.path.join(self.model_dir, 'urbansound_cnn.keras')
        if not os.path.exists(model_path):
            # Fallback to .h5 format (old format)
            model_path = os.path.join(self.model_dir, 'best_model.h5')
        
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                logger.info(f"Loaded pretrained model from {model_path}")
                return model
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        
        return None
    
    def retrain_model(self,
                     X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_val: np.ndarray,
                     y_val: np.ndarray,
                     use_pretrained: bool = True,
                     epochs: int = 30,
                     batch_size: int = 32) -> dict:
        """
        Retrain model with new data
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            use_pretrained: Whether to use pretrained model
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting model retraining...")
        
        # Backup current model
        backup_path = self.backup_current_model()
        
        # Add channel dimension if needed
        if len(X_train.shape) == 3:
            X_train = X_train[..., np.newaxis]
            X_val = X_val[..., np.newaxis]
        
        input_shape = X_train.shape[1:]
        num_classes = len(self.class_names)
        
        # Load or create model
        if use_pretrained:
            model_instance = self.load_pretrained_model()
            if model_instance is not None:
                # Check if output layer matches
                if model_instance.output_shape[-1] != num_classes:
                    logger.warning("Model output shape mismatch. Creating new model.")
                    model_wrapper = UrbanSoundCNN(input_shape, num_classes, 'custom')
                    model_wrapper.build_model()
                else:
                    # Wrap existing model
                    model_wrapper = UrbanSoundCNN(input_shape, num_classes, 'custom')
                    model_wrapper.model = model_instance
            else:
                model_wrapper = UrbanSoundCNN(input_shape, num_classes, 'custom')
                model_wrapper.build_model()
        else:
            model_wrapper = UrbanSoundCNN(input_shape, num_classes, 'custom')
            model_wrapper.build_model()
        
        # Train model
        history = model_wrapper.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            model_dir=self.model_dir
        )
        
        # Evaluate on validation set (precision/recall via sklearn to avoid TF metric issues)
        val_loss, val_accuracy = model_wrapper.model.evaluate(X_val, y_val, verbose=0)
        y_val_pred = np.argmax(model_wrapper.model.predict(X_val, verbose=0), axis=1)
        val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
        val_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
        
        # Save training info
        training_info = {
            "timestamp": datetime.now().isoformat(),
            "training_samples": int(len(X_train)),
            "validation_samples": int(len(X_val)),
            "num_classes": num_classes,
            "classes": self.class_names,
            "epochs": epochs,
            "batch_size": batch_size,
            "val_accuracy": float(val_accuracy),
            "val_loss": float(val_loss),
            "val_precision": float(val_precision),
            "val_recall": float(val_recall),
            "val_f1": float(val_f1),
            "backup_path": backup_path,
            "use_pretrained": use_pretrained
        }
        
        # Save training info
        info_path = os.path.join(self.model_dir, 'latest_training_info.json')
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=4)
        
        logger.info(f"Retraining completed. Validation accuracy: {val_accuracy:.4f}")
        
        return training_info
    
    def full_retrain_pipeline(self,
                              augment: bool = True,
                              use_pretrained: bool = True,
                              epochs: int = 30,
                              test_size: float = 0.2) -> dict:
        """
        Execute complete retraining pipeline
        
        Args:
            augment: Apply data augmentation
            use_pretrained: Use existing model as starting point
            epochs: Number of training epochs
            test_size: Validation split size
            
        Returns:
            Results dictionary
        """
        try:
            # Step 1: Preprocess uploaded data
            logger.info("Step 1: Preprocessing uploaded data")
            X_new, y_new = self.preprocess_uploaded_data(augment=augment)
            
            if len(X_new) == 0:
                return {"error": "No data to retrain on"}
            
            # Step 2: Load existing data
            logger.info("Step 2: Loading existing data")
            X_existing, y_existing = self.load_existing_data()
            
            # Step 3: Combine datasets
            logger.info("Step 3: Combining datasets")
            X_combined, y_combined = self.combine_datasets(
                X_existing, y_existing, X_new, y_new
            )
            
            # Step 4: Split data
            logger.info("Step 4: Splitting data")
            from sklearn.model_selection import train_test_split
            
            # Add channel dimension
            X_combined = X_combined[..., np.newaxis]
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_combined, y_combined,
                test_size=test_size,
                random_state=42,
                stratify=y_combined
            )
            
            # Step 5: Retrain model
            logger.info("Step 5: Retraining model")
            training_info = self.retrain_model(
                X_train, y_train,
                X_val, y_val,
                use_pretrained=use_pretrained,
                epochs=epochs
            )
            
            # Step 6: Save updated training data
            logger.info("Step 6: Saving updated data")
            np.save(os.path.join(self.processed_dir, 'X_train.npy'), X_train)
            np.save(os.path.join(self.processed_dir, 'y_train.npy'), y_train)
            
            return {
                "success": True,
                "training_info": training_info,
                "message": "Retraining completed successfully"
            }
            
        except Exception as e:
            logger.error(f"Retraining pipeline failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Retraining failed"
            }
    
    def get_training_history(self) -> Optional[dict]:
        """Get latest training information"""
        info_path = os.path.join(self.model_dir, 'latest_training_info.json')
        
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                return json.load(f)
        
        return None
    
    def clear_uploaded_data(self):
        """Clear uploaded data after successful retraining"""
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
            os.makedirs(self.data_dir, exist_ok=True)
            logger.info("Uploaded data cleared")


# FastAPI endpoints for retraining
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import asyncio

retrain_app = FastAPI(title="Model Retraining API")
pipeline = RetrainingPipeline()


@retrain_app.post("/upload")
async def upload_training_data(
    files: List[UploadFile] = File(...),
    class_label: str = Form(...)
):
    """Upload training data for a specific class"""
    try:
        file_data = []
        for file in files:
            content = await file.read()
            file_data.append((file.filename, content))
        
        result = pipeline.save_uploaded_files(file_data, class_label)
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@retrain_app.post("/retrain")
async def trigger_retraining(
    augment: bool = Form(True),
    use_pretrained: bool = Form(True),
    epochs: int = Form(30)
):
    """Trigger model retraining"""
    try:
        result = pipeline.full_retrain_pipeline(
            augment=augment,
            use_pretrained=use_pretrained,
            epochs=epochs
        )
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@retrain_app.get("/training-history")
async def get_training_history():
    """Get latest training history"""
    history = pipeline.get_training_history()
    if history:
        return JSONResponse(content=history)
    return JSONResponse(content={"message": "No training history available"})


@retrain_app.post("/clear-uploads")
async def clear_uploads():
    """Clear uploaded data"""
    try:
        pipeline.clear_uploaded_data()
        return JSONResponse(content={"message": "Uploaded data cleared"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(retrain_app, host="0.0.0.0", port=8001)