"""
Quick test script to verify API setup before starting the server
"""

import os
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    size = ""
    if exists and os.path.isfile(filepath):
        size_mb = os.path.getsize(filepath) / (1024*1024)
        size = f" ({size_mb:.2f} MB)"
    print(f"{status} {description}: {filepath}{size}")
    return exists

def main():
    print("=" * 80)
    print("🔍 API SETUP VERIFICATION")
    print("=" * 80)
    
    # Check model files
    print("\n📦 Model Files:")
    model_keras = check_file_exists("models/urbansound_cnn.keras", "Primary model (.keras)")
    model_h5 = check_file_exists("models/best_model.h5", "Compatible model (.h5)")
    
    # Check class names
    print("\n📋 Class Names:")
    classes_json = check_file_exists("models/class_names.json", "Class names (JSON)")
    classes_txt = check_file_exists("data/processed/class_names.txt", "Class names (TXT)")
    
    # Check directories
    print("\n📁 Required Directories:")
    check_file_exists("data/processed", "Processed data directory")
    check_file_exists("data/uploaded", "Upload directory")
    check_file_exists("models/backups", "Model backups directory")
    
    # Check source files
    print("\n🐍 API Source Files:")
    prediction_py = check_file_exists("src/prediction.py", "Prediction API")
    retrain_py = check_file_exists("src/retrain.py", "Retraining API")
    preprocessing_py = check_file_exists("src/preprocessing.py", "Preprocessing module")
    model_py = check_file_exists("src/model.py", "Model module")
    
    # Check test data
    print("\n🎵 Test Data:")
    test_dir = Path("data/test")
    if test_dir.exists():
        wav_files = list(test_dir.rglob("*.wav"))
        print(f"✅ Test audio files: {len(wav_files)} files found")
    else:
        print(f"❌ Test audio directory not found")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    
    all_required = [
        model_keras or model_h5,  # At least one model format
        classes_json,
        prediction_py,
        preprocessing_py,
        model_py
    ]
    
    if all(all_required):
        print("✅ ALL CRITICAL FILES PRESENT - API IS READY TO START!")
        print("\n🚀 To start the API, run:")
        print("   python -m uvicorn src.prediction:app --host 0.0.0.0 --port 8000 --reload")
        print("\n📖 API Documentation will be available at:")
        print("   http://localhost:8000/docs")
        return 0
    else:
        print("❌ MISSING CRITICAL FILES - Please complete training first!")
        print("\n📝 Missing items:")
        if not (model_keras or model_h5):
            print("   - Model file (run notebook to train model)")
        if not classes_json:
            print("   - Class names JSON (run notebook to save classes)")
        if not prediction_py:
            print("   - Prediction API source file")
        if not preprocessing_py:
            print("   - Preprocessing module")
        if not model_py:
            print("   - Model module")
        return 1

if __name__ == "__main__":
    exit(main())
