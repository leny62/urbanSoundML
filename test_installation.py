"""
Test script to verify installation and basic functionality
"""

import sys
import importlib

def test_imports():
    """Test if all required packages are installed"""
    required_packages = [
        'numpy',
        'pandas',
        'sklearn',
        'tensorflow',
        'librosa',
        'soundfile',
        'fastapi',
        'uvicorn',
        'streamlit',
        'matplotlib',
        'seaborn',
        'plotly',
        'locust'
    ]
    
    print("Testing package imports...")
    failed = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed.append(package)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True


def test_custom_modules():
    """Test custom modules"""
    print("\nTesting custom modules...")
    
    try:
        sys.path.append('src')
        from preprocessing import AudioPreprocessor
        from model import UrbanSoundCNN
        
        print("✅ preprocessing.py")
        print("✅ model.py")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_directory_structure():
    """Test if directory structure exists"""
    import os
    
    print("\nTesting directory structure...")
    required_dirs = [
        'data/train',
        'data/test',
        'models',
        'notebook',
        'src'
    ]
    
    failed = []
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ (missing)")
            failed.append(directory)
    
    if failed:
        print(f"\n⚠️  Missing directories: {', '.join(failed)}")
        print("Run setup script to create directories")
    
    return len(failed) == 0


def test_audio_processing():
    """Test basic audio processing"""
    print("\nTesting audio processing capabilities...")
    
    try:
        import librosa
        import numpy as np
        
        # Generate test audio
        sr = 22050
        duration = 1.0
        frequency = 440  # A4 note
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Test mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        
        print(f"✅ Generated test audio: {len(audio)} samples")
        print(f"✅ Mel-spectrogram shape: {mel_spec.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Urban Sound Classification - Installation Test")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Custom Modules", test_custom_modules),
        ("Directory Structure", test_directory_structure),
        ("Audio Processing", test_audio_processing)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} failed with error: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to go!")
        print("\nNext steps:")
        print("1. Add UrbanSound8K data to data/train/ and data/test/")
        print("2. Run: jupyter notebook notebook/urbansound_pipeline.ipynb")
        print("3. Train your model and start building!")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
    
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
