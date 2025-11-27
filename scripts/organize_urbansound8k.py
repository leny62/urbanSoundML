"""
Script to organize UrbanSound8K dataset into train/test folders by class.
Uses fold 10 for testing and folds 1-9 for training (standard practice).
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def organize_urbansound8k(
    audio_dir="../data/UrbanSound8K/audio",
    metadata_csv="../data/UrbanSound8K/metadata/UrbanSound8K.csv",
    output_dir="../data"
):
    """
    Organize UrbanSound8K dataset into train/test class folders.
    
    Args:
        audio_dir: Path to 'audio' folder containing fold1-fold10
        metadata_csv: Path to UrbanSound8K.csv
        output_dir: Destination folder (creates train/ and test/ subdirs)
    """
    
    print("=" * 70)
    print("URBANSOUND8K DATASET ORGANIZATION")
    print("=" * 70)
    
    # Class mapping (classID -> class name)
    class_names = {
        0: 'air_conditioner',
        1: 'car_horn',
        2: 'children_playing',
        3: 'dog_bark',
        4: 'drilling',
        5: 'engine_idling',
        6: 'gun_shot',
        7: 'jackhammer',
        8: 'siren',
        9: 'street_music'
    }
    
    # Read metadata
    print(f"\n📄 Reading metadata from: {metadata_csv}")
    metadata = pd.read_csv(metadata_csv)
    print(f"✅ Found {len(metadata)} audio files")
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    
    print(f"\n📁 Creating directory structure...")
    for class_name in class_names.values():
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
    print("✅ Directories created")
    
    # Statistics
    train_counts = {name: 0 for name in class_names.values()}
    test_counts = {name: 0 for name in class_names.values()}
    errors = []
    
    print(f"\n🔄 Copying files...")
    print("   Using fold 10 for testing, folds 1-9 for training")
    
    # Process each file
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing"):
        slice_file_name = row['slice_file_name']
        class_id = row['classID']
        fold = row['fold']
        
        class_name = class_names[class_id]
        
        # Source path: audio/fold{X}/filename.wav
        source_path = os.path.join(audio_dir, f'fold{fold}', slice_file_name)
        
        # Determine if train or test (fold 10 = test, rest = train)
        if fold == 10:
            dest_dir = test_dir
            test_counts[class_name] += 1
        else:
            dest_dir = train_dir
            train_counts[class_name] += 1
        
        # Destination path
        dest_path = os.path.join(dest_dir, class_name, slice_file_name)
        
        # Copy file
        try:
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
            else:
                errors.append(f"File not found: {source_path}")
        except Exception as e:
            errors.append(f"Error copying {slice_file_name}: {str(e)}")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("ORGANIZATION COMPLETE!")
    print("=" * 70)
    
    print("\n📊 TRAINING SET:")
    print("-" * 70)
    total_train = 0
    for class_name in sorted(class_names.values()):
        count = train_counts[class_name]
        total_train += count
        print(f"  {class_name:20s}: {count:4d} files")
    print("-" * 70)
    print(f"  {'TOTAL':20s}: {total_train:4d} files\n")
    
    print("📊 TEST SET:")
    print("-" * 70)
    total_test = 0
    for class_name in sorted(class_names.values()):
        count = test_counts[class_name]
        total_test += count
        print(f"  {class_name:20s}: {count:4d} files")
    print("-" * 70)
    print(f"  {'TOTAL':20s}: {total_test:4d} files\n")
    
    print(f"📈 SPLIT RATIO: {total_train/(total_train+total_test)*100:.1f}% train, "
          f"{total_test/(total_train+total_test)*100:.1f}% test")
    
    # Report errors
    if errors:
        print(f"\n⚠️  {len(errors)} errors occurred:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    else:
        print("\n✅ No errors!")
    
    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Run: python scripts/verify_data_structure.py")
    print("  2. Open: notebook/urbansound_pipeline.ipynb")
    print("  3. Start training your model!")
    print("=" * 70)

if __name__ == "__main__":
    # Update these paths if your structure is different
    AUDIO_DIR = "../data/UrbanSound8K/audio"
    METADATA_CSV = "../data/UrbanSound8K/metadata/UrbanSound8K.csv"
    OUTPUT_DIR = "../data"
    
    # Check if paths exist
    if not os.path.exists(AUDIO_DIR):
        print(f"❌ Error: Audio directory not found: {AUDIO_DIR}")
        print("   Please update AUDIO_DIR in the script")
        exit(1)
    
    if not os.path.exists(METADATA_CSV):
        print(f"❌ Error: Metadata CSV not found: {METADATA_CSV}")
        print("   Please update METADATA_CSV in the script")
        exit(1)
    
    organize_urbansound8k(AUDIO_DIR, METADATA_CSV, OUTPUT_DIR)