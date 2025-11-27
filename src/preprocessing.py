"""
Audio Preprocessing Module for UrbanSound8K Classification
Handles data loading, mel-spectrogram extraction, and augmentation
"""

import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
import soundfile as sf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


class AudioPreprocessor:
    """Handles audio preprocessing for urban sound classification"""
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 duration: float = 4.0,
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 hop_length: int = 512):
        """
        Initialize preprocessor with audio parameters
        
        Args:
            sample_rate: Target sampling rate
            duration: Fixed duration for all audio clips
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length = int(sample_rate * duration)
        
    def load_audio(self, file_path: str) -> np.ndarray:
        """
        Load and preprocess audio file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Preprocessed audio signal
        """
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            # Pad or truncate to fixed length
            audio = self._pad_or_truncate(audio)
            
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def _pad_or_truncate(self, audio: np.ndarray) -> np.ndarray:
        """Pad or truncate audio to fixed length"""
        if len(audio) < self.max_length:
            # Pad with zeros
            audio = np.pad(audio, (0, self.max_length - len(audio)), mode='constant')
        else:
            # Truncate
            audio = audio[:self.max_length]
        return audio
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel-spectrogram from audio signal
        
        Args:
            audio: Audio signal
            
        Returns:
            Mel-spectrogram in dB scale
        """
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 40) -> np.ndarray:
        """
        Extract MFCC features from audio
        
        Args:
            audio: Audio signal
            n_mfcc: Number of MFCCs to extract
            
        Returns:
            MFCC features
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return mfcc
    
    def augment_audio(self, audio: np.ndarray, augmentation_type: str = 'noise') -> np.ndarray:
        """
        Apply data augmentation to audio
        
        Args:
            audio: Input audio signal
            augmentation_type: Type of augmentation ('noise', 'shift', 'stretch', 'pitch')
            
        Returns:
            Augmented audio signal
        """
        if augmentation_type == 'noise':
            # Add white noise
            noise = np.random.randn(len(audio))
            audio_noisy = audio + 0.005 * noise
            return audio_noisy
        
        elif augmentation_type == 'shift':
            # Time shift
            shift = np.random.randint(self.sample_rate * 0.5)
            return np.roll(audio, shift)
        
        elif augmentation_type == 'stretch':
            # Time stretch
            rate = np.random.uniform(0.8, 1.2)
            return librosa.effects.time_stretch(audio, rate=rate)
        
        elif augmentation_type == 'pitch':
            # Pitch shift
            n_steps = np.random.randint(-3, 3)
            return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
        
        return audio
    
    def process_dataset(self, 
                       data_dir: str, 
                       metadata_file: Optional[str] = None,
                       augment: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Process entire dataset
        
        Args:
            data_dir: Directory containing audio files
            metadata_file: Optional CSV file with metadata
            augment: Whether to apply data augmentation
            
        Returns:
            Tuple of (features, labels, class_names)
        """
        features = []
        labels = []
        class_names = []
        
        if metadata_file and os.path.exists(metadata_file):
            # Load from metadata CSV
            df = pd.read_csv(metadata_file)
            class_names = sorted(df['class'].unique().tolist())
            
            for idx, row in df.iterrows():
                file_path = os.path.join(data_dir, row['filename'])
                if os.path.exists(file_path):
                    audio = self.load_audio(file_path)
                    if audio is not None:
                        mel_spec = self.extract_mel_spectrogram(audio)
                        features.append(mel_spec)
                        labels.append(class_names.index(row['class']))
                        
                        # Apply augmentation
                        if augment:
                            for aug_type in ['noise', 'shift']:
                                aug_audio = self.augment_audio(audio, aug_type)
                                aug_mel_spec = self.extract_mel_spectrogram(aug_audio)
                                features.append(aug_mel_spec)
                                labels.append(class_names.index(row['class']))
        else:
            # Load from directory structure (class folders)
            data_path = Path(data_dir)
            class_folders = [f for f in data_path.iterdir() if f.is_dir()]
            class_names = sorted([f.name for f in class_folders])
            
            for class_idx, class_folder in enumerate(sorted(class_folders)):
                audio_files = list(class_folder.glob('*.wav'))
                
                for audio_file in audio_files:
                    audio = self.load_audio(str(audio_file))
                    if audio is not None:
                        mel_spec = self.extract_mel_spectrogram(audio)
                        features.append(mel_spec)
                        labels.append(class_idx)
                        
                        if augment:
                            for aug_type in ['noise', 'shift']:
                                aug_audio = self.augment_audio(audio, aug_type)
                                aug_mel_spec = self.extract_mel_spectrogram(aug_audio)
                                features.append(aug_mel_spec)
                                labels.append(class_idx)
        
        return np.array(features), np.array(labels), class_names
    
    def visualize_spectrogram(self, audio: np.ndarray, title: str = "Mel-Spectrogram"):
        """Visualize mel-spectrogram"""
        mel_spec = self.extract_mel_spectrogram(audio)
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            mel_spec,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis='mel'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.show()


def prepare_train_test_split(X: np.ndarray, 
                            y: np.ndarray, 
                            test_size: float = 0.2,
                            random_state: int = 42) -> Tuple:
    """
    Split data into train and test sets
    
    Args:
        X: Features
        y: Labels
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Add channel dimension for CNN
    X = X[..., np.newaxis]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def save_preprocessed_data(X_train: np.ndarray, 
                          X_test: np.ndarray,
                          y_train: np.ndarray,
                          y_test: np.ndarray,
                          class_names: List[str],
                          save_dir: str = 'data/processed'):
    """Save preprocessed data to disk"""
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
    
    with open(os.path.join(save_dir, 'class_names.txt'), 'w') as f:
        f.write('\n'.join(class_names))
    
    print(f"Preprocessed data saved to {save_dir}")


def load_preprocessed_data(save_dir: str = 'data/processed') -> Tuple:
    """Load preprocessed data from disk"""
    X_train = np.load(os.path.join(save_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(save_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(save_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(save_dir, 'y_test.npy'))
    
    with open(os.path.join(save_dir, 'class_names.txt'), 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    return X_train, X_test, y_train, y_test, class_names