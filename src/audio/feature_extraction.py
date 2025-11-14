from typing import Tuple, Dict, List
import warnings

import numpy as np
import librosa
import pandas as pd

# Suppress some librosa warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


def load_audio_file(path_or_file, target_sr: int = 16000, mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Load audio from a filesystem path or a file-like object.
    
    Args:
        path_or_file: File path string or file-like object
        target_sr: Target sample rate for resampling
        mono: Convert to mono if True
    
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        y, sr = librosa.load(path_or_file, sr=target_sr, mono=mono)
        if y.ndim > 1:
            y = np.mean(y, axis=0)
        
        # Normalize audio to prevent clipping
        if len(y) > 0:
            y = y / (np.max(np.abs(y)) + 1e-8)
            
        return y, sr
    except Exception as e:
        raise ValueError(f"Could not load audio file: {str(e)}")


def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mel-scale spectrogram from audio signal.
    
    Args:
        y: Audio signal
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length for STFT
    
    Returns:
        Tuple of (mel_spectrogram_db, time_frames)
    """
    try:
        # Compute mel spectrogram
        S = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=n_mels, 
            n_fft=n_fft, 
            hop_length=hop_length,
            fmax=sr//2  # Nyquist frequency
        )
        
        # Convert to dB scale
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Compute time frames
        times = librosa.frames_to_time(
            np.arange(S_db.shape[1]), 
            sr=sr, 
            hop_length=hop_length
        )
        
        return S_db, times
    except Exception as e:
        raise ValueError(f"Error computing mel spectrogram: {str(e)}")


def extract_frame_features(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
    n_mfcc: int = 13,
) -> Dict[str, np.ndarray]:
    """
    Extract comprehensive frame-level features from audio signal.
    
    Args:
        y: Audio signal
        sr: Sample rate
        frame_length: Frame length for analysis
        hop_length: Hop length between frames
        n_mfcc: Number of MFCC coefficients
    
    Returns:
        Dictionary of extracted features
    """
    try:
        # Time-domain features
        rms = librosa.feature.rms(
            y=y, 
            frame_length=frame_length, 
            hop_length=hop_length
        )[0]
        
        zcr = librosa.feature.zero_crossing_rate(
            y=y, 
            frame_length=frame_length, 
            hop_length=hop_length
        )[0]
        
        # Spectral features
        centroid = librosa.feature.spectral_centroid(
            y=y, 
            sr=sr, 
            n_fft=frame_length, 
            hop_length=hop_length
        )[0]
        
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y, 
            sr=sr, 
            n_fft=frame_length, 
            hop_length=hop_length
        )[0]
        
        rolloff = librosa.feature.spectral_rolloff(
            y=y, 
            sr=sr, 
            n_fft=frame_length, 
            hop_length=hop_length
        )[0]
        
        flatness = librosa.feature.spectral_flatness(
            y=y, 
            n_fft=frame_length, 
            hop_length=hop_length
        )[0]
        
        # Cepstral features
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=n_mfcc, 
            n_fft=frame_length, 
            hop_length=hop_length
        )
        
        # Additional spectral features
        contrast = librosa.feature.spectral_contrast(
            y=y, 
            sr=sr, 
            n_fft=frame_length, 
            hop_length=hop_length
        )
        
        tonnetz = librosa.feature.tonnetz(
            y=librosa.effects.harmonic(y), 
            sr=sr
        )
        
        # Ensure all features have the same number of frames
        min_frames = min(
            rms.shape[0], zcr.shape[0], centroid.shape[0], 
            bandwidth.shape[0], rolloff.shape[0], flatness.shape[0],
            mfcc.shape[1], contrast.shape[1], tonnetz.shape[1]
        )
        
        # Trim all features to the same length
        times = librosa.frames_to_time(
            np.arange(min_frames), 
            sr=sr, 
            hop_length=hop_length
        )
        
        return {
            "times": times,
            "rms": rms[:min_frames],
            "zcr": zcr[:min_frames],
            "centroid": centroid[:min_frames],
            "bandwidth": bandwidth[:min_frames],
            "rolloff": rolloff[:min_frames],
            "flatness": flatness[:min_frames],
            "mfcc": mfcc[:, :min_frames].T,  # shape: (frames, n_mfcc)
            "contrast": contrast[:, :min_frames].T,  # shape: (frames, 7)
            "tonnetz": tonnetz[:, :min_frames].T,  # shape: (frames, 6)
        }
    except Exception as e:
        raise ValueError(f"Error extracting features: {str(e)}")


def build_feature_matrix(features: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Stack selected features per frame to form a 2D matrix: (num_frames, num_features)
    
    Args:
        features: Dictionary of extracted features
    
    Returns:
        Tuple of (feature_matrix, feature_names, frame_times)
    """
    try:
        times = features["times"]
        safe_log = lambda x: np.log(np.maximum(x, 1e-12))

        columns = [
            (safe_log(features["rms"]), "log_rms"),
            (features["zcr"], "zcr"),
            (safe_log(features["centroid"]), "log_centroid"),
            (safe_log(features["bandwidth"]), "log_bandwidth"),
            (safe_log(features["rolloff"]), "log_rolloff"),
            (safe_log(features["flatness"]), "log_flatness"),
        ]

        # Add MFCC coefficients
        mfcc = features["mfcc"]  # (frames, n_mfcc)
        num_mfcc = mfcc.shape[1]
        for i in range(num_mfcc):
            columns.append((mfcc[:, i], f"mfcc_{i+1}"))

        # Add spectral contrast
        contrast = features["contrast"]  # (frames, 7)
        num_contrast = contrast.shape[1]
        for i in range(num_contrast):
            columns.append((contrast[:, i], f"contrast_{i+1}"))

        # Add tonnetz features
        tonnetz = features["tonnetz"]  # (frames, 6)
        num_tonnetz = tonnetz.shape[1]
        for i in range(num_tonnetz):
            columns.append((tonnetz[:, i], f"tonnetz_{i+1}"))

        # Stack all features
        matrix = np.vstack([c[0] for c in columns]).T.astype(np.float32)
        names = [c[1] for c in columns]
        
        # Handle NaN/inf values
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return matrix, names, times
    except Exception as e:
        raise ValueError(f"Error building feature matrix: {str(e)}")


def load_vibration_csv(file_path, time_col='time', signal_cols=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load vibration data from CSV file.
    
    Args:
        file_path: Path to CSV file or file-like object
        time_col: Name of time column
        signal_cols: List of signal column names (if None, use all numeric columns)
    
    Returns:
        Tuple of (time_array, signal_matrix)
    """
    try:
        df = pd.read_csv(file_path)
        
        if signal_cols is None:
            # Use all numeric columns except time
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if time_col in numeric_cols:
                numeric_cols.remove(time_col)
            signal_cols = numeric_cols
        
        time_data = df[time_col].values if time_col in df.columns else np.arange(len(df))
        signal_data = df[signal_cols].values
        
        return time_data, signal_data
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {str(e)}")