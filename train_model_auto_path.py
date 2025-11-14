"""
Train supervised models on MIMII dataset for predictive maintenance.
UPDATED: Handles MIMII structure with id_XX subdirectories + Auto path detection

This script:
1. Loads audio files from the MIMII dataset (with id_XX structure)
2. Extracts features using feature_extraction.py
3. Trains classification models
4. Saves trained models for use in the Streamlit app
"""

import os
import glob
from pathlib import Path
from typing import Tuple, List
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# Add src directory to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from audio.feature_extraction import load_audio_file, extract_frame_features, build_feature_matrix

warnings.filterwarnings('ignore')


class MIMIIDatasetLoader:
    """Load and prepare MIMII dataset for training."""
    
    def __init__(self, dataset_root: str, machine_type: str = 'fan'):
        """
        Initialize dataset loader.
        
        Args:
            dataset_root: Path to MIMII dataset root directory
            machine_type: Type of machine ('fan' or 'pump')
        """
        self.dataset_root = Path(dataset_root)
        self.machine_type = machine_type
        self.machine_path = self.dataset_root / machine_type
        
        if not self.machine_path.exists():
            raise ValueError(f"Machine path does not exist: {self.machine_path}")
    
    def load_dataset(
        self, 
        target_sr: int = 16000,
        max_samples_per_class: int = None,
        id_list: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load all audio files and extract features.
        Handles MIMII structure: machine_type/id_XX/normal|abnormal/*.wav
        
        Args:
            target_sr: Target sample rate
            max_samples_per_class: Maximum samples per class (for quick testing)
            id_list: List of IDs to use (e.g., ['id_00', 'id_02']). If None, uses all.
        
        Returns:
            Tuple of (features, labels, file_paths)
        """
        print(f"\nLoading {self.machine_type} dataset from {self.machine_path}")
        
        # Find all id_XX directories
        id_dirs = sorted([d for d in self.machine_path.iterdir() 
                         if d.is_dir() and d.name.startswith('id_')])
        
        if not id_dirs:
            raise ValueError(f"No id_XX directories found in {self.machine_path}")
        
        print(f"Found ID directories: {[d.name for d in id_dirs]}")
        
        # Filter by id_list if provided
        if id_list is not None:
            id_dirs = [d for d in id_dirs if d.name in id_list]
            print(f"Using only: {[d.name for d in id_dirs]}")
        
        # Collect all audio files
        normal_files = []
        abnormal_files = []
        
        for id_dir in id_dirs:
            normal_dir = id_dir / 'normal'
            abnormal_dir = id_dir / 'abnormal'
            
            if normal_dir.exists():
                files = sorted(glob.glob(str(normal_dir / '*.wav')))
                normal_files.extend(files)
                print(f"  {id_dir.name}/normal: {len(files)} files")
            else:
                print(f"  ⚠️ {id_dir.name}/normal: NOT FOUND")
            
            if abnormal_dir.exists():
                files = sorted(glob.glob(str(abnormal_dir / '*.wav')))
                abnormal_files.extend(files)
                print(f"  {id_dir.name}/abnormal: {len(files)} files")
            else:
                print(f"  ⚠️ {id_dir.name}/abnormal: NOT FOUND")
        
        print(f"\nTotal normal files: {len(normal_files)}")
        print(f"Total abnormal files: {len(abnormal_files)}")
        
        if len(normal_files) == 0 or len(abnormal_files) == 0:
            raise ValueError("Not enough data! Need both normal and abnormal files.")
        
        # Limit samples if specified
        if max_samples_per_class:
            normal_files = normal_files[:max_samples_per_class]
            abnormal_files = abnormal_files[:max_samples_per_class]
            print(f"\nLimited to {len(normal_files)} normal and {len(abnormal_files)} abnormal files")
        
        # Process all files
        all_features = []
        all_labels = []
        all_paths = []
        
        # Process normal files
        print("\n" + "="*60)
        print("PROCESSING NORMAL FILES")
        print("="*60)
        for i, file_path in enumerate(normal_files):
            if (i + 1) % 20 == 0 or i == 0:
                print(f"Processing normal file {i+1}/{len(normal_files)}")
            
            try:
                features = self._extract_file_features(file_path, target_sr)
                all_features.append(features)
                all_labels.append(0)  # 0 = Normal
                all_paths.append(file_path)
            except Exception as e:
                print(f"  ⚠️ Error processing {Path(file_path).name}: {e}")
        
        # Process abnormal files
        print("\n" + "="*60)
        print("PROCESSING ABNORMAL FILES")
        print("="*60)
        for i, file_path in enumerate(abnormal_files):
            if (i + 1) % 20 == 0 or i == 0:
                print(f"Processing abnormal file {i+1}/{len(abnormal_files)}")
            
            try:
                features = self._extract_file_features(file_path, target_sr)
                all_features.append(features)
                all_labels.append(1)  # 1 = Abnormal/Faulty
                all_paths.append(file_path)
            except Exception as e:
                print(f"  ⚠️ Error processing {Path(file_path).name}: {e}")
        
        # Convert to arrays
        X = np.vstack(all_features)
        y = np.array(all_labels)
        
        print("\n" + "="*60)
        print("DATASET LOADING COMPLETE")
        print("="*60)
        print(f"Total samples: {len(y)}")
        print(f"Normal samples: {np.sum(y == 0)}")
        print(f"Abnormal samples: {np.sum(y == 1)}")
        print(f"Feature dimensions: {X.shape}")
        print(f"Class balance: {np.sum(y == 0)/len(y)*100:.1f}% normal, {np.sum(y == 1)/len(y)*100:.1f}% abnormal")
        
        return X, y, all_paths
    
    def _extract_file_features(self, file_path: str, target_sr: int) -> np.ndarray:
        """
        Extract statistical features from a single audio file.
        
        Args:
            file_path: Path to audio file
            target_sr: Target sample rate
        
        Returns:
            Feature vector for the file
        """
        # Load audio
        y, sr = load_audio_file(file_path, target_sr=target_sr)
        
        # Extract frame-level features
        features_dict = extract_frame_features(y, sr)
        
        # Build feature matrix
        feature_matrix, feature_names, times = build_feature_matrix(features_dict)
        
        # Compute statistical summaries across time
        # This converts (frames, features) -> (features,)
        features_mean = np.mean(feature_matrix, axis=0)
        features_std = np.std(feature_matrix, axis=0)
        features_max = np.max(feature_matrix, axis=0)
        features_min = np.min(feature_matrix, axis=0)
        features_median = np.median(feature_matrix, axis=0)
        
        # Combine all statistics
        combined_features = np.concatenate([
            features_mean,
            features_std,
            features_max,
            features_min,
            features_median
        ])
        
        return combined_features


class ModelTrainer:
    """Train and evaluate classification models."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_scaler = None
    
    def train_models(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ):
        """
        Train multiple models and select the best one.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
        """
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models to train
        models_to_train = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42
            ),
            'SVM (RBF)': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=42
            )
        }
        
        best_score = 0
        
        for name, model in models_to_train.items():
            print(f"\n{'-'*60}")
            print(f"Training {name}...")
            print(f"{'-'*60}")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate on test set
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = np.mean(y_pred == y_test)
            
            print(f"\n{name} Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            
            if y_proba is not None:
                auc = roc_auc_score(y_test, y_proba)
                print(f"  ROC-AUC: {auc:.4f}")
            
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Normal', 'Faulty']))
            
            print("\nConfusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            print(f"  True Negatives:  {cm[0,0]}")
            print(f"  False Positives: {cm[0,1]}")
            print(f"  False Negatives: {cm[1,0]}")
            print(f"  True Positives:  {cm[1,1]}")
            
            # Store model
            self.models[name] = model
            
            # Track best model
            if accuracy > best_score:
                best_score = accuracy
                self.best_model = model
                self.best_model_name = name
        
        print("\n" + "="*60)
        print(f"🏆 BEST MODEL: {self.best_model_name} (Accuracy: {best_score:.4f})")
        print("="*60)
    
    def save_models(self, save_dir: str = 'models'):
        """
        Save all trained models to disk.
        
        Args:
            save_dir: Directory to save models
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n" + "="*60)
        print(f"SAVING MODELS TO {save_path}")
        print("="*60)
        
        # Save scaler
        scaler_path = save_path / 'feature_scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Saved scaler: {scaler_path}")
        
        # Save all models
        for name, model in self.models.items():
            model_filename = name.lower().replace(' ', '_').replace('(', '').replace(')', '') + '.pkl'
            model_path = save_path / model_filename
            joblib.dump(model, model_path)
            print(f"✅ Saved {name}: {model_path}")
        
        # Save best model separately
        best_model_path = save_path / 'best_model.pkl'
        joblib.dump(self.best_model, best_model_path)
        print(f"✅ Saved best model: {best_model_path}")
        
        # Save metadata
        metadata = {
            'best_model_name': self.best_model_name,
            'models_trained': list(self.models.keys()),
            'feature_scaler': str(scaler_path),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        metadata_path = save_path / 'model_metadata.pkl'
        joblib.dump(metadata, metadata_path)
        print(f"✅ Saved metadata: {metadata_path}")
        
        print("\n" + "="*60)
        print("ALL MODELS SAVED SUCCESSFULLY")
        print("="*60)


def main():
    """Main training pipeline."""
    
    # Auto-detect dataset path
    print("="*60)
    print("PREDICTIVE MAINTENANCE MODEL TRAINING")
    print("="*60)
    
    # Try multiple possible paths
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    possible_paths = [
        'data/mimii_dataset',  # From workspace directory
        'workspace/data/mimii_dataset',  # From parent directory
        os.path.join(current_dir, 'data', 'mimii_dataset'),  # Absolute path
        r'D:\workspace\data\mimii_dataset',  # Windows absolute
    ]
    
    DATASET_ROOT = None
    for path in possible_paths:
        if os.path.exists(path):
            DATASET_ROOT = path
            print(f"✅ Found dataset at: {DATASET_ROOT}")
            break
    
    if DATASET_ROOT is None:
        print("\n❌ ERROR: Could not find MIMII dataset!")
        print("\nTried these paths:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nPlease check your dataset location and update the script.")
        return
    
    # Configuration
    MACHINE_TYPE = 'fan'  # Change to 'pump' for pump dataset
    TARGET_SR = 16000
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # For quick testing, use a subset of IDs
    # Set to None to use all IDs
    USE_IDS = None  # e.g., ['id_00', 'id_02'] for quick test, or None for all
    
    # For quick testing, limit samples per class
    # Set to None to use all samples
    MAX_SAMPLES = None  # e.g., 50 for quick test, or None for all
    
    print(f"Machine Type: {MACHINE_TYPE}")
    print(f"Sample Rate: {TARGET_SR} Hz")
    print(f"Test Size: {TEST_SIZE * 100}%")
    if USE_IDS:
        print(f"Using IDs: {USE_IDS}")
    else:
        print("Using: ALL IDs")
    if MAX_SAMPLES:
        print(f"Max samples per class: {MAX_SAMPLES}")
    else:
        print("Max samples: ALL available")
    
    # Load dataset
    try:
        loader = MIMIIDatasetLoader(DATASET_ROOT, MACHINE_TYPE)
        X, y, file_paths = loader.load_dataset(
            target_sr=TARGET_SR,
            max_samples_per_class=MAX_SAMPLES,
            id_list=USE_IDS
        )
    except Exception as e:
        print(f"\n❌ ERROR loading dataset: {e}")
        print("\nPlease check:")
        print("1. Dataset path is correct")
        print("2. Dataset has structure: mimii_dataset/fan/id_XX/normal|abnormal/*.wav")
        print("3. Audio files exist in normal and abnormal directories")
        return
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"\n" + "="*60)
    print("TRAIN/TEST SPLIT")
    print("="*60)
    print(f"Training samples: {len(y_train)} (Normal: {np.sum(y_train==0)}, Faulty: {np.sum(y_train==1)})")
    print(f"Test samples: {len(y_test)} (Normal: {np.sum(y_test==0)}, Faulty: {np.sum(y_test==1)})")
    
    # Train models
    trainer = ModelTrainer()
    trainer.train_models(X_train, y_train, X_test, y_test)
    
    # Save models
    trainer.save_models()
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: streamlit run app/streamlit_app.py")
    print("2. In the app:")
    print("   - Check 'Use Trained MIMII Models'")
    print("   - Set directory to 'models'")
    print("   - Click 'Load Models'")
    print("   - Upload a .wav file to test")
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
