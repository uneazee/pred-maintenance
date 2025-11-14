"""
Alternative model loader that tries multiple loading methods
Use this if pickle fails
"""

import pickle
import joblib
from pathlib import Path
import numpy as np

def load_model_safe(file_path):
    """
    Try multiple methods to load a model file.
    Returns (success, model, error_message)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return False, None, "File not found"
    
    # Method 1: Try joblib (recommended for sklearn)
    try:
        model = joblib.load(file_path)
        return True, model, None
    except Exception as e1:
        error_joblib = str(e1)
    
    # Method 2: Try pickle
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return True, model, None
    except Exception as e2:
        error_pickle = str(e2)
    
    # Method 3: Try pickle with different protocols
    for protocol in [4, 3, 2]:
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f, encoding='latin1')
            return True, model, None
        except:
            pass
    
    # All methods failed
    error_msg = f"joblib: {error_joblib}, pickle: {error_pickle}"
    return False, None, error_msg


def load_all_models(models_dir='models'):
    """Load all models using the safest method."""
    models_path = Path(models_dir)
    
    print("=" * 80)
    print("LOADING MODELS (SAFE MODE)")
    print("=" * 80)
    print()
    
    if not models_path.exists():
        print(f"❌ Models directory not found: {models_path}")
        return None, None, None
    
    results = {}
    
    # Load metadata
    print("Loading metadata...")
    success, metadata, error = load_model_safe(models_path / 'model_metadata.pkl')
    if success:
        print(f"  ✅ model_metadata.pkl")
        print(f"     Best model: {metadata['best_model_name']}")
    else:
        print(f"  ⚠️  model_metadata.pkl - {error}")
        metadata = None
    
    # Load scaler
    print("\nLoading feature scaler...")
    success, scaler, error = load_model_safe(models_path / 'feature_scaler.pkl')
    if success:
        print(f"  ✅ feature_scaler.pkl")
    else:
        print(f"  ❌ feature_scaler.pkl - {error}")
        print("     Cannot proceed without scaler!")
        return None, None, None
    
    # Load models
    print("\nLoading models...")
    model_files = {
        'Random Forest': 'random_forest.pkl',
        'Gradient Boosting': 'gradient_boosting.pkl',
        'SVM (RBF)': 'svm_rbf.pkl'
    }
    
    models = {}
    for name, filename in model_files.items():
        success, model, error = load_model_safe(models_path / filename)
        if success:
            models[name] = model
            print(f"  ✅ {name:25s} from {filename}")
        else:
            print(f"  ❌ {name:25s} - {error}")
    
    if not models:
        print("\n❌ No models could be loaded!")
        return None, None, None
    
    print(f"\n✅ Successfully loaded {len(models)} models")
    print()
    
    return models, scaler, metadata


def check_sklearn_version():
    """Check if sklearn versions are compatible."""
    import sklearn
    print("=" * 80)
    print("VERSION CHECK")
    print("=" * 80)
    print(f"scikit-learn version: {sklearn.__version__}")
    
    version_parts = sklearn.__version__.split('.')
    major = int(version_parts[0])
    minor = int(version_parts[1])
    
    if major == 0 and minor < 24:
        print("⚠️  Warning: Old scikit-learn version")
        print("   Models trained with newer versions may not load")
        print("   Consider: pip install --upgrade scikit-learn")
    elif major >= 1:
        print("✅ Modern scikit-learn version")
    
    print()


def main():
    """Test loading all models."""
    
    check_sklearn_version()
    
    models, scaler, metadata = load_all_models('models')
    
    if models is None:
        print("\n" + "=" * 80)
        print("TROUBLESHOOTING STEPS")
        print("=" * 80)
        print()
        print("1. Check if models were trained with a different Python/sklearn version")
        print("2. Try re-training models with your current environment:")
        print("   python train_model.py")
        print()
        print("3. If you have the original training environment, you can:")
        print("   - Re-export models using joblib.dump()")
        print("   - Create a virtual environment matching the training environment")
        print()
        print("4. Check if model files are corrupted:")
        print("   python diagnose_models.py")
        print()
        return
    
    # Test the models
    print("=" * 80)
    print("MODEL INFO")
    print("=" * 80)
    
    for name, model in models.items():
        print(f"\n{name}:")
        print(f"  Type: {type(model).__name__}")
        print(f"  Module: {type(model).__module__}")
        
        # Try to get model info
        if hasattr(model, 'n_estimators'):
            print(f"  n_estimators: {model.n_estimators}")
        if hasattr(model, 'max_depth'):
            print(f"  max_depth: {model.max_depth}")
        if hasattr(model, 'C'):
            print(f"  C: {model.C}")
        if hasattr(model, 'kernel'):
            print(f"  kernel: {model.kernel}")
    
    print("\n" + "=" * 80)
    print("✅ ALL MODELS LOADED SUCCESSFULLY")
    print("=" * 80)
    print("\nYou can now use these models for analysis")
    print()


if __name__ == '__main__':
    main()
