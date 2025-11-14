"""
Diagnostic script to check if your predictive maintenance project is set up correctly.

Run this from the workspace directory:
    cd workspace
    python check_setup.py
"""

import os
import sys
from pathlib import Path

def check_directory_structure():
    """Check if the expected directory structure exists."""
    print("\n" + "="*60)
    print("CHECKING DIRECTORY STRUCTURE")
    print("="*60)
    
    expected_dirs = [
        'app',
        'src/audio',
        'src/models',
        'src/plots',
        'data/mimii_dataset',
        'models',
        'notebooks'
    ]
    
    all_good = True
    for dir_path in expected_dirs:
        full_path = Path(dir_path)
        if full_path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - NOT FOUND")
            all_good = False
    
    return all_good


def check_mimii_dataset():
    """Check if MIMII dataset is properly organized."""
    print("\n" + "="*60)
    print("CHECKING MIMII DATASET")
    print("="*60)
    
    dataset_root = Path('data/mimii_dataset')
    
    if not dataset_root.exists():
        print(f"❌ Dataset root not found: {dataset_root}")
        print("   Expected location: workspace/data/mimii_dataset/")
        return False
    
    machine_types = []
    for item in dataset_root.iterdir():
        if item.is_dir():
            machine_types.append(item.name)
    
    print(f"\nMachine types found: {machine_types}")
    
    all_good = True
    for machine_type in machine_types:
        machine_path = dataset_root / machine_type
        normal_path = machine_path / 'normal'
        abnormal_path = machine_path / 'abnormal'
        
        print(f"\n📁 {machine_type}/")
        
        if normal_path.exists():
            normal_files = list(normal_path.glob('*.wav'))
            print(f"  ✅ normal/ - {len(normal_files)} .wav files")
        else:
            print(f"  ❌ normal/ - NOT FOUND")
            all_good = False
        
        if abnormal_path.exists():
            abnormal_files = list(abnormal_path.glob('*.wav'))
            print(f"  ✅ abnormal/ - {len(abnormal_files)} .wav files")
        else:
            print(f"  ❌ abnormal/ - NOT FOUND")
            all_good = False
    
    return all_good


def check_python_files():
    """Check if required Python files exist."""
    print("\n" + "="*60)
    print("CHECKING PYTHON FILES")
    print("="*60)
    
    expected_files = {
        'app/streamlit_app.py': 'Main Streamlit application',
        'src/audio/feature_extraction.py': 'Audio feature extraction',
        'src/models/baseline.py': 'Anomaly detection models',
        'src/plots/plots.py': 'Visualization utilities',
    }
    
    all_good = True
    for file_path, description in expected_files.items():
        full_path = Path(file_path)
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"✅ {file_path}")
            print(f"   {description} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_good = False
    
    return all_good


def check_trained_models():
    """Check if any trained models exist."""
    print("\n" + "="*60)
    print("CHECKING TRAINED MODELS")
    print("="*60)
    
    models_dir = Path('models')
    
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        print("   This is OK if you haven't trained models yet")
        return False
    
    model_files = list(models_dir.glob('*.pkl'))
    
    if not model_files:
        print(f"⚠️  No .pkl files found in {models_dir}")
        print("   You need to run train_model.py to create models")
        return False
    
    print(f"\nFound {len(model_files)} model files:")
    for model_file in model_files:
        size = model_file.stat().st_size
        print(f"  ✅ {model_file.name} ({size:,} bytes)")
    
    # Check for required files
    required_files = ['feature_scaler.pkl', 'best_model.pkl']
    missing_files = []
    
    for required_file in required_files:
        if not (models_dir / required_file).exists():
            missing_files.append(required_file)
    
    if missing_files:
        print(f"\n⚠️  Missing required files: {missing_files}")
        print("   Run train_model.py to generate these files")
        return False
    
    return True


def check_dependencies():
    """Check if required Python packages are installed."""
    print("\n" + "="*60)
    print("CHECKING PYTHON DEPENDENCIES")
    print("="*60)
    
    required_packages = [
        'streamlit',
        'numpy',
        'pandas',
        'scipy',
        'librosa',
        'soundfile',
        'matplotlib',
        'plotly',
        'scikit-learn',
        'joblib'
    ]
    
    all_good = True
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            all_good = False
    
    if not all_good:
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
    
    return all_good


def print_next_steps(results):
    """Print recommended next steps based on check results."""
    print("\n" + "="*60)
    print("RECOMMENDED NEXT STEPS")
    print("="*60)
    
    if not results['dependencies']:
        print("\n1️⃣  INSTALL DEPENDENCIES (CRITICAL)")
        print("   cd workspace")
        print("   pip install -r requirements.txt")
    
    if not results['dataset']:
        print("\n2️⃣  ORGANIZE MIMII DATASET (CRITICAL)")
        print("   Expected structure:")
        print("   workspace/data/mimii_dataset/")
        print("   ├── fan/")
        print("   │   ├── normal/*.wav")
        print("   │   └── abnormal/*.wav")
        print("   └── pump/")
        print("       ├── normal/*.wav")
        print("       └── abnormal/*.wav")
    
    if not results['models']:
        print("\n3️⃣  TRAIN MODELS (THIS IS YOUR ISSUE!)")
        print("   cd workspace")
        print("   python train_model.py")
        print("   ")
        print("   This will:")
        print("   - Load your MIMII dataset")
        print("   - Extract features from all audio files")
        print("   - Train multiple classifiers")
        print("   - Save models to workspace/models/")
    
    if results['models']:
        print("\n✅ MODELS ARE READY!")
        print("   Run the Streamlit app:")
        print("   cd workspace")
        print("   streamlit run app/streamlit_app.py")
        print("   ")
        print("   Then in the app:")
        print("   1. Check 'Use Trained MIMII Models'")
        print("   2. Set directory to 'models'")
        print("   3. Click 'Load Models'")
        print("   4. Upload audio file")


def main():
    """Run all diagnostic checks."""
    print("\n🔍 PREDICTIVE MAINTENANCE PROJECT DIAGNOSTICS")
    print("Current directory:", os.getcwd())
    
    results = {
        'structure': check_directory_structure(),
        'dataset': check_mimii_dataset(),
        'python_files': check_python_files(),
        'models': check_trained_models(),
        'dependencies': check_dependencies()
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    status_emoji = {
        True: "✅",
        False: "❌"
    }
    
    for check_name, status in results.items():
        emoji = status_emoji[status]
        print(f"{emoji} {check_name.replace('_', ' ').title()}: {'PASSED' if status else 'FAILED'}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED!")
        print("Your setup is complete and ready to use.")
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("Follow the recommended steps below to fix issues.")
        print_next_steps(results)
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()