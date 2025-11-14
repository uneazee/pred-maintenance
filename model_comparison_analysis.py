"""
Model Comparison Analysis - Configured for your workspace
Run from workspace root: python analyze_models_fixed.py

This script will:
1. Load your 3 trained models from the models/ directory
2. Evaluate them on test data
3. Generate comparison visualizations
4. Create detailed reports
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, List
import warnings

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_models(models_dir='models'):
    """Load all trained models and scaler."""
    models_path = Path(models_dir)
    
    print(f"Looking for models in: {models_path.absolute()}")
    
    if not models_path.exists():
        raise FileNotFoundError(f"Models directory not found: {models_path}")
    
    # Load metadata
    metadata_path = models_path / 'model_metadata.pkl'
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"✓ Loaded metadata - Best model: {metadata['best_model_name']}")
    else:
        metadata = None
        print("⚠ model_metadata.pkl not found")
    
    # Load scaler
    scaler_path = models_path / 'feature_scaler.pkl'
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("✓ Loaded feature scaler")
    else:
        raise FileNotFoundError(f"feature_scaler.pkl not found in {models_path}")
    
    # Load models
    model_files = {
        'Random Forest': 'random_forest.pkl',
        'Gradient Boosting': 'gradient_boosting.pkl',
        'SVM (RBF)': 'svm_rbf.pkl'
    }
    
    models = {}
    for name, filename in model_files.items():
        model_path = models_path / filename
        if model_path.exists():
            with open(model_path, 'rb') as f:
                models[name] = pickle.load(f)
            print(f"✓ Loaded {name}")
        else:
            print(f"✗ {filename} not found")
    
    if not models:
        raise FileNotFoundError("No model files found!")
    
    print(f"\n✓ Successfully loaded {len(models)} models\n")
    
    return models, scaler, metadata


def evaluate_model(model, X_test, y_test):
    """Evaluate a single model and return metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred,
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        metrics['avg_precision'] = average_precision_score(y_test, y_proba)
        metrics['y_proba'] = y_proba
    
    tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['tn'] = tn
    metrics['fp'] = fp
    metrics['fn'] = fn
    metrics['tp'] = tp
    
    return metrics


def compare_models(models, scaler, X_test, y_test):
    """Compare all models and return results dataframe."""
    print("Evaluating models on test set...\n")
    
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    comparison_data = []
    
    for name, model in models.items():
        print(f"  Evaluating {name}...")
        metrics = evaluate_model(model, X_test_scaled, y_test)
        results[name] = metrics
        
        comparison_data.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'Specificity': metrics['specificity'],
            'Sensitivity': metrics['sensitivity'],
            'ROC-AUC': metrics.get('roc_auc', np.nan),
            'Avg Precision': metrics.get('avg_precision', np.nan),
            'True Negatives': metrics['tn'],
            'False Positives': metrics['fp'],
            'False Negatives': metrics['fn'],
            'True Positives': metrics['tp']
        })
    
    df = pd.DataFrame(comparison_data)
    df['Rank (F1)'] = df['F1-Score'].rank(ascending=False).astype(int)
    df['Rank (ROC-AUC)'] = df['ROC-AUC'].rank(ascending=False).astype(int)
    
    return df, results


def plot_confusion_matrices(results, save_path=None):
    """Plot confusion matrices for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (name, metrics) in enumerate(results.items()):
        cm = metrics['confusion_matrix']
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Normal', 'Faulty'],
            yticklabels=['Normal', 'Faulty'],
            ax=axes[idx],
            cbar_kws={'label': 'Count'}
        )
        
        axes[idx].set_title(f'{name}\nAccuracy: {metrics["accuracy"]:.3f}', fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    return fig


def plot_roc_curves(results, y_test, save_path=None):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (name, metrics) in enumerate(results.items()):
        if 'y_proba' in metrics:
            fpr, tpr, _ = roc_curve(y_test, metrics['y_proba'])
            auc = metrics['roc_auc']
            
            ax.plot(
                fpr, tpr, 
                color=colors[idx],
                lw=2,
                label=f'{name} (AUC = {auc:.3f})'
            )
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    return fig


def plot_metrics_comparison(comparison_df, save_path=None):
    """Plot bar charts comparing key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = [
        ('Accuracy', 'Accuracy Score'),
        ('F1-Score', 'F1 Score'),
        ('ROC-AUC', 'ROC-AUC Score'),
        ('Recall', 'Recall (Sensitivity)')
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (metric, title) in enumerate(metrics_to_plot):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        data = comparison_df[['Model', metric]].sort_values(metric, ascending=False)
        bars = ax.bar(range(len(data)), data[metric].values, color=colors)
        
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data['Model'].values, rotation=45, ha='right')
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'Model Comparison: {title}', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    return fig


def generate_report(comparison_df, results, output_path=None):
    """Generate a text report."""
    report = []
    report.append("=" * 80)
    report.append("MODEL COMPARISON REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Summary table
    report.append("PERFORMANCE SUMMARY")
    report.append("-" * 80)
    report.append(comparison_df[['Model', 'Accuracy', 'Precision', 'Recall', 
                                  'F1-Score', 'ROC-AUC']].to_string(index=False))
    report.append("")
    
    # Best performers
    best_f1 = comparison_df.loc[comparison_df['F1-Score'].idxmax()]
    best_auc = comparison_df.loc[comparison_df['ROC-AUC'].idxmax()]
    
    report.append("BEST PERFORMERS")
    report.append("-" * 80)
    report.append(f"  🏆 Best F1-Score:  {best_f1['Model']} ({best_f1['F1-Score']:.4f})")
    report.append(f"  🏆 Best ROC-AUC:   {best_auc['Model']} ({best_auc['ROC-AUC']:.4f})")
    report.append("")
    
    # Detailed metrics
    report.append("DETAILED METRICS")
    report.append("-" * 80)
    
    for name, metrics in results.items():
        report.append("")
        report.append(f"{name}")
        report.append("-" * 40)
        report.append(f"  Accuracy:     {metrics['accuracy']:.4f}")
        report.append(f"  Precision:    {metrics['precision']:.4f}")
        report.append(f"  Recall:       {metrics['recall']:.4f}")
        report.append(f"  F1-Score:     {metrics['f1_score']:.4f}")
        report.append(f"  Specificity:  {metrics['specificity']:.4f}")
        report.append(f"  Sensitivity:  {metrics['sensitivity']:.4f}")
        if 'roc_auc' in metrics:
            report.append(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
        report.append("")
        report.append("  Confusion Matrix:")
        report.append(f"    True Negatives:  {metrics['tn']}")
        report.append(f"    False Positives: {metrics['fp']}")
        report.append(f"    False Negatives: {metrics['fn']}")
        report.append(f"    True Positives:  {metrics['tp']}")
    
    report.append("")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"  ✓ Saved: {output_path}")
    
    return report_text


def main():
    """Main analysis pipeline."""
    
    print("=" * 80)
    print("MODEL COMPARISON AND METRIC ANALYSIS")
    print("=" * 80)
    print()
    
    # Configuration
    MODELS_DIR = 'models'
    OUTPUT_DIR = 'outputs/model_analysis'
    
    # Check if we're in the right directory
    if not Path(MODELS_DIR).exists():
        print("❌ ERROR: 'models' directory not found!")
        print(f"   Current directory: {Path.cwd()}")
        print("   Please run this script from the workspace root directory")
        print()
        print("   cd D:\\workspace")
        print("   python analyze_models_fixed.py")
        return
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_path.absolute()}\n")
    
    # Load models
    try:
        models, scaler, metadata = load_models(MODELS_DIR)
    except Exception as e:
        print(f"❌ ERROR loading models: {e}")
        return
    
    # Check for test data
    test_data_paths = [
        'data/X_test.npy',
        'data/y_test.npy',
        Path('data') / 'X_test.npy',
        Path('data') / 'y_test.npy'
    ]
    
    X_test_path = None
    y_test_path = None
    
    for path in ['data/X_test.npy', Path('data/X_test.npy')]:
        if Path(path).exists():
            X_test_path = path
            break
    
    for path in ['data/y_test.npy', Path('data/y_test.npy')]:
        if Path(path).exists():
            y_test_path = path
            break
    
    if not X_test_path or not y_test_path:
        print("=" * 80)
        print("⚠️  TEST DATA NOT FOUND")
        print("=" * 80)
        print()
        print("To run the full analysis, you need test data files:")
        print("  - data/X_test.npy")
        print("  - data/y_test.npy")
        print()
        print("HOW TO GENERATE TEST DATA:")
        print("=" * 80)
        print()
        print("Option 1: Modify train_model.py")
        print("-" * 80)
        print("Add these lines after the train/test split (around line 423):")
        print()
        print("    # Save test data for analysis")
        print("    np.save('data/X_test.npy', X_test)")
        print("    np.save('data/y_test.npy', y_test)")
        print("    print('✓ Saved test data for analysis')")
        print()
        print("Then re-run: python train_model.py")
        print()
        print("Option 2: Run from Python")
        print("-" * 80)
        print("If you have X_test and y_test variables in memory:")
        print()
        print("    import numpy as np")
        print("    np.save('data/X_test.npy', X_test)")
        print("    np.save('data/y_test.npy', y_test)")
        print()
        print("=" * 80)
        return
    
    # Load test data
    print("Loading test data...")
    try:
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        print(f"✓ Loaded test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        print(f"  Normal samples: {np.sum(y_test == 0)}")
        print(f"  Faulty samples: {np.sum(y_test == 1)}")
        print()
    except Exception as e:
        print(f"❌ ERROR loading test data: {e}")
        return
    
    # Compare models
    print("=" * 80)
    print("COMPARING MODELS")
    print("=" * 80)
    comparison_df, results = compare_models(models, scaler, X_test, y_test)
    
    print("\n")
    print("RESULTS")
    print("=" * 80)
    print(comparison_df[['Model', 'Accuracy', 'Precision', 'Recall', 
                         'F1-Score', 'ROC-AUC', 'Rank (F1)']].to_string(index=False))
    print()
    
    # Generate visualizations
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    plot_confusion_matrices(results, output_path / 'confusion_matrices.png')
    plot_roc_curves(results, y_test, output_path / 'roc_curves.png')
    plot_metrics_comparison(comparison_df, output_path / 'metrics_comparison.png')
    
    plt.close('all')  # Close all figures
    print()
    
    # Generate report
    print("=" * 80)
    print("GENERATING REPORT")
    print("=" * 80)
    report = generate_report(comparison_df, results, output_path / 'comparison_report.txt')
    
    # Save comparison table
    comparison_df.to_csv(output_path / 'comparison_results.csv', index=False)
    print(f"  ✓ Saved: {output_path / 'comparison_results.csv'}")
    print()
    
    # Summary
    print("=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print()
    print(f"All results saved to: {output_path.absolute()}")
    print()
    print("Generated files:")
    print("  - confusion_matrices.png")
    print("  - roc_curves.png")
    print("  - metrics_comparison.png")
    print("  - comparison_report.txt")
    print("  - comparison_results.csv")
    print()
    
    # Best model recommendation
    best_f1 = comparison_df.loc[comparison_df['F1-Score'].idxmax()]
    print("RECOMMENDATION")
    print("=" * 80)
    print(f"🏆 Best Overall Model: {best_f1['Model']}")
    print(f"   F1-Score: {best_f1['F1-Score']:.4f}")
    print(f"   ROC-AUC:  {best_f1['ROC-AUC']:.4f}")
    print(f"   Accuracy: {best_f1['Accuracy']:.4f}")
    print()


if __name__ == '__main__':
    main()