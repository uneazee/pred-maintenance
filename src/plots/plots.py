from typing import Optional
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import librosa.display

# Set matplotlib style for better plots
plt.style.use('default')
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def plot_mel_spectrogram(
    S_db: np.ndarray, 
    sr: int, 
    hop_length: int, 
    title: str = "Mel-Spectrogram (dB)"
):
    """
    Plot mel-scale spectrogram with proper formatting.
    
    Args:
        S_db: Mel spectrogram in dB
        sr: Sample rate
        hop_length: Hop length used for STFT
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    
    try:
        img = librosa.display.specshow(
            S_db,
            sr=sr,
            hop_length=hop_length,
            x_axis="time",
            y_axis="mel",
            cmap="viridis",
            ax=ax,
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Mel Frequency", fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(img, ax=ax, format="%+2.0f dB")
        cbar.set_label("Power (dB)", fontsize=11)
        
        # Improve grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
    except Exception as e:
        ax.text(0.5, 0.5, f"Error plotting spectrogram: {str(e)}", 
                ha='center', va='center', transform=ax.transAxes)
    
    return fig


def plot_anomaly_score(
    times_s: np.ndarray, 
    scores: np.ndarray, 
    threshold: float = 3.0,
    title: str = "Anomaly Score"
):
    """
    Plot anomaly scores over time with threshold line and colored regions.
    
    Args:
        times_s: Time array in seconds
        scores: Anomaly scores
        threshold: Threshold for anomaly detection
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    
    try:
        # Plot main anomaly score line
        line = ax.plot(times_s, scores, color="tab:blue", linewidth=2, 
                      label="Anomaly Score", alpha=0.8)
        
        # Add threshold line
        ax.axhline(y=threshold, color="tab:red", linestyle="--", linewidth=2, 
                  label=f"Threshold ({threshold:.1f})", alpha=0.8)
        
        # Color regions above threshold
        above_threshold = scores >= threshold
        if np.any(above_threshold):
            ax.fill_between(times_s, scores, threshold, 
                           where=above_threshold, 
                           color="red", alpha=0.2, 
                           label="Anomalous regions")
        
        # Color normal regions
        ax.fill_between(times_s, 0, scores, 
                       where=~above_threshold, 
                       color="green", alpha=0.1, 
                       label="Normal regions")
        
        # Formatting
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Anomaly Score", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=10)
        
        # Set y-axis limits for better visualization
        y_max = max(np.nanmax(scores), threshold) * 1.1
        ax.set_ylim(0, y_max)
        
        # Add statistics text box
        max_score = np.nanmax(scores)
        mean_score = np.nanmean(scores)
        anomaly_pct = np.mean(scores >= threshold) * 100
        
        stats_text = f"Max: {max_score:.2f}\nMean: {mean_score:.2f}\nAnomaly %: {anomaly_pct:.1f}%"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8), fontsize=9)
        
    except Exception as e:
        ax.text(0.5, 0.5, f"Error plotting anomaly scores: {str(e)}", 
                ha='center', va='center', transform=ax.transAxes)
    
    return fig


def plot_feature_importance(
    feature_names: list, 
    importance_scores: np.ndarray,
    title: str = "Feature Importance"
):
    """
    Plot feature importance scores as horizontal bar chart.
    
    Args:
        feature_names: List of feature names
        importance_scores: Importance scores for each feature
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, max(6, len(feature_names) * 0.3)), 
                          constrained_layout=True)
    
    try:
        # Sort features by importance
        sorted_idx = np.argsort(importance_scores)
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_scores = importance_scores[sorted_idx]
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(sorted_features)), sorted_scores, 
                      color='steelblue', alpha=0.7)
        
        # Customize plot
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features, fontsize=10)
        ax.set_xlabel("Importance Score", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            ax.text(score + 0.01 * max(sorted_scores), bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', ha='left', va='center', fontsize=8)
        
    except Exception as e:
        ax.text(0.5, 0.5, f"Error plotting feature importance: {str(e)}", 
                ha='center', va='center', transform=ax.transAxes)
    
    return fig


def plot_health_trend(
    times_s: np.ndarray,
    health_scores: np.ndarray,
    title: str = "Machine Health Trend"
):
    """
    Plot machine health trend over time.
    
    Args:
        times_s: Time array in seconds
        health_scores: Health scores (0-100)
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    
    try:
        # Plot health trend
        ax.plot(times_s, health_scores, color='green', linewidth=2, 
               marker='o', markersize=3, alpha=0.8)
        
        # Add health zones
        ax.axhspan(80, 100, alpha=0.1, color='green', label='Healthy (80-100%)')
        ax.axhspan(60, 80, alpha=0.1, color='yellow', label='Warning (60-80%)')
        ax.axhspan(0, 60, alpha=0.1, color='red', label='Critical (0-60%)')
        
        # Formatting
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Health Score (%)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', fontsize=10)
        
        # Add trend line if enough points
        if len(health_scores) > 10:
            z = np.polyfit(times_s, health_scores, 1)
            p = np.poly1d(z)
            ax.plot(times_s, p(times_s), "--", color='red', alpha=0.7, 
                   linewidth=1, label=f'Trend (slope: {z[0]:.2f})')
            ax.legend(loc='lower left', fontsize=10)
        
    except Exception as e:
        ax.text(0.5, 0.5, f"Error plotting health trend: {str(e)}", 
                ha='center', va='center', transform=ax.transAxes)
    
    return fig


def plot_feature_correlation_matrix(
    feature_matrix: np.ndarray,
    feature_names: list,
    title: str = "Feature Correlation Matrix"
):
    """
    Plot correlation matrix of features.
    
    Args:
        feature_matrix: Feature matrix (frames, features)
        feature_names: List of feature names
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    
    try:
        # Compute correlation matrix
        corr_matrix = np.corrcoef(feature_matrix.T)
        
        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, 
                      aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation', fontsize=12)
        
        # Set ticks and labels
        ax.set_xticks(range(len(feature_names)))
        ax.set_yticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(feature_names, fontsize=8)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add correlation values to cells
        if len(feature_names) <= 20:  # Only for smaller matrices
            for i in range(len(feature_names)):
                for j in range(len(feature_names)):
                    text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.5 else "white",
                                 fontsize=6)
        
    except Exception as e:
        ax.text(0.5, 0.5, f"Error plotting correlation matrix: {str(e)}", 
                ha='center', va='center', transform=ax.transAxes)
    
    return fig


def plot_time_series_features(
    times_s: np.ndarray,
    feature_matrix: np.ndarray,
    feature_names: list,
    selected_features: Optional[list] = None,
    title: str = "Feature Time Series"
):
    """
    Plot selected features over time.
    
    Args:
        times_s: Time array in seconds
        feature_matrix: Feature matrix (frames, features)
        feature_names: List of feature names
        selected_features: List of feature indices to plot (if None, plot first 6)
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    if selected_features is None:
        selected_features = list(range(min(6, len(feature_names))))
    
    n_features = len(selected_features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows), 
                            constrained_layout=True)
    
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    try:
        for i, feat_idx in enumerate(selected_features):
            ax = axes[i]
            
            feature_data = feature_matrix[:, feat_idx]
            ax.plot(times_s, feature_data, linewidth=1.5, color='tab:blue')
            
            ax.set_title(feature_names[feat_idx], fontsize=12, fontweight='bold')
            ax.set_xlabel("Time (s)", fontsize=10)
            ax.set_ylabel("Value", fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add mean line
            mean_val = np.mean(feature_data)
            ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.6, 
                      label=f'Mean: {mean_val:.3f}')
            ax.legend(fontsize=8)
        
        # Hide unused subplots
        for i in range(len(selected_features), len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
    except Exception as e:
        axes[0].text(0.5, 0.5, f"Error plotting features: {str(e)}", 
                    ha='center', va='center', transform=axes[0].transAxes)
    
    return fig


def plot_anomaly_detection_summary(
    times_s: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    health_score: float,
    title: str = "Anomaly Detection Summary"
):
    """
    Create a comprehensive summary plot with multiple visualizations.
    
    Args:
        times_s: Time array in seconds
        scores: Anomaly scores
        threshold: Detection threshold
        health_score: Overall health score
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    try:
        # Main anomaly score plot
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(times_s, scores, color="tab:blue", linewidth=2, alpha=0.8)
        ax1.axhline(y=threshold, color="tab:red", linestyle="--", linewidth=2, alpha=0.8)
        ax1.fill_between(times_s, scores, threshold, where=(scores >= threshold), 
                        color="red", alpha=0.2)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Anomaly Score")
        ax1.set_title("Anomaly Score Timeline", fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Score distribution histogram
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(threshold, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel("Anomaly Score")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Score Distribution", fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Health gauge
        ax3 = fig.add_subplot(gs[1, 1])
        colors = ['red' if health_score < 60 else 'yellow' if health_score < 80 else 'green']
        wedge = patches.Wedge((0, 0), 1, 0, health_score * 3.6, facecolor=colors[0], alpha=0.7)
        ax3.add_patch(wedge)
        circle = plt.Circle((0, 0), 0.6, facecolor='white')
        ax3.add_patch(circle)
        ax3.text(0, 0, f'{health_score:.1f}%', ha='center', va='center', 
                fontsize=16, fontweight='bold')
        ax3.set_xlim(-1.2, 1.2)
        ax3.set_ylim(-1.2, 1.2)
        ax3.set_aspect('equal')
        ax3.set_title("Health Score", fontweight='bold')
        ax3.axis('off')
        
        # Statistics table
        ax4 = fig.add_subplot(gs[1, 2])
        stats_data = [
            ['Max Score', f'{np.max(scores):.3f}'],
            ['Mean Score', f'{np.mean(scores):.3f}'],
            ['Std Score', f'{np.std(scores):.3f}'],
            ['Threshold', f'{threshold:.3f}'],
            ['Anomaly %', f'{np.mean(scores >= threshold)*100:.1f}%'],
            ['Health Score', f'{health_score:.1f}%']
        ]
        
        table = ax4.table(cellText=stats_data, 
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax4.axis('off')
        ax4.set_title("Statistics", fontweight='bold')
        
        # Score trend (if enough data)
        ax5 = fig.add_subplot(gs[2, :])
        if len(scores) > 10:
            window_size = min(20, len(scores) // 4)
            rolling_mean = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
            rolling_times = times_s[window_size-1:]
            
            ax5.plot(times_s, scores, alpha=0.3, color='gray', label='Raw scores')
            ax5.plot(rolling_times, rolling_mean, linewidth=2, color='blue', label=f'Rolling mean ({window_size})')
            ax5.axhline(threshold, color='red', linestyle='--', alpha=0.7, label='Threshold')
            ax5.legend()
        else:
            ax5.plot(times_s, scores, linewidth=2, color='blue')
            ax5.axhline(threshold, color='red', linestyle='--', alpha=0.7)
        
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Anomaly Score")
        ax5.set_title("Score Trend Analysis", fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=18, fontweight='bold')
        
    except Exception as e:
        fig.text(0.5, 0.5, f"Error creating summary plot: {str(e)}", 
                ha='center', va='center', fontsize=14)
    
    return fig