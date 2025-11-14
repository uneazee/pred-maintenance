from typing import Tuple, Optional
import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """
    Apply moving average smoothing to a 1D array.
    
    Args:
        values: Input array
        window: Window size for smoothing
    
    Returns:
        Smoothed array
    """
    if window <= 1 or len(values) == 0:
        return values
    
    window = int(min(window, len(values)))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(values, (window - 1, 0), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def compute_robust_anomaly_scores(
    feature_matrix: np.ndarray, 
    smoothing_window: int = 5, 
    eps: float = 1e-8
) -> np.ndarray:
    """
    Compute per-frame anomaly scores using robust z-scores per feature
    then averaging across features.
    
    Args:
        feature_matrix: Feature matrix (num_frames, num_features)
        smoothing_window: Window size for temporal smoothing
        eps: Small epsilon to avoid division by zero
    
    Returns:
        Anomaly scores per frame
    """
    if feature_matrix.size == 0:
        return np.array([])
    
    try:
        # Use median and MAD for robust statistics
        med = np.nanmedian(feature_matrix, axis=0)
        mad = np.nanmedian(np.abs(feature_matrix - med), axis=0)
        mad = np.maximum(mad, eps)

        # Compute robust z-scores
        robust_z = np.abs((feature_matrix - med) / mad)
        
        # Average across features to get per-frame score
        scores = np.nanmean(robust_z, axis=1).astype(np.float32)
        
        # Apply temporal smoothing
        scores = _moving_average(scores, smoothing_window)
        
        return scores
    except Exception as e:
        print(f"Warning: Error in robust anomaly scoring: {e}")
        return np.zeros(feature_matrix.shape[0])


def compute_isolation_forest_scores(
    feature_matrix: np.ndarray,
    contamination: float = 0.1,
    random_state: int = 42
) -> np.ndarray:
    """
    Compute anomaly scores using Isolation Forest.
    
    Args:
        feature_matrix: Feature matrix (num_frames, num_features)
        contamination: Expected proportion of outliers
        random_state: Random seed for reproducibility
    
    Returns:
        Anomaly scores per frame (higher = more anomalous)
    """
    if feature_matrix.size == 0:
        return np.array([])
    
    try:
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_matrix)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        
        # Get anomaly scores (negative values = more anomalous)
        scores = iso_forest.decision_function(X_scaled)
        
        # Convert to positive anomaly scores (higher = more anomalous)
        scores = -scores
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)
        scores *= 10  # Scale to roughly match robust z-score range
        
        return scores.astype(np.float32)
    except Exception as e:
        print(f"Warning: Error in Isolation Forest scoring: {e}")
        # Fallback to robust z-score
        return compute_robust_anomaly_scores(feature_matrix)


def compute_mahalanobis_scores(
    feature_matrix: np.ndarray,
    support_fraction: Optional[float] = None
) -> np.ndarray:
    """
    Compute anomaly scores using Mahalanobis distance with robust covariance.
    
    Args:
        feature_matrix: Feature matrix (num_frames, num_features)
        support_fraction: Fraction of data to use for covariance estimation
    
    Returns:
        Anomaly scores per frame
    """
    if feature_matrix.size == 0:
        return np.array([])
    
    try:
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_matrix)
        
        # Use robust covariance estimation
        if support_fraction is None:
            support_fraction = min(0.8, (len(feature_matrix) + feature_matrix.shape[1] + 1) / (2 * len(feature_matrix)))
        
        robust_cov = EllipticEnvelope(
            support_fraction=support_fraction,
            random_state=42
        )
        
        # Fit and get Mahalanobis distances
        robust_cov.fit(X_scaled)
        scores = robust_cov.decision_function(X_scaled)
        
        # Convert to positive anomaly scores
        scores = -scores
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)
        scores *= 5  # Scale appropriately
        
        return scores.astype(np.float32)
    except Exception as e:
        print(f"Warning: Error in Mahalanobis scoring: {e}")
        # Fallback to robust z-score
        return compute_robust_anomaly_scores(feature_matrix)


def classify_by_threshold(
    scores: np.ndarray, 
    threshold: float = 3.0
) -> Tuple[str, float]:
    """
    Classify machine state based on anomaly score threshold.
    
    Args:
        scores: Anomaly scores per frame
        threshold: Classification threshold
    
    Returns:
        Tuple of (label, decision_value)
    """
    if scores.size == 0:
        return "Normal", 0.0
    
    decision_value = float(np.nanmax(scores))
    label = "Faulty" if decision_value >= threshold else "Normal"
    return label, decision_value


def compute_health_score(
    scores: np.ndarray,
    threshold: float = 3.0,
    method: str = "inverse_max"
) -> float:
    """
    Compute a normalized health score (0-100) from anomaly scores.
    
    Args:
        scores: Anomaly scores per frame
        threshold: Reference threshold
        method: Method for computing health score
    
    Returns:
        Health score (0-100, higher is healthier)
    """
    if scores.size == 0:
        return 100.0
    
    try:
        if method == "inverse_max":
            max_score = np.nanmax(scores)
            health = max(0, 100 - (max_score / threshold * 100))
        elif method == "inverse_mean":
            mean_score = np.nanmean(scores)
            health = max(0, 100 - (mean_score / threshold * 100))
        elif method == "percentile":
            # Based on 95th percentile
            p95_score = np.nanpercentile(scores, 95)
            health = max(0, 100 - (p95_score / threshold * 100))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return float(np.clip(health, 0, 100))
    except Exception:
        return 50.0  # Default neutral score


def detect_trend(
    scores: np.ndarray,
    times: np.ndarray,
    window_size: int = 20
) -> Tuple[str, float]:
    """
    Detect trend in anomaly scores over time.
    
    Args:
        scores: Anomaly scores per frame
        times: Time stamps for each frame
        window_size: Window size for trend analysis
    
    Returns:
        Tuple of (trend_label, trend_slope)
    """
    if len(scores) < window_size:
        return "Stable", 0.0
    
    try:
        # Use linear regression to detect trend
        recent_scores = scores[-window_size:]
        recent_times = times[-window_size:]
        
        # Compute slope using least squares
        A = np.vstack([recent_times, np.ones(len(recent_times))]).T
        slope, intercept = np.linalg.lstsq(A, recent_scores, rcond=None)[0]
        
        # Classify trend
        if slope > 0.1:
            trend_label = "Degrading"
        elif slope < -0.1:
            trend_label = "Improving"
        else:
            trend_label = "Stable"
        
        return trend_label, float(slope)
    except Exception:
        return "Stable", 0.0


def predict_remaining_life(
    scores: np.ndarray,
    times: np.ndarray,
    failure_threshold: float = 5.0,
    confidence_threshold: float = 0.7
) -> Tuple[Optional[float], str]:
    """
    Simple linear extrapolation to predict remaining useful life.
    
    Args:
        scores: Anomaly scores per frame
        times: Time stamps for each frame
        failure_threshold: Score threshold indicating failure
        confidence_threshold: Minimum R² for prediction confidence
    
    Returns:
        Tuple of (predicted_time_to_failure, confidence_level)
    """
    if len(scores) < 10:
        return None, "Insufficient data"
    
    try:
        # Fit linear trend to recent data
        recent_window = min(50, len(scores))
        recent_scores = scores[-recent_window:]
        recent_times = times[-recent_window:]
        
        # Linear regression
        A = np.vstack([recent_times, np.ones(len(recent_times))]).T
        slope, intercept = np.linalg.lstsq(A, recent_scores, rcond=None)[0]
        
        # Check trend quality (R²)
        predicted = slope * recent_times + intercept
        ss_res = np.sum((recent_scores - predicted) ** 2)
        ss_tot = np.sum((recent_scores - np.mean(recent_scores)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
        
        if r_squared < confidence_threshold or slope <= 0:
            return None, "No clear degradation trend"
        
        # Predict time to reach failure threshold
        current_time = times[-1]
        current_score = scores[-1]
        
        if current_score >= failure_threshold:
            return 0.0, "Already at failure threshold"
        
        time_to_failure = (failure_threshold - current_score) / slope
        
        # Determine confidence level
        if r_squared >= 0.9:
            confidence = "High"
        elif r_squared >= confidence_threshold:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return time_to_failure, confidence
    except Exception:
        return None, "Prediction error"