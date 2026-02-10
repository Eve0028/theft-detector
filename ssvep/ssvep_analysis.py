"""
SSVEP frequency detection from short EEG windows.

Preprocessing: bandpass filter, optional common-average reference.
Detection: FFT power or CCA (Canonical Correlation Analysis) at stimulus
frequencies; returns which target frequency dominates for feedback.
Supports N targets and optional rest state (no dominant frequency).
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal
from scipy.linalg import sqrtm


def bandpass_filter(
    data: np.ndarray,
    low_hz: float,
    high_hz: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply zero-phase bandpass Butterworth along last axis (time).
    `data` shape: (..., n_times).
    """
    nyq = 0.5 * fs
    low = max(0.01, low_hz / nyq)
    high = min(0.99, high_hz / nyq)
    if low >= high:
        return data
    b, a = scipy_signal.butter(order, [low, high], btype="band")
    return scipy_signal.filtfilt(b, a, data, axis=-1)


def common_average_reference(data: np.ndarray, axis: int = -2) -> np.ndarray:
    """Subtract mean across channels. `data` shape: (n_samples, n_channels, ...) or (n_samples, n_channels)."""
    mean = np.mean(data, axis=axis, keepdims=True)
    return data - mean


def power_at_frequency(
    sig: np.ndarray,
    fs: float,
    freq_hz: float,
    tol_hz: float = 0.5,
    use_second_harmonic: bool = True,
) -> float:
    """
    Sum of power in bins around freq_hz and optionally 2*freq_hz.
    `sig`: 1D time series (or last axis is time).
    """
    n = sig.shape[-1]
    if n == 0:
        return 0.0
    fft_vals = np.fft.rfft(sig, axis=-1)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    power = np.abs(fft_vals) ** 2

    def band_power(f_target: float) -> float:
        mask = (freqs >= f_target - tol_hz) & (freqs <= f_target + tol_hz)
        return float(np.sum(power[..., mask]))

    p = band_power(freq_hz)
    if use_second_harmonic:
        p += band_power(2.0 * freq_hz)
    return p


def compute_power_spectrum(
    data: np.ndarray,
    fs: float,
    freq_min_hz: float = 5.0,
    freq_max_hz: float = 30.0,
    step_hz: float = 0.5,
    bandpass_low: float = 5.0,
    bandpass_high: float = 30.0,
    filter_order: int = 4,
    car: bool = True,
    tol_hz: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power at frequencies from freq_min_hz to freq_max_hz with step_hz.
    Uses same preprocessing as detection (bandpass, optional CAR). Returns
    (freqs, powers) for plotting; power is from averaged channels, no second harmonic.
    """
    if data is None or data.size == 0:
        freqs = np.arange(freq_min_hz, freq_max_hz + step_hz * 0.5, step_hz)
        return freqs, np.zeros_like(freqs)
    filtered = bandpass_filter(
        data.T, low_hz=bandpass_low, high_hz=bandpass_high, fs=fs, order=filter_order
    ).T
    if car and filtered.shape[1] > 1:
        filtered = common_average_reference(filtered, axis=1)
    ch_series = np.mean(filtered, axis=0)
    freqs = np.arange(freq_min_hz, freq_max_hz + step_hz * 0.5, step_hz)
    powers = np.array([
        power_at_frequency(ch_series, fs, f, tol_hz=tol_hz, use_second_harmonic=False)
        for f in freqs
    ])
    return freqs, powers


def detect_ssvep(
    data: np.ndarray,
    fs: float,
    freq_left_hz: float,
    freq_right_hz: float,
    bandpass_low: float = 5.0,
    bandpass_high: float = 30.0,
    filter_order: int = 4,
    car: bool = True,
    freq_tol_hz: float = 0.5,
    use_second_harmonic: bool = True,
    method: str = "fft",
    cca_n_harmonics: int = 2,
    cca_components: int = 1,
    cca_reg: float = 1e-4,
) -> Tuple[int, float, float]:
    """
    Run preprocessing and return which target (0 = left, 1 = right) and raw scores.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_samples, n_channels).
    fs : float
        Sampling rate in Hz.
    freq_left_hz, freq_right_hz : float
        Left and right stimulus frequencies.
    bandpass_low, bandpass_high : float
        Bandpass filter range.
    filter_order : int
        Butterworth order.
    car : bool
        Apply common-average reference.
    freq_tol_hz : float
        FFT bin tolerance around each frequency (FFT method only).
    use_second_harmonic : bool
        Include 2*f in power sum (FFT method only).
    method : str
        "fft" = power at frequency; "cca" = canonical correlation with reference signals.
    cca_n_harmonics : int
        Number of harmonics in CCA reference (method "cca").
    cca_components : int
        Sum of first N canonical correlations as score (method "cca").
    cca_reg : float
        Regularization for CCA covariance matrices (method "cca").

    Returns
    -------
    tuple
        (selected_index 0 or 1, score_left, score_right).
    """
    if data is None or data.size == 0:
        return 0, 0.0, 0.0
    if method == "cca":
        return _detect_ssvep_cca(
            data, fs, freq_left_hz, freq_right_hz,
            bandpass_low=bandpass_low,
            bandpass_high=bandpass_high,
            filter_order=filter_order,
            car=car,
            n_harmonics=cca_n_harmonics,
            cca_components=cca_components,
            cca_reg=cca_reg,
        )
    # FFT method (bandpass_filter expects time on last axis; data is (samples, channels))
    filtered = bandpass_filter(
        data.T, low_hz=bandpass_low, high_hz=bandpass_high, fs=fs, order=filter_order
    ).T
    if car and filtered.shape[1] > 1:
        filtered = common_average_reference(filtered, axis=1)
    ch_series = np.mean(filtered, axis=0)
    p_left = power_at_frequency(
        ch_series, fs, freq_left_hz,
        tol_hz=freq_tol_hz,
        use_second_harmonic=use_second_harmonic,
    )
    p_right = power_at_frequency(
        ch_series, fs, freq_right_hz,
        tol_hz=freq_tol_hz,
        use_second_harmonic=use_second_harmonic,
    )
    selected = 1 if p_right > p_left else 0
    return selected, p_left, p_right


def _build_cca_reference(
    n_samples: int,
    fs: float,
    freq_hz: float,
    n_harmonics: int = 2,
) -> np.ndarray:
    """Build reference matrix Y: [sin(f*t), cos(f*t), sin(2f*t), cos(2f*t), ...]. Shape (n_samples, 2*n_harmonics)."""
    t = np.arange(n_samples, dtype=float) / fs
    cols = []
    for h in range(1, n_harmonics + 1):
        f = h * freq_hz
        cols.append(np.sin(2 * np.pi * f * t))
        cols.append(np.cos(2 * np.pi * f * t))
    return np.column_stack(cols)


def _cca_correlation(X: np.ndarray, Y: np.ndarray, reg: float = 1e-4) -> np.ndarray:
    """
    Canonical correlations between X (n_samples, n_x) and Y (n_samples, n_y).
    Returns array of min(n_x, n_y) canonical correlations (sorted descending).
    """
    n = X.shape[0]
    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)
    Cxx = (X.T @ X) / n + reg * np.eye(X.shape[1])
    Cyy = (Y.T @ Y) / n + reg * np.eye(Y.shape[1])
    Cxy = (X.T @ Y) / n
    try:
        Cxx_sqrt = np.real(sqrtm(Cxx))
        Cyy_sqrt = np.real(sqrtm(Cyy))
        Cxx_inv_sqrt = np.linalg.inv(Cxx_sqrt)
        Cyy_inv_sqrt = np.linalg.inv(Cyy_sqrt)
    except (np.linalg.LinAlgError, ValueError):
        return np.array([0.0])
    M = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt
    M = np.real(M)
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    return np.clip(s, 0.0, 1.0)


def _detect_ssvep_cca(
    data: np.ndarray,
    fs: float,
    freq_left_hz: float,
    freq_right_hz: float,
    bandpass_low: float,
    bandpass_high: float,
    filter_order: int,
    car: bool,
    n_harmonics: int = 2,
    cca_components: int = 1,
    cca_reg: float = 1e-4,
) -> Tuple[int, float, float]:
    """CCA-based detection: reference signals at f and harmonics; max canonical correlation wins."""
    if data is None or data.size == 0:
        return 0, 0.0, 0.0
    # bandpass_filter expects time on last axis; data is (samples, channels)
    filtered = bandpass_filter(
        data.T, low_hz=bandpass_low, high_hz=bandpass_high, fs=fs, order=filter_order
    ).T
    if car and filtered.shape[1] > 1:
        filtered = common_average_reference(filtered, axis=1)
    n_samples = filtered.shape[0]
    Y_left = _build_cca_reference(n_samples, fs, freq_left_hz, n_harmonics=n_harmonics)
    Y_right = _build_cca_reference(n_samples, fs, freq_right_hz, n_harmonics=n_harmonics)
    r_left = _cca_correlation(filtered, Y_left, reg=cca_reg)
    r_right = _cca_correlation(filtered, Y_right, reg=cca_reg)
    score_left = float(np.sum(r_left[:cca_components]))
    score_right = float(np.sum(r_right[:cca_components]))
    selected = 1 if score_right > score_left else 0
    return selected, score_left, score_right


def detect_ssvep_multi(
    data: np.ndarray,
    fs: float,
    freqs_hz: List[float],
    bandpass_low: float = 5.0,
    bandpass_high: float = 30.0,
    filter_order: int = 4,
    car: bool = True,
    freq_tol_hz: float = 0.5,
    use_second_harmonic: bool = True,
    method: str = "fft",
    cca_n_harmonics: int = 2,
    cca_components: int = 1,
    cca_reg: float = 1e-4,
    rest_threshold: Optional[float] = None,
) -> Tuple[int, List[float]]:
    """
    Detect among N SSVEP targets; optional rest when max score below rest_threshold.

    Returns
    -------
    tuple
        (selected_index 0..N-1, or -1 for rest; list of N scores).
    """
    n_targets = len(freqs_hz)
    if data is None or data.size == 0 or n_targets == 0:
        return 0, [0.0] * n_targets
    if method == "cca":
        idx, scores = _detect_ssvep_multi_cca(
            data, fs, freqs_hz,
            bandpass_low=bandpass_low,
            bandpass_high=bandpass_high,
            filter_order=filter_order,
            car=car,
            n_harmonics=cca_n_harmonics,
            cca_components=cca_components,
            cca_reg=cca_reg,
        )
    else:
        filtered = bandpass_filter(
            data.T, low_hz=bandpass_low, high_hz=bandpass_high, fs=fs, order=filter_order
        ).T
        if car and filtered.shape[1] > 1:
            filtered = common_average_reference(filtered, axis=1)
        ch_series = np.mean(filtered, axis=0)
        scores = [
            power_at_frequency(
                ch_series, fs, f,
                tol_hz=freq_tol_hz,
                use_second_harmonic=use_second_harmonic,
            )
            for f in freqs_hz
        ]
        idx = int(np.argmax(scores))
    if rest_threshold is not None and len(scores) > 0 and max(scores) < rest_threshold:
        idx = -1
    return idx, scores


def _detect_ssvep_multi_cca(
    data: np.ndarray,
    fs: float,
    freqs_hz: List[float],
    bandpass_low: float,
    bandpass_high: float,
    filter_order: int,
    car: bool,
    n_harmonics: int = 2,
    cca_components: int = 1,
    cca_reg: float = 1e-4,
) -> Tuple[int, List[float]]:
    """CCA for N frequencies; returns index of max score and list of scores."""
    if data is None or data.size == 0 or len(freqs_hz) == 0:
        return 0, []
    filtered = bandpass_filter(
        data.T, low_hz=bandpass_low, high_hz=bandpass_high, fs=fs, order=filter_order
    ).T
    if car and filtered.shape[1] > 1:
        filtered = common_average_reference(filtered, axis=1)
    n_samples = filtered.shape[0]
    scores = []
    for f_hz in freqs_hz:
        Y = _build_cca_reference(n_samples, fs, f_hz, n_harmonics=n_harmonics)
        r = _cca_correlation(filtered, Y, reg=cca_reg)
        scores.append(float(np.sum(r[:cca_components])))
    idx = int(np.argmax(scores))
    return idx, scores


def get_smoothed_selection(
    history: List[int],
    min_agreements: int = 2,
    n_classes: Optional[int] = None,
) -> Optional[int]:
    """
    Require `min_agreements` same consecutive results before returning class index.
    Supports 2 classes (0, 1) or N classes (0..N-1); use n_classes to allow 0..n_classes-1.
    Returns None if not enough agreement (no feedback).
    """
    if len(history) < min_agreements:
        return None
    recent = history[-min_agreements:]
    if n_classes is None:
        n_classes = 2
    for c in range(n_classes):
        if all(x == c for x in recent):
            return c
    return None
