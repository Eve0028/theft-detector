"""
Signal quality check for EEG: per-channel stats and quality assessment.

Used before main SSVEP loop to verify electrode contact and detect saturation/artifacts.
Dry electrodes have no direct impedance; we infer poor contact from flat or noisy signal.
"""

from typing import Any, Dict, List, Tuple

import numpy as np


# Thresholds (tunable): poor contact vs normal EEG vs artifacts/saturation
STD_MIN = 1.0
PTP_MIN = 5.0
STD_HIGH_ARTIFACT = 100.0
PTP_SATURATION = 500.0
STD_FAIR = 50.0
PTP_FAIR = 200.0
ABS_CLIP_WARN = 400.0  # Single-sample absolute value above this suggests clipping


def assess_quality(std_val: float, ptp_val: float, max_abs: float) -> str:
    """
    Classify channel quality from std, peak-to-peak, and max absolute value.

    Returns
    -------
    str
        "good", "fair", or "poor"
    """
    if std_val < STD_MIN or ptp_val < PTP_MIN:
        return "poor"  # Flat line, likely bad contact
    if std_val > STD_HIGH_ARTIFACT or ptp_val > PTP_SATURATION or max_abs > ABS_CLIP_WARN:
        return "poor"  # Artifacts or saturation
    if std_val > STD_FAIR or ptp_val > PTP_FAIR:
        return "fair"
    return "good"


def compute_channel_stats(
    data: np.ndarray,
    channel_names: List[str],
    snippet_len: int = 10,
) -> List[Dict[str, Any]]:
    """
    Compute per-channel statistics and a short value snippet.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_samples, n_channels), order of columns = channel_names.
    channel_names : list of str
        Names for each channel.
    snippet_len : int
        Number of recent samples to include as snippet.

    Returns
    -------
    list of dict
        Each dict: name, mean, std, min, max, ptp, quality, snippet (list of floats).
    """
    if data is None or data.size == 0:
        return []
    n_ch = data.shape[1]
    results = []
    for c in range(min(n_ch, len(channel_names))):
        ch_name = channel_names[c] if c < len(channel_names) else f"Ch{c}"
        col = data[:, c]
        mean_val = float(np.mean(col))
        std_val = float(np.std(col))
        min_val = float(np.min(col))
        max_val = float(np.max(col))
        ptp_val = float(np.ptp(col))
        max_abs = float(np.max(np.abs(col)))
        quality = assess_quality(std_val, ptp_val, max_abs)
        snippet = col[-snippet_len:].tolist() if len(col) >= snippet_len else col.tolist()
        results.append({
            "name": ch_name,
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "ptp": ptp_val,
            "quality": quality,
            "snippet": snippet,
        })
    return results


def overall_status(stats: List[Dict[str, Any]]) -> Tuple[str, bool]:
    """
    Overall status message and whether it is OK to proceed.

    Returns
    -------
    tuple
        (message, ok_to_proceed)
    """
    if not stats:
        return "No data", False
    poor = sum(1 for s in stats if s["quality"] == "poor")
    fair = sum(1 for s in stats if s["quality"] == "fair")
    good = sum(1 for s in stats if s["quality"] == "good")
    if poor == len(stats):
        return "All channels poor — check electrode contact and reference (Fp1).", False
    if poor > 0:
        return f"OK (1+ channel poor: check contact). Good: {good}, Fair: {fair}, Poor: {poor}", True
    if fair > 0:
        return f"OK. Good: {good}, Fair: {fair}", True
    return f"Signal good on all {len(stats)} channels.", True
