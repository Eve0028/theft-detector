"""
EEG FIF Analyzer - Streamlit Application
=========================================

Interactive web application for analyzing EEG data in FIF format
using MNE-Python with CTP-BAD classification.

Features:
- File upload and inspection
- Signal quality assessment
- Interactive filtering (notch + bandpass)
- Epoch creation and visualization
- ERP analysis (probe vs irrelevant)
- P300 amplitude analysis (300-600ms configurable window)
- CTP-BAD bootstrap analysis (guilty/innocent classification)
- Interactive plots and visualizations
- Export results (CSV, PNG, NPY)

CTP-BAD Method:
- Bootstrap Amplitude Difference for statistical classification
- 1000 iterations (configurable)
- 90% threshold (configurable)
- Per-channel and overall classification
- Confidence levels and visualization

Usage:
    streamlit run scripts/eeg_analyzer_app.py
    
    # Or use launchers:
    run_analyzer.bat  # Windows
    ./run_analyzer.sh # Linux/Mac

Documentation:
    - docs/STREAMLIT_APP_GUIDE.md - Quick start guide
    - scripts/README_ANALYZER_APP.md - Full documentation
    - docs/CTP_BAD_METHOD.md - CTP-BAD method details

"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from itertools import product as _iter_product
import io
import json
import os as _os
import pickle
import time as _time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import mne
    mne.set_log_level('WARNING')
except ImportError:
    st.error("MNE-Python not installed! Install with: pip install mne")
    st.stop()

try:
    from autoreject import AutoReject as _AutoReject  # noqa: F401
    _AUTOREJECT_AVAILABLE = True
except ImportError:
    _AUTOREJECT_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

try:
    from optuna.visualization import plot_optimization_history
    _OPTUNA_VIZ_AVAILABLE = True
except ImportError:
    plot_optimization_history = None  # type: ignore[misc, assignment]
    _OPTUNA_VIZ_AVAILABLE = False

from scipy.signal import butter, filtfilt


# ============================================================================
# Configuration
# ============================================================================

st.set_page_config(
    page_title="EEG FIF Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================

if 'raw' not in st.session_state:
    st.session_state.raw = None
if 'raw_filtered' not in st.session_state:
    st.session_state.raw_filtered = None
if 'events' not in st.session_state:
    st.session_state.events = None
if 'event_id' not in st.session_state:
    st.session_state.event_id = None
if 'epochs' not in st.session_state:
    st.session_state.epochs = None
if 'probe_erp' not in st.session_state:
    st.session_state.probe_erp = None
if 'irrelevant_erp' not in st.session_state:
    st.session_state.irrelevant_erp = None
if 'individual_p300_window' not in st.session_state:
    st.session_state.individual_p300_window = None
if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = None


# ============================================================================
# Helper Functions
# ============================================================================

def load_fif_file(file_path):
    """
    Load FIF file and extract events.
    
    Parameters
    ----------
    file_path : str or file-like
        Path to FIF file or uploaded file object
        
    Returns
    -------
    raw : mne.io.Raw
        Raw EEG data
    events : np.ndarray
        Events array
    event_id : dict
        Event ID mapping
    """
    try:
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose='ERROR')
        
        # Extract events from annotations
        if len(raw.annotations) > 0:
            events, event_id = mne.events_from_annotations(raw, verbose='ERROR')
        else:
            events = None
            event_id = None
            
        return raw, events, event_id
    except Exception as e:
        raise Exception(f"Error loading FIF file: {str(e)}")


def check_signal_quality(raw, duration=10.0):
    """
    Assess signal quality for each channel.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    duration : float
        Duration to analyze (seconds)
        
    Returns
    -------
    pd.DataFrame
        Quality metrics per channel
    """
    # Get data segment
    end_sample = min(int(duration * raw.info['sfreq']), len(raw.times))
    data = raw.get_data()[:, :end_sample] * 1e6  # Convert to µV
    
    results = []
    
    for i, ch_name in enumerate(raw.ch_names):
        ch_data = data[i, :]
        
        mean_val = np.mean(ch_data)
        std_val = np.std(ch_data)
        min_val = np.min(ch_data)
        max_val = np.max(ch_data)
        p2p = max_val - min_val
        
        # Assess quality
        if std_val < 1.0 or p2p < 5.0:
            quality = "FLAT"
            status = "🔴"
        elif std_val > 200.0 or p2p > 1000.0:
            quality = "VERY NOISY"
            status = "🔴"
        elif std_val > 100.0 or p2p > 500.0:
            quality = "POOR"
            status = "🟡"
        else:
            quality = "GOOD"
            status = "🟢"
        
        results.append({
            'Channel': ch_name,
            'Status': status,
            'Quality': quality,
            'Mean (µV)': f"{mean_val:.2f}",
            'Std (µV)': f"{std_val:.2f}",
            'Peak-to-Peak (µV)': f"{p2p:.2f}"
        })
    
    return pd.DataFrame(results)


def apply_filters(raw, notch_freqs=None, lowcut=None, highcut=None,
                   method='fir', iir_order=4):
    """
    Apply filters to raw EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    notch_freqs : list or None
        Frequencies for notch filter (Hz).
    lowcut : float or None
        High-pass filter cutoff (Hz).
    highcut : float or None
        Low-pass filter cutoff (Hz).
    method : str
        ``'fir'`` (MNE default, linear-phase) or ``'iir'`` (Butterworth,
        steeper rolloff — better for aggressive drift removal).  IIR
        filtering uses zero-phase (forward-backward) ``sosfiltfilt`` to
        avoid latency shifts of ERP components.
    iir_order : int
        Butterworth filter order per pass (used only when *method* is
        ``'iir'``).  Even values recommended for zero-phase ``filtfilt``.

    Returns
    -------
    mne.io.Raw
        Filtered EEG data.
    """
    raw_filtered = raw.copy()

    if notch_freqs:
        raw_filtered.notch_filter(notch_freqs, verbose='ERROR')

    filter_kwargs: dict = {'verbose': 'ERROR'}
    if method == 'iir':
        filter_kwargs['method'] = 'iir'
        filter_kwargs['iir_params'] = {
            'order': iir_order, 'ftype': 'butter', 'output': 'sos',
        }
        filter_kwargs['phase'] = 'zero-double'

    if lowcut is not None and highcut is not None:
        raw_filtered.filter(lowcut, highcut, **filter_kwargs)
    elif lowcut is not None:
        raw_filtered.filter(lowcut, None, **filter_kwargs)
    elif highcut is not None:
        raw_filtered.filter(None, highcut, **filter_kwargs)

    return raw_filtered


def smooth_epochs_array(
    data: np.ndarray,
    sfreq: float,
    method: str = 'lowpass',
    lowpass_hz: float = 6.0,
    moving_avg_ms: float = 100.0,
) -> np.ndarray:
    """
    Smooth epoch data along the time axis (axis=-1).

    Returns a smoothed **copy** — the original array is not modified.

    Parameters
    ----------
    data
        Array of shape *(n_epochs, n_channels, n_times)* **or**
        *(n_channels, n_times)*.
    sfreq
        Sampling frequency (Hz).
    method
        ``'lowpass'`` — zero-phase Butterworth low-pass via
        :func:`scipy.signal.filtfilt`.
        ``'moving_average'`` — uniform moving average (symmetric window).
    lowpass_hz
        Cutoff frequency for the Butterworth filter.
    moving_avg_ms
        Window length (ms) for the moving average.

    Returns
    -------
    np.ndarray
        Smoothed copy of *data* (same shape).
    """
    out = data.copy()
    if method == 'lowpass':
        nyq = sfreq / 2.0
        cutoff = min(lowpass_hz, nyq - 1.0)
        if cutoff <= 0:
            return out
        b, a = butter(N=4, Wn=cutoff / nyq, btype='low')
        if out.ndim == 2:
            for ch in range(out.shape[0]):
                out[ch] = filtfilt(b, a, out[ch])
        else:
            for ep in range(out.shape[0]):
                for ch in range(out.shape[1]):
                    out[ep, ch] = filtfilt(b, a, out[ep, ch])
    elif method == 'moving_average':
        win_samples = max(1, int(round(moving_avg_ms / 1000.0 * sfreq)))
        kernel = np.ones(win_samples) / win_samples
        if out.ndim == 2:
            for ch in range(out.shape[0]):
                out[ch] = np.convolve(out[ch], kernel, mode='same')
        else:
            for ep in range(out.shape[0]):
                for ch in range(out.shape[1]):
                    out[ep, ch] = np.convolve(out[ep, ch], kernel, mode='same')
    return out


def create_epochs(raw, events, event_id, tmin, tmax, baseline,
                   reject_threshold_uv, detrend=None):
    """
    Create epochs around stimulus events.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    events : np.ndarray
        Events array.
    event_id : dict
        Event ID mapping.
    tmin : float
        Start time (seconds).
    tmax : float
        End time (seconds).
    baseline : tuple or None
        Baseline period.
    reject_threshold_uv : float or None
        Artifact rejection threshold (µV).
    detrend : int or None
        ``0`` to remove DC offset, ``1`` to remove linear trend from each
        epoch (straightens residual slope after filtering), ``None`` to skip.

    Returns
    -------
    mne.Epochs
        Epoched data.
    """
    s1_event_id = {
        k: v for k, v in event_id.items()
        if 'S1_onset' in k or 's1_onset' in k.lower()
    }

    if not s1_event_id:
        s1_event_id = event_id

    if reject_threshold_uv is not None:
        reject = dict(eeg=reject_threshold_uv * 1e-6)
    else:
        reject = None

    epochs = mne.Epochs(
        raw,
        events,
        event_id=s1_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        detrend=detrend,
        preload=True,
        reject=reject,
        verbose='ERROR'
    )

    return epochs


def _parse_s2_responses(raw):
    """
    Parse S2_response annotations from a raw EEG recording.

    Extracts trial number, correctness, and reaction time from annotations
    of the form ``S2_response|trial=N,key=...,rt=X.XXXX,correct=0|1``.

    :param raw: Raw EEG data with annotations.
    :type raw: mne.io.Raw
    :return: Mapping ``{trial_number: {'correct': bool, 'rt': float}}``.
    :rtype: dict[int, dict]
    """
    import re
    s2_info = {}
    for desc in raw.annotations.description:
        if 'S2_response' not in desc:
            continue
        trial_m = re.search(r'trial=(\d+)', desc)
        correct_m = re.search(r'correct=(\d)', desc)
        rt_m = re.search(r'rt=([\d.]+)', desc)
        if trial_m is None:
            continue
        trial_num = int(trial_m.group(1))
        s2_info[trial_num] = {
            'correct': bool(int(correct_m.group(1))) if correct_m else False,
            'rt': float(rt_m.group(1)) if rt_m else np.nan,
        }
    return s2_info


def reject_s1_by_s2_performance(epochs, raw, max_rt=None):
    """
    Drop S1 epochs whose associated S2 response was incorrect or too slow.

    Each S1 epoch annotation contains ``trial=N``. The function looks up the
    corresponding ``S2_response|trial=N`` annotation in *raw* and drops the
    epoch when the S2 answer was wrong (``correct=0``) or the reaction time
    exceeded *max_rt*.

    :param epochs: Epochs with ``trial=N`` in event descriptions (S1 or S2).
    :type epochs: mne.Epochs
    :param raw: Raw recording with full annotations (including S2_response).
    :type raw: mne.io.Raw
    :param max_rt: Maximum allowed S2 reaction time (seconds). ``None`` to
        skip the RT criterion.
    :type max_rt: float or None
    :return: ``(cleaned_epochs, n_dropped, drop_log)`` where *drop_log* is a
        list of ``(epoch_index, trial_num, reason)`` tuples.
    :rtype: tuple[mne.Epochs, int, list[tuple]]
    """
    import re

    s2_info = _parse_s2_responses(raw)
    if not s2_info:
        return epochs, 0, []

    rev_event_id = {v: k for k, v in epochs.event_id.items()}
    drop_indices = []
    drop_log = []

    for idx in range(len(epochs)):
        ev_code = epochs.events[idx, 2]
        desc = rev_event_id.get(ev_code, '')
        trial_m = re.search(r'trial=(\d+)', desc)
        if trial_m is None:
            continue
        trial_num = int(trial_m.group(1))
        info = s2_info.get(trial_num)
        if info is None:
            drop_indices.append(idx)
            drop_log.append((idx, trial_num, 'no S2 response'))
            continue
        if not info['correct']:
            drop_indices.append(idx)
            drop_log.append((idx, trial_num, 'S2 incorrect'))
            continue
        if max_rt is not None and not np.isnan(info['rt']) and info['rt'] > max_rt:
            drop_indices.append(idx)
            drop_log.append((idx, trial_num, f"S2 RT {info['rt']:.3f}s > {max_rt}s"))

    if drop_indices:
        epochs = epochs.drop(drop_indices, reason='S2_performance')

    return epochs, len(drop_indices), drop_log


def adaptive_reject_epochs(epochs, method="iqr", k=3.0):
    """
    Remove artifact epochs using statistical outlier detection.

    Computes the per-epoch peak-to-peak amplitude (worst channel) and rejects
    epochs whose amplitude is a statistical outlier.

    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs (not modified in-place).
    method : str
        Detection method: ``"iqr"``, ``"zscore"``.
    k : float
        Sensitivity parameter:
        - IQR: reject epochs with amplitude > Q3 + k×IQR (typical 2.5–4.0).
        - Z-score: reject epochs with amplitude z-score > k (typical 2.5–3.5).

    Returns
    -------
    tuple
        ``(clean_epochs, n_dropped, ptp_per_epoch, keep_mask)``
    """
    data = epochs.get_data() * 1e6          # (n_epochs, n_channels, n_times) µV
    ptp = np.ptp(data, axis=2)              # (n_epochs, n_channels)
    ptp_max = ptp.max(axis=1)               # worst channel per epoch

    if method == "iqr":
        q1, q3 = np.percentile(ptp_max, [25, 75])
        upper = q3 + k * (q3 - q1)
        keep_mask = ptp_max <= upper
    else:  # zscore
        z = (ptp_max - ptp_max.mean()) / (ptp_max.std() + 1e-10)
        keep_mask = z <= k

    n_dropped = int((~keep_mask).sum())
    clean_epochs = epochs[keep_mask]
    return clean_epochs, n_dropped, ptp_max, keep_mask


def reject_noisy_baseline(epochs, k=3.0):
    """Reject epochs whose baseline window has abnormally high peak-to-peak.

    A noisy baseline corrupts the baseline correction (mean subtraction),
    shifting the entire post-stimulus waveform.  This check complements
    autoreject / adaptive rejection, which evaluate the full epoch and
    may miss baseline-specific artifacts (e.g. swallowing, blink tail).

    The baseline window is taken from ``epochs.baseline``.  If no baseline
    is set, the function returns the epochs unchanged.

    Rejection uses an IQR criterion: epochs with baseline ptp >
    Q3 + *k* * IQR are dropped.

    :param epochs: Input epochs (not modified in-place).
    :param k: IQR multiplier (lower = stricter).
    :returns: ``(clean_epochs, n_dropped, keep_mask)``
    """
    bl = epochs.baseline
    if bl is None:
        return epochs, 0, np.ones(len(epochs), dtype=bool)

    bl_start, bl_end = bl
    times = epochs.times
    bl_mask = (times >= bl_start) & (times <= bl_end)
    if not bl_mask.any():
        return epochs, 0, np.ones(len(epochs), dtype=bool)

    data = epochs.get_data() * 1e6
    bl_data = data[:, :, bl_mask]
    bl_ptp = np.ptp(bl_data, axis=2).max(axis=1)

    q1, q3 = np.percentile(bl_ptp, [25, 75])
    iqr = q3 - q1
    upper = q3 + k * iqr
    keep_mask = bl_ptp <= upper
    n_dropped = int((~keep_mask).sum())
    clean = epochs[keep_mask]
    return clean, n_dropped, keep_mask


def _ensure_montage(inst):
    """
    Set a standard 10-20 montage if channel positions are missing.

    Autoreject requires valid channel locations.  BrainAccess FIF files
    typically lack a montage, so we attach one automatically for any
    channels whose names match the standard 10-20 system.

    Parameters
    ----------
    inst : mne.io.Raw | mne.Epochs
        MNE object (modified in-place).
    """
    dig = inst.info.get('dig')
    has_positions = dig is not None and len(dig) > 0
    if has_positions:
        return

    montage = mne.channels.make_standard_montage('standard_1020')
    montage_names = {ch.upper() for ch in montage.ch_names}
    matching = [
        ch for ch in inst.ch_names if ch.upper() in montage_names
    ]
    if not matching:
        return
    inst.set_montage(montage, on_missing='ignore')


def _apply_s1_style_global_rejection(epochs, s1_rejection, adaptive_k, ar_n_jobs=1):
    """Global artifact stage matching ``run_pipeline`` S1 (after epoching).

    Applies *adaptive_reject_epochs* for ``iqr`` / ``zscore``, or
    *autoreject_clean_epochs* for ``autoreject``.  For ``static``, MNE
    ``reject=`` is applied at epoch creation — this function is a no-op.
    For ``none``, no second-stage rejection.

    :returns: ``(epochs, autoreject_info)`` where *autoreject_info* is set
        only when autoreject ran.
    """
    ar_info = None
    if s1_rejection in ('iqr', 'zscore'):
        epochs, _, _, _ = adaptive_reject_epochs(
            epochs, method=s1_rejection, k=adaptive_k,
        )
    elif s1_rejection == 'autoreject' and _AUTOREJECT_AVAILABLE:
        epochs, _, ar_info = autoreject_clean_epochs(
            epochs, n_jobs=ar_n_jobs,
        )
    return epochs, ar_info


def autoreject_clean_epochs(epochs, n_jobs=1, random_state=42):
    """
    Clean epochs using the *autoreject* library.

    With **≥ 4 channels** the full ``AutoReject`` algorithm is used
    (per-channel thresholds, interpolation, cross-validation).

    With **< 4 channels** (e.g. 3-electrode BrainAccess setup) the full
    algorithm cannot run, so the function falls back to
    ``get_rejection_threshold`` which computes a single optimal global
    peak-to-peak threshold via Bayesian optimization.

    A standard 10-20 montage is attached automatically when channel
    positions are missing (required by autoreject).

    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs (not modified in-place).
    n_jobs : int
        Number of parallel jobs for cross-validation.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        ``(clean_epochs, reject_log_or_None, info)`` where *info* is a dict
        with keys ``method`` (``'full'`` or ``'threshold'``), ``n_dropped``,
        and optionally ``threshold_uv`` (for the threshold fallback).
    """
    from autoreject import AutoReject, get_rejection_threshold

    epochs = epochs.copy()
    _ensure_montage(epochs)

    n_ch = len(epochs.ch_names)

    if n_ch >= 4:
        ar = AutoReject(
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=False,
        )
        clean_epochs, reject_log = ar.fit_transform(
            epochs, return_log=True,
        )
        n_dropped = int(reject_log.bad_epochs.sum())
        return clean_epochs, reject_log, {
            'method': 'full',
            'n_dropped': n_dropped,
        }

    # Fallback for few-channel setups (< 4 ch)
    reject = get_rejection_threshold(
        epochs, random_state=random_state, verbose=False,
    )
    threshold_v = reject.get('eeg', None)
    threshold_uv = threshold_v * 1e6 if threshold_v else 0.0
    n_before = len(epochs)
    clean_epochs = epochs.drop_bad(reject=reject)
    n_dropped = n_before - len(clean_epochs)
    return clean_epochs, None, {
        'method': 'threshold',
        'n_dropped': n_dropped,
        'threshold_uv': threshold_uv,
    }


def extract_stim_ids(event_id: dict) -> list:
    """
    Extract unique stimulus object names from event annotations.

    Parses ``stim_id=<name>`` fields embedded in annotation strings such as
    ``S1_onset_probe|trial=1,stim_id=wolf``.

    Parameters
    ----------
    event_id : dict
        MNE event_id mapping.

    Returns
    -------
    list of str
        Sorted list of unique stim_id values, e.g. ``['bear', 'dog', 'wolf']``.
    """
    stim_ids = set()
    for key in event_id:
        if 'stim_id=' not in key:
            continue
        idx = key.index('stim_id=') + len('stim_id=')
        val = key[idx:]
        for sep in (',', '|', ' '):
            if sep in val:
                val = val[:val.index(sep)]
        stim_ids.add(val.strip())
    return sorted(stim_ids)


def compute_erps(epochs, target_stim=None, baseline_stims=None, channels=None):
    """
    Compute ERPs for target vs baseline conditions.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data.
    target_stim : str or None
        Label for the target condition.  ``None`` or ``'probe'`` selects all
        probe events.  Any stim_id string (e.g. ``'wolf'``) selects epochs
        annotated with ``stim_id=wolf``.
    baseline_stims : list of str or None
        stim_id labels for baseline epochs.  ``None`` defaults to all
        irrelevant events.
    channels : list of str or None
        Channel subset for the returned Evoked objects.  ``None`` uses all.

    Returns
    -------
    tuple
        ``(target_erp, baseline_erp, target_events, baseline_events)``
    """
    available_events = list(epochs.event_id.keys())

    # Resolve target events
    if target_stim is None or target_stim == 'probe':
        target_events = [k for k in available_events
                         if 'probe' in k.lower() and 's1_onset' in k.lower()]
    else:
        target_events = [k for k in available_events
                         if f'stim_id={target_stim}' in k
                         and 's1_onset' in k.lower()]

    # Resolve baseline events
    if not baseline_stims:
        baseline_events = [k for k in available_events
                           if 'irrelevant' in k.lower() and 's1_onset' in k.lower()]
    else:
        baseline_events = []
        for stim in baseline_stims:
            baseline_events += [k for k in available_events
                                 if f'stim_id={stim}' in k
                                 and 's1_onset' in k.lower()]
        # Deduplicate while preserving order
        seen: set = set()
        baseline_events = [x for x in baseline_events
                           if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]

    target_erp = None
    baseline_erp = None

    ep = epochs.copy().pick_channels(channels) if channels else epochs

    if target_events:
        t_ep = ep[target_events]
        if len(t_ep) > 0:
            target_erp = t_ep.average()

    if baseline_events:
        b_ep = ep[baseline_events]
        if len(b_ep) > 0:
            baseline_erp = b_ep.average()

    return target_erp, baseline_erp, target_events, baseline_events


def analyze_p300(probe_erp, irrelevant_erp, tmin=0.3, tmax=0.6):
    """
    Analyze P300 amplitude in specified time window.
    
    Parameters
    ----------
    probe_erp : mne.Evoked
        Probe ERP
    irrelevant_erp : mne.Evoked
        Irrelevant ERP
    tmin : float
        Window start (seconds)
    tmax : float
        Window end (seconds)
        
    Returns
    -------
    pd.DataFrame
        P300 analysis results
    """
    times = probe_erp.times
    time_mask = (times >= tmin) & (times <= tmax)
    
    results = []
    
    for idx, ch_name in enumerate(probe_erp.ch_names):
        probe_data = probe_erp.data[idx] * 1e6  # µV
        irrelevant_data = irrelevant_erp.data[idx] * 1e6
        
        probe_mean = probe_data[time_mask].mean()
        irrelevant_mean = irrelevant_data[time_mask].mean()
        diff = probe_mean - irrelevant_mean
        
        # Determine significance
        if diff > 10.0:
            effect = "Strong"
            status = "🟢"
        elif diff > 5.0:
            effect = "Moderate"
            status = "🟡"
        elif diff > 2.0:
            effect = "Weak"
            status = "🟡"
        else:
            effect = "None"
            status = "🔴"
        
        results.append({
            'Channel': ch_name,
            'Status': status,
            'Probe (µV)': f"{probe_mean:.2f}",
            'Irrelevant (µV)': f"{irrelevant_mean:.2f}",
            'Difference (µV)': f"{diff:.2f}",
            'Effect': effect
        })
    
    return pd.DataFrame(results)


def ctp_bad_analysis(
    epochs,
    tmin=0.3,
    tmax=0.6,
    n_bootstrap=10000,
    threshold=0.75,
    channels=None,
    target_stim=None,
    baseline_stims=None,
    amplitude_method='peak_valley',
    p2p_tmax_negative=0.9,
    smoothing_method=None,
    smoothing_lowpass_hz=12.0,
    smoothing_window_ms=100.0,
):
    """
    Perform CTP-BAD (Bootstrap Amplitude Difference) analysis.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data.
    tmin : float
        Analysis window start (seconds).
    tmax : float
        Analysis window end (seconds).
    n_bootstrap : int
        Bootstrap iterations.
    threshold : float
        Guilty classification threshold (default 0.75).
    channels : list of str or None
        Channel subset for analysis.  ``None`` uses all channels.
    target_stim : str or None
        Target stim_id (or ``None``/``'probe'`` for default probe events).
    baseline_stims : list of str or None
        Baseline stim_id list.  ``None`` defaults to all irrelevant events.
    amplitude_method : str
        How to summarise each epoch's amplitude within the time window:

        - ``'mean'``: mean amplitude in *[tmin, tmax]* (default, signed).
          Best for noisy data — point-by-point averaging cancels random
          spikes.
        - ``'peak_to_peak'``: Rosenfeld method — finds the maximum positive
          peak in *[tmin, tmax]*, then the maximum negative trough between
          that peak and *p2p_tmax_negative*.  Robust against slow baseline
          drift and CNV but sensitive to single sharp artifacts.
        - ``'peak_valley'``: absolute peak-to-peak — finds the global
          maximum and minimum in *[tmin, tmax]* regardless of temporal
          order.  Amplitude = max − min.
        - ``'baseline_to_peak'``: maximum positive amplitude in
          *[tmin, tmax]* relative to the zeroed baseline.
    p2p_tmax_negative : float
        End of the negative-trough search window (seconds) for the
        Rosenfeld ``'peak_to_peak'`` method.  Ignored by other methods.
    smoothing_method : str or None
        Smoothing applied to individual epochs before peak-based amplitude
        extraction (``'peak_to_peak'``, ``'peak_valley'``,
        ``'baseline_to_peak'``).  Ignored when *amplitude_method* is
        ``'mean'``.

        - ``None``: no smoothing.
        - ``'lowpass'``: zero-phase Butterworth low-pass via
          :func:`scipy.signal.filtfilt`.
        - ``'moving_average'``: symmetric moving average.
    smoothing_lowpass_hz : float
        Cutoff frequency for the ``'lowpass'`` smoothing method.
    smoothing_window_ms : float
        Window length (ms) for the ``'moving_average'`` smoothing method.

    Returns
    -------
    dict
        Per-channel bootstrap results and overall classification.
    """
    available_events = list(epochs.event_id.keys())

    # Resolve target events
    if target_stim is None or target_stim == 'probe':
        target_events = [k for k in available_events
                         if 'probe' in k.lower() and 's1_onset' in k.lower()]
    else:
        target_events = [k for k in available_events
                         if f'stim_id={target_stim}' in k
                         and 's1_onset' in k.lower()]

    # Resolve baseline events
    if not baseline_stims:
        baseline_events = [k for k in available_events
                           if 'irrelevant' in k.lower() and 's1_onset' in k.lower()]
    else:
        baseline_events = []
        for stim in baseline_stims:
            baseline_events += [k for k in available_events
                                 if f'stim_id={stim}' in k
                                 and 's1_onset' in k.lower()]

    if not target_events or not baseline_events:
        raise ValueError(
            f"Could not find target ({target_stim or 'probe'}) or baseline events. "
            f"Available: {available_events[:5]}"
        )

    target_epochs = epochs[target_events]
    baseline_epochs = epochs[baseline_events]

    times = epochs.times
    time_mask = (times >= tmin) & (times <= tmax)

    # Channel subset via index — no copy needed
    all_ch = list(epochs.ch_names)
    if channels:
        ch_names_to_use = [ch for ch in channels if ch in all_ch]
        ch_indices = np.array([all_ch.index(ch) for ch in ch_names_to_use])
    else:
        ch_names_to_use = all_ch
        ch_indices = np.arange(len(all_ch))

    sfreq = float(epochs.info['sfreq'])

    # (n_epochs, n_channels_total, n_times) → subset → amplitude per epoch
    target_data = target_epochs.get_data() * 1e6
    baseline_data = baseline_epochs.get_data() * 1e6

    def _rosenfeld_p2p(sub):
        """Rosenfeld peak-to-peak for data subset *(n_ep, n_ch, n_times)*.

        1. Positive peak in *[tmin, tmax]*.
        2. Negative trough between that peak and *p2p_tmax_negative*.
        3. Amplitude = positive peak − negative trough.
        """
        pos_indices = np.nonzero(time_mask)[0]
        neg_end = min(
            int(np.searchsorted(times, p2p_tmax_negative, side='right')),
            sub.shape[2],
        )

        pos_win = sub[:, :, pos_indices]
        pos_peak_val = pos_win.max(axis=2)
        pos_peak_local = np.argmax(pos_win, axis=2)
        pos_peak_global = pos_indices[pos_peak_local]

        n_ep, n_ch, _ = sub.shape
        result = np.empty((n_ep, n_ch))
        for ep in range(n_ep):
            for ch in range(n_ch):
                start = pos_peak_global[ep, ch]
                neg_window = sub[ep, ch, start:neg_end]
                neg_val = (neg_window.min() if len(neg_window) > 0
                           else pos_peak_val[ep, ch])
                result[ep, ch] = pos_peak_val[ep, ch] - neg_val
        return result

    def _epoch_amplitude(data_full, ch_idx, method):
        """Per-epoch, per-channel amplitude → *(n_epochs, len(ch_idx))*."""
        sub = data_full[:, ch_idx, :]
        need_smooth = method != 'mean' and smoothing_method is not None
        if need_smooth:
            sub = smooth_epochs_array(
                sub, sfreq=sfreq,
                method=smoothing_method,
                lowpass_hz=smoothing_lowpass_hz,
                moving_avg_ms=smoothing_window_ms,
            )
        if method == 'peak_to_peak':
            return _rosenfeld_p2p(sub)
        win = sub[:, :, time_mask]
        if method == 'peak_valley':
            return win.max(axis=2) - win.min(axis=2)
        if method == 'baseline_to_peak':
            return win.max(axis=2)
        return win.mean(axis=2)

    target_amp = _epoch_amplitude(target_data, ch_indices, amplitude_method)
    baseline_amp = _epoch_amplitude(baseline_data, ch_indices, amplitude_method)

    n_target = target_amp.shape[0]
    n_baseline = baseline_amp.shape[0]

    channel_results = []
    bootstrap_proportions = []
    target_label = target_stim if target_stim else "probe"

    for ch_idx, ch_name in enumerate(ch_names_to_use):
        target_ch = target_amp[:, ch_idx]
        baseline_ch = baseline_amp[:, ch_idx]

        count_target_greater = sum(
            np.random.choice(target_ch, size=n_target, replace=True).mean()
            > np.random.choice(baseline_ch, size=n_baseline, replace=True).mean()
            for _ in range(n_bootstrap)
        )

        proportion = count_target_greater / n_bootstrap
        bootstrap_proportions.append(proportion)

        if proportion >= threshold:
            classification = "Guilty"
            status = "🔴"
            confidence = "High" if proportion >= 0.95 else "Moderate"
        else:
            classification = "Innocent"
            status = "🟢"
            confidence = "High" if proportion <= 0.60 else (
                "Moderate" if proportion <= 0.75 else "Low"
            )

        p_value = 1.0 - proportion

        channel_results.append({
            'Channel': ch_name,
            'Status': status,
            'Bootstrap %': f"{proportion * 100:.1f}%",
            'p-value': round(p_value, 4),
            'Classification': classification,
            'Confidence': confidence,
            'Target > Baseline': f"{count_target_greater}/{n_bootstrap}",
        })

    max_proportion = max(bootstrap_proportions)
    max_channel = ch_names_to_use[bootstrap_proportions.index(max_proportion)]
    overall_p_value = 1.0 - max_proportion

    if max_proportion >= threshold:
        overall_classification = "GUILTY"
        overall_status = "🔴"
        verdict = (
            f"Participant likely recognized '{target_label}' "
            f"(p={overall_p_value:.4f}, "
            f"max: {max_proportion*100:.1f}% at {max_channel})"
        )
    else:
        overall_classification = "INNOCENT"
        overall_status = "🟢"
        verdict = (
            f"No clear recognition of '{target_label}' "
            f"(p={overall_p_value:.4f}, "
            f"max: {max_proportion*100:.1f}% at {max_channel})"
        )

    return {
        'channel_results': pd.DataFrame(channel_results),
        'overall_classification': overall_classification,
        'overall_status': overall_status,
        'verdict': verdict,
        'p_value': overall_p_value,
        'bootstrap_proportions': bootstrap_proportions,
        'max_proportion': max_proportion,
        'max_channel': max_channel,
        'n_probe_epochs': n_target,
        'n_irrelevant_epochs': n_baseline,
        'n_bootstrap': n_bootstrap,
        'threshold': threshold,
        'target_label': target_label,
        'amplitude_method': amplitude_method,
        'p2p_tmax_negative': p2p_tmax_negative,
        'smoothing_method': smoothing_method,
        'smoothing_lowpass_hz': smoothing_lowpass_hz,
        'smoothing_window_ms': smoothing_window_ms,
    }


def compute_individual_p300_window(
    raw,
    events,
    event_id,
    peak_channels='Pz',
    s2_tmin=-0.2,
    s2_tmax=0.8,
    baseline=(-0.2, 0),
    peak_search_tmin=0.25,
    peak_search_tmax=0.60,
    window_margin=0.15,
    detrend=None,
    erp_lowpass_hz=None,
    s1_rejection='autoreject',
    s1_threshold_uv=None,
    s1_adaptive_k=3.0,
    ar_n_jobs=1,
    s2_trial_rejection=False,
    s2_max_rt=None,
):
    """
    Compute individualized P300 time window from S2 target responses.

    Rejection mirrors ``run_pipeline`` for S1: optional behavioural drops,
    global *s1_rejection* (``autoreject`` / ``iqr`` / ``zscore`` /
    ``static`` / ``none``), then :func:`reject_noisy_baseline` using
    *s1_adaptive_k*.

    Parameters
    ----------
    raw, events, event_id
        Used to build S2-target :class:`mne.Epochs` on filtered data.
    peak_channels
        Channel name(s) for peak picking; multiple channels are averaged.
    s2_tmin
        Epoch start relative to S2 onset (seconds).
    s2_tmax
        Epoch end relative to S2 onset (seconds).
    baseline
        Baseline correction window.
    peak_search_tmin
        Start of peak search window (seconds).
    peak_search_tmax
        End of peak search window (seconds).
    window_margin
        Half-width of the individualized window (seconds).
    detrend
        Passed to :class:`mne.Epochs` (per-epoch detrend).
    erp_lowpass_hz
        Low-pass on the averaged ERP before peak search; ``None`` skips.
    s1_rejection
        Global artifact policy (same as S1): ``autoreject``, ``iqr``,
        ``zscore``, ``static``, ``none``.
    s1_threshold_uv
        MNE amplitude reject (µV) when *s1_rejection* is ``static``.
    s1_adaptive_k
        ``k`` for IQR/Z-score global rejection and baseline QC.
    ar_n_jobs
        Parallel jobs when *s1_rejection* is ``autoreject``.
    s2_trial_rejection
        Enable behavioural filtering via paired ``S2_response`` annotations.
    s2_max_rt
        Maximum allowed S2 reaction time (seconds).  ``None`` to skip
        the RT criterion.  Only used when *s2_trial_rejection* is
        ``True``.

    Returns
    -------
    dict
        Peak latency/amplitude, ``window_start`` / ``window_end``, ERPs,
        epoch counts, ``autoreject_info``, behavioural drop log.
    """
    s2_target_ids = {
        k: v for k, v in event_id.items()
        if 's2_onset' in k.lower() and 'target' in k.lower()
    }
    if not s2_target_ids:
        raise ValueError(
            "No S2_onset target events found in annotations. "
            f"Available (first 10): {list(event_id.keys())[:10]}"
        )

    reject = None
    if s1_rejection == 'static' and s1_threshold_uv:
        reject = dict(eeg=s1_threshold_uv * 1e-6)

    s2_epochs = mne.Epochs(
        raw, events,
        event_id=s2_target_ids,
        tmin=s2_tmin, tmax=s2_tmax,
        baseline=baseline,
        detrend=detrend,
        preload=True,
        reject=reject,
        verbose='ERROR',
    )

    if len(s2_epochs) == 0:
        raise ValueError("No S2 target epochs remaining after artifact rejection.")

    n_s2_perf_dropped = 0
    s2_perf_drop_log: list = []
    if s2_trial_rejection:
        s2_epochs, n_s2_perf_dropped, s2_perf_drop_log = (
            reject_s1_by_s2_performance(
                s2_epochs, raw, max_rt=s2_max_rt,
            )
        )
        if len(s2_epochs) == 0:
            raise ValueError(
                "No S2 target epochs remaining after S2 performance "
                f"rejection (dropped {n_s2_perf_dropped})."
            )

    n_before_ar = len(s2_epochs)

    s2_epochs, ar_info = _apply_s1_style_global_rejection(
        s2_epochs,
        s1_rejection=s1_rejection,
        adaptive_k=s1_adaptive_k,
        ar_n_jobs=ar_n_jobs,
    )
    if len(s2_epochs) == 0:
        raise ValueError(
            "No S2 target epochs remaining after S1-style global rejection."
        )

    s2_epochs, _, _ = reject_noisy_baseline(
        s2_epochs, k=s1_adaptive_k,
    )
    if len(s2_epochs) == 0:
        raise ValueError(
            "No S2 target epochs remaining after baseline QC."
        )

    n_after_ar = len(s2_epochs)
    s2_erp = s2_epochs.average()

    # Smoothed copy for peak detection (original preserved for display)
    if erp_lowpass_hz is not None:
        s2_erp_smooth = s2_erp.copy().filter(
            l_freq=None, h_freq=erp_lowpass_hz, verbose='ERROR',
        )
    else:
        s2_erp_smooth = s2_erp

    # --- Resolve peak_channels (single str or list) ---
    if isinstance(peak_channels, str):
        peak_channels = [peak_channels]

    ch_names = s2_erp_smooth.ch_names
    valid_channels = [ch for ch in peak_channels if ch in ch_names]
    if not valid_channels:
        valid_channels = [ch_names[0]]

    ch_indices = [ch_names.index(ch) for ch in valid_channels]
    erp_data_uv = s2_erp_smooth.data[ch_indices] * 1e6  # (n_ch, n_times)
    erp_data = erp_data_uv.mean(axis=0)  # average across channels
    times = s2_erp_smooth.times

    if len(valid_channels) == 1:
        peak_channel_label = valid_channels[0]
    else:
        peak_channel_label = f"mean({','.join(valid_channels)})"

    search_mask = (times >= peak_search_tmin) & (times <= peak_search_tmax)
    search_data = erp_data[search_mask]
    search_times = times[search_mask]

    if len(search_data) == 0:
        raise ValueError(
            f"No time points in search window "
            f"[{peak_search_tmin}, {peak_search_tmax}] s."
        )

    peak_local_idx = int(np.argmax(search_data))
    peak_time = float(search_times[peak_local_idx])
    peak_amplitude = float(search_data[peak_local_idx])

    window_start = max(0.0, peak_time - window_margin)
    window_end = peak_time + window_margin

    return {
        'peak_time': peak_time,
        'peak_amplitude': peak_amplitude,
        'peak_channel': peak_channel_label,
        'peak_channels_used': valid_channels,
        'window_start': window_start,
        'window_end': window_end,
        'window_margin': window_margin,
        'n_s2_epochs': n_after_ar,
        'n_s2_epochs_before_ar': n_before_ar,
        'n_s2_epochs_after_ar': n_after_ar,
        'autoreject_info': ar_info,
        'n_s2_perf_dropped': n_s2_perf_dropped,
        's2_perf_drop_log': s2_perf_drop_log,
        's2_erp': s2_erp,
        's2_erp_smooth': s2_erp_smooth if erp_lowpass_hz is not None else None,
    }


def extract_block_boundaries(raw):
    """
    Extract block start/end times from annotations.

    Parses ``block_start|block=N`` and ``block_end|block=N`` annotations
    embedded in the FIF file.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data with annotations.

    Returns
    -------
    dict[int, dict]
        Mapping ``{block_number: {'start': float, 'end': float}}``.
        Times are in seconds relative to ``raw.first_time``.
    """
    blocks: dict = {}
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        if desc.startswith('block_start|block='):
            block_num = int(desc.split('block=')[1].split(',')[0])
            blocks.setdefault(block_num, {})['start'] = float(onset)
        elif desc.startswith('block_end|block='):
            block_num = int(desc.split('block=')[1].split(',')[0])
            blocks.setdefault(block_num, {})['end'] = float(onset)

    # Fill missing end times with next block start or recording end
    sorted_nums = sorted(blocks)
    for i, num in enumerate(sorted_nums):
        if 'end' not in blocks[num]:
            if i + 1 < len(sorted_nums):
                blocks[num]['end'] = blocks[sorted_nums[i + 1]].get(
                    'start', raw.times[-1]
                )
            else:
                blocks[num]['end'] = raw.times[-1]
        if 'start' not in blocks[num]:
            blocks[num]['start'] = 0.0

    return blocks


def extract_block_events(raw, block_start, block_end):
    """
    Collect event annotations that fall within a block time window.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data with annotations.
    block_start : float
        Block start time (seconds).
    block_end : float
        Block end time (seconds).

    Returns
    -------
    list[dict]
        List of ``{'time': float, 'label': str, 'type': str}`` dicts
        for annotations inside the window.
    """
    events = []
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        t = float(onset)
        if block_start <= t <= block_end:
            if 'S1_onset' in desc:
                etype = 'probe' if 'probe' in desc.lower() else 'irrelevant'
            elif 'S2_onset' in desc:
                etype = 'S2'
            elif 'block_start' in desc or 'block_end' in desc:
                etype = 'block'
            else:
                etype = 'other'
            events.append({'time': t, 'label': desc.split('|')[0], 'type': etype})
    return events


def plot_block_signal(raw, block_start, block_end, events=None, title=""):
    """
    Plot full continuous signal for a single block with optional event markers.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    block_start : float
        Start time (seconds).
    block_end : float
        End time (seconds).
    events : list[dict] or None
        Events from :func:`extract_block_events`.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    sfreq = raw.info['sfreq']
    start_sample = max(0, int(block_start * sfreq))
    end_sample = min(int(block_end * sfreq), len(raw.times))

    data = raw.get_data()[:, start_sample:end_sample] * 1e6
    times = raw.times[start_sample:end_sample]

    n_ch = len(raw.ch_names)
    fig, axes = plt.subplots(n_ch, 1, figsize=(14, 2.5 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]

    event_colors = {
        'probe': '#d62728',
        'irrelevant': '#1f77b4',
        'S2': '#2ca02c',
        'block': '#ff7f0e',
        'other': '#7f7f7f',
    }

    for idx, (ch_name, ax) in enumerate(zip(raw.ch_names, axes)):
        ax.plot(times, data[idx], linewidth=0.4, color='#1f77b4')
        ax.set_ylabel(f'{ch_name}\n(\u00b5V)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(times[0], times[-1])

        if events:
            added_labels: set = set()
            for ev in events:
                if ev['type'] in ('block',):
                    continue
                color = event_colors.get(ev['type'], '#7f7f7f')
                label = ev['type'] if ev['type'] not in added_labels else None
                ax.axvline(ev['time'], color=color, alpha=0.5,
                           linewidth=0.7, linestyle='--', label=label)
                if label:
                    added_labels.add(ev['type'])

        if idx == 0 and ax.get_legend_handles_labels()[1]:
            ax.legend(loc='upper right', fontsize=8, ncol=4)

    axes[-1].set_xlabel('Time (s)', fontsize=11)
    fig.suptitle(title or f'Block signal ({block_start:.1f}s \u2013 {block_end:.1f}s)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_raw_data(raw, duration=10.0, start=0.0):
    """Plot raw EEG data."""
    fig, axes = plt.subplots(len(raw.ch_names), 1, 
                             figsize=(12, 2*len(raw.ch_names)),
                             sharex=True)
    
    if len(raw.ch_names) == 1:
        axes = [axes]
    
    # Get data segment
    start_sample = int(start * raw.info['sfreq'])
    end_sample = int((start + duration) * raw.info['sfreq'])
    end_sample = min(end_sample, len(raw.times))
    
    data = raw.get_data()[:, start_sample:end_sample] * 1e6  # µV
    times = raw.times[start_sample:end_sample]
    
    for idx, (ch_name, ax) in enumerate(zip(raw.ch_names, axes)):
        ax.plot(times, data[idx], linewidth=0.5, color='#1f77b4')
        ax.set_ylabel(f'{ch_name}\n(µV)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(times[0], times[-1])
    
    axes[-1].set_xlabel('Time (s)', fontsize=11)
    fig.suptitle(f'Raw EEG Data ({start:.1f}s to {start+duration:.1f}s)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_psd(raw):
    """Plot power spectral density."""
    fig = raw.compute_psd(fmax=100).plot(show=False)
    fig.suptitle('Power Spectral Density', fontsize=14, fontweight='bold')
    return fig


def plot_psd_single_channel(raw, ch_name, raw_filtered=None, fmax=100):
    """Plot PSD for a single channel, optionally comparing raw vs filtered.

    :param raw: Raw EEG data.
    :param ch_name: Channel name to plot.
    :param raw_filtered: Filtered EEG data for comparison (optional).
    :param fmax: Maximum frequency to display.
    :return: Matplotlib figure.
    """
    has_filt = raw_filtered is not None
    ncols = 2 if has_filt else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 3.5))
    if ncols == 1:
        axes = [axes]

    raw.compute_psd(fmax=fmax, picks=[ch_name]).plot(
        axes=axes[0], show=False,
    )
    axes[0].set_title(f"{ch_name} — Raw", fontweight='bold')

    if has_filt:
        raw_filtered.compute_psd(fmax=fmax, picks=[ch_name]).plot(
            axes=axes[1], show=False,
        )
        axes[1].set_title(f"{ch_name} — Filtered", fontweight='bold')

    fig.suptitle(
        f"PSD — {ch_name}", fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    return fig


def plot_s2_correct_vs_incorrect(raw, tmin=-0.2, tmax=0.8,
                                 baseline=(-0.2, 0), detrend=0):
    """
    Create S2 epochs split by response correctness and plot the comparison.

    :param raw: Raw EEG data with annotations.
    :type raw: mne.io.Raw
    :param tmin: Epoch start relative to S2 onset (seconds).
    :type tmin: float
    :param tmax: Epoch end relative to S2 onset (seconds).
    :type tmax: float
    :param baseline: Baseline correction window or ``None``.
    :type baseline: tuple or None
    :param detrend: Detrend mode — ``None``, ``0`` (DC offset), or ``1``
        (linear). Default ``0``.
    :type detrend: int or None
    :return: ``(fig, stats)`` — matplotlib figure and a dict with epoch
        counts (``n_correct``, ``n_incorrect``, ``n_no_response``).
    :rtype: tuple[matplotlib.figure.Figure, dict]
    """
    import re

    s2_info = _parse_s2_responses(raw)

    s2_onset_descs = [
        d for d in raw.annotations.description
        if 's2_onset' in d.lower()
    ]
    if not s2_onset_descs:
        return None, {'n_correct': 0, 'n_incorrect': 0, 'n_no_response': 0}

    events, event_id = mne.events_from_annotations(raw, verbose='ERROR')
    s2_event_id = {
        k: v for k, v in event_id.items()
        if 's2_onset' in k.lower()
    }
    if not s2_event_id:
        return None, {'n_correct': 0, 'n_incorrect': 0, 'n_no_response': 0}

    all_s2_epochs = mne.Epochs(
        raw, events, event_id=s2_event_id,
        tmin=tmin, tmax=tmax, baseline=baseline, detrend=detrend,
        preload=True, verbose='ERROR',
    )

    rev_id = {v: k for k, v in all_s2_epochs.event_id.items()}
    correct_idx = []
    incorrect_idx = []
    no_resp_idx = []
    for idx in range(len(all_s2_epochs)):
        ev_code = all_s2_epochs.events[idx, 2]
        desc = rev_id.get(ev_code, '')
        trial_m = re.search(r'trial=(\d+)', desc)
        if trial_m is None:
            continue
        trial_num = int(trial_m.group(1))
        info = s2_info.get(trial_num)
        if info is None:
            no_resp_idx.append(idx)
        elif info['correct']:
            correct_idx.append(idx)
        else:
            incorrect_idx.append(idx)

    stats = {
        'n_correct': len(correct_idx),
        'n_incorrect': len(incorrect_idx),
        'n_no_response': len(no_resp_idx),
    }

    has_correct = len(correct_idx) > 0
    has_incorrect = len(incorrect_idx) > 0
    if not has_correct and not has_incorrect:
        return None, stats

    data_all = all_s2_epochs.get_data() * 1e6
    times_ms = all_s2_epochs.times * 1000
    ch_names = all_s2_epochs.ch_names
    n_ch = len(ch_names)

    fig, axes = plt.subplots(n_ch, 1, figsize=(10, 3 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]

    for ch_idx, (ch, ax) in enumerate(zip(ch_names, axes)):
        if has_correct:
            avg_ok = data_all[correct_idx, ch_idx, :].mean(axis=0)
            sem_ok = (data_all[correct_idx, ch_idx, :].std(axis=0)
                      / np.sqrt(len(correct_idx)))
            ax.plot(times_ms, avg_ok, color='#2ca02c', linewidth=2,
                    label=f'Correct (n={len(correct_idx)})')
            ax.fill_between(times_ms, avg_ok - sem_ok, avg_ok + sem_ok,
                            color='#2ca02c', alpha=0.15)
        if has_incorrect:
            avg_bad = data_all[incorrect_idx, ch_idx, :].mean(axis=0)
            sem_bad = (data_all[incorrect_idx, ch_idx, :].std(axis=0)
                       / np.sqrt(len(incorrect_idx)))
            ax.plot(times_ms, avg_bad, color='#d62728', linewidth=2,
                    label=f'Incorrect (n={len(incorrect_idx)})')
            ax.fill_between(times_ms, avg_bad - sem_bad, avg_bad + sem_bad,
                            color='#d62728', alpha=0.15)

        ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.4)
        ax.set_ylabel(f'{ch}\n(\u00b5V)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel('Time (ms)')
    fig.suptitle('S2 ERP: Correct vs Incorrect Responses',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig, stats


def plot_epochs(epochs, n_epochs=20):
    """Plot individual epochs."""
    fig = epochs.plot(
        n_epochs=min(n_epochs, len(epochs)),
        n_channels=len(epochs.ch_names),
        scalings='auto',
        show=False,
        block=False
    )
    return fig


def plot_erps(probe_erp, irrelevant_erp, p300_window=(0.3, 0.6),
              erp_lowpass_hz=None):
    """
    Plot ERPs with P300 window.

    When *erp_lowpass_hz* is set, the original waveforms are drawn faded and
    smoothed waveforms are overlaid with full opacity.

    Parameters
    ----------
    probe_erp : mne.Evoked
        Probe (target) ERP.
    irrelevant_erp : mne.Evoked
        Irrelevant (baseline) ERP.
    p300_window : tuple
        (start, end) in seconds.
    erp_lowpass_hz : float or None
        Low-pass cutoff for smoothed overlay.  ``None`` disables smoothing.
    """
    has_smooth = erp_lowpass_hz is not None
    if has_smooth:
        probe_smooth = probe_erp.copy().filter(
            l_freq=None, h_freq=erp_lowpass_hz, verbose='ERROR',
        )
        irr_smooth = irrelevant_erp.copy().filter(
            l_freq=None, h_freq=erp_lowpass_hz, verbose='ERROR',
        )

    fig, axes = plt.subplots(1, len(probe_erp.ch_names),
                             figsize=(5 * len(probe_erp.ch_names), 4))

    if len(probe_erp.ch_names) == 1:
        axes = [axes]

    times = probe_erp.times * 1000

    for idx, (ch_name, ax) in enumerate(zip(probe_erp.ch_names, axes)):
        probe_data = probe_erp.data[idx] * 1e6
        irrelevant_data = irrelevant_erp.data[idx] * 1e6

        alpha_orig = 0.25 if has_smooth else 0.8
        lw_orig = 1.0 if has_smooth else 2.0

        ax.plot(times, probe_data, color='#d62728',
                linewidth=lw_orig, alpha=alpha_orig,
                label='Probe (orig)' if has_smooth else 'Probe')
        ax.plot(times, irrelevant_data, color='#1f77b4',
                linewidth=lw_orig, alpha=alpha_orig,
                label='Irrelevant (orig)' if has_smooth else 'Irrelevant')

        if has_smooth:
            ps = probe_smooth.data[idx] * 1e6
            irs = irr_smooth.data[idx] * 1e6
            diff_s = ps - irs
            ax.plot(times, ps,
                    label=f'Probe (LP {erp_lowpass_hz} Hz)',
                    color='#d62728', linewidth=2, alpha=0.9)
            ax.plot(times, irs,
                    label=f'Irrelevant (LP {erp_lowpass_hz} Hz)',
                    color='#1f77b4', linewidth=2, alpha=0.9)
            ax.plot(times, diff_s,
                    label=f'Difference (LP {erp_lowpass_hz} Hz)',
                    color='#2ca02c', linewidth=2, linestyle='--', alpha=0.9)
        else:
            diff_data = probe_data - irrelevant_data
            ax.plot(times, diff_data, label='Difference', color='#2ca02c',
                    linewidth=2, linestyle='--', alpha=0.8)

        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(0, color='black', linestyle='--', linewidth=1,
                   label='Stimulus')

        ax.axvspan(p300_window[0] * 1000, p300_window[1] * 1000,
                   alpha=0.1, color='gray', label='P300 window')

        ax.set_xlabel('Time (ms)', fontsize=11)
        ax.set_ylabel('Amplitude (µV)', fontsize=11)
        ax.set_title(f'{ch_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    title = 'Event-Related Potentials (ERPs)'
    if has_smooth:
        title += f'  [LP smoothing: {erp_lowpass_hz} Hz]'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_topography(probe_erp, irrelevant_erp, times=[0.3, 0.4, 0.5]):
    """Plot topographic maps at specific time points."""
    try:
        fig, axes = plt.subplots(2, len(times), 
                                figsize=(4*len(times), 6))
        
        # Probe topography
        probe_erp.plot_topomap(
            times=times,
            axes=axes[0] if len(times) > 1 else [axes[0]],
            show=False,
            colorbar=True,
            title='Probe'
        )
        
        # Irrelevant topography
        irrelevant_erp.plot_topomap(
            times=times,
            axes=axes[1] if len(times) > 1 else [axes[1]],
            show=False,
            colorbar=True,
            title='Irrelevant'
        )
        
        fig.suptitle('Topographic Maps', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.warning(f"Cannot create topographic maps: {str(e)}")
        return None


# ============================================================================
# Pipeline Runner
# ============================================================================

def run_pipeline(raw, events, event_id, cfg):
    """
    Execute the full CTP-BAD pipeline on a single recording.

    All processing uses local copies — the passed *raw* is not mutated.

    Parameters
    ----------
    raw
        Loaded ``mne.io.Raw``.
    events
        MNE events array.
    event_id
        Event-ID mapping dict.
    cfg
        Dict with pipeline configuration keys (see Quick Pipeline page).
        If ``s2_match_s1_preprocessing`` is ``True``, S2 target epochs for
        the individual P300 window use ``s1_tmin``, ``s1_tmax``,
        ``s1_baseline``, and ``s1_detrend`` instead of ``s2_*``.

    Returns
    -------
    dict
        Full pipeline results including intermediate data.
    """
    log = []

    # ------------------------------------------------------------------
    # Step 1 — Filtering
    # ------------------------------------------------------------------
    filter_preset = cfg.get('filter_preset', 'aggressive')
    if filter_preset == 'skip':
        raw_f = raw.copy()
        log.append("Filtering: skipped")
    elif filter_preset == 'aggressive':
        agg_notch = cfg.get('notch_freqs', [50])
        agg_hp = cfg.get('hp_cutoff', 0.7)
        agg_lp = cfg.get('lp_cutoff', 30.0)
        agg_order = cfg.get('iir_order', 4)
        raw_f = apply_filters(
            raw, notch_freqs=agg_notch, lowcut=agg_hp, highcut=agg_lp,
            method='iir', iir_order=agg_order,
        )
        log.append(
            f"Filtering: Aggressive (IIR {agg_hp}-{agg_lp} Hz, "
            f"order {agg_order}, notch {agg_notch or 'off'})"
        )
    else:
        raw_f = apply_filters(
            raw,
            notch_freqs=cfg.get('notch_freqs'),
            lowcut=cfg.get('hp_cutoff'),
            highcut=cfg.get('lp_cutoff'),
            method=cfg.get('filter_method', 'fir'),
            iir_order=cfg.get('iir_order', 4),
        )
        log.append(
            f"Filtering: Custom (HP={cfg.get('hp_cutoff')}, "
            f"LP={cfg.get('lp_cutoff')}, "
            f"method={cfg.get('filter_method', 'fir')})"
        )

    # ------------------------------------------------------------------
    # Step 2 — S2 epoching params (timing / baseline for individual window)
    # ------------------------------------------------------------------
    s2_detrend_map = {'None': None, 'DC offset (0)': 0, 'Linear (1)': 1}
    if cfg.get('s2_match_s1_preprocessing'):
        s2_tmin = cfg.get('s1_tmin', -0.2)
        s2_tmax = cfg.get('s1_tmax', 0.8)
        s2_baseline = (
            (s2_tmin, 0) if cfg.get('s1_baseline', True) else None
        )
        s2_detrend = s2_detrend_map.get(
            cfg.get('s1_detrend', 'DC offset (0)'), 0,
        )
        log.append(
            "S2 epoching mirrors S1 (tmin/tmax, baseline on/off, detrend)."
        )
    else:
        s2_tmin = cfg.get('s2_tmin', -0.2)
        s2_tmax = cfg.get('s2_tmax', 0.8)
        s2_baseline = (
            (s2_tmin, 0) if cfg.get('s2_baseline', True) else None
        )
        s2_detrend = s2_detrend_map.get(
            cfg.get('s2_detrend', 'DC offset (0)'), 0,
        )
    s1_rej = cfg.get('s1_rejection', 'autoreject')

    # ------------------------------------------------------------------
    # Step 3 — Individual P300 window
    # ------------------------------------------------------------------
    use_individual = cfg.get('use_individual_window', True)
    ip_result = None

    if use_individual:
        ip_result = compute_individual_p300_window(
            raw_f, events, event_id,
            peak_channels=cfg.get('peak_channels', ['Pz']),
            s2_tmin=s2_tmin,
            s2_tmax=s2_tmax,
            baseline=s2_baseline,
            peak_search_tmin=cfg.get('peak_search_tmin', 0.25),
            peak_search_tmax=cfg.get('peak_search_tmax', 0.60),
            window_margin=cfg.get('window_margin', 0.15),
            detrend=s2_detrend,
            erp_lowpass_hz=cfg.get('s2_erp_lowpass_hz'),
            s1_rejection=s1_rej,
            s1_threshold_uv=(
                cfg.get('s1_threshold_uv') if s1_rej == 'static' else None
            ),
            s1_adaptive_k=cfg.get('s1_adaptive_k', 3.0),
            ar_n_jobs=cfg.get('ar_n_jobs', 1),
            s2_trial_rejection=cfg.get('s2_trial_rejection', True),
            s2_max_rt=cfg.get('s2_max_rt'),
        )
        p300_tmin = ip_result['window_start']
        p300_tmax = ip_result['window_end']
        n_perf = ip_result.get('n_s2_perf_dropped', 0)
        if n_perf:
            perf_reasons: dict = {}
            for _, _, r in ip_result.get('s2_perf_drop_log', []):
                perf_reasons[r] = perf_reasons.get(r, 0) + 1
            perf_str = ', '.join(
                f'{r}: {c}' for r, c in perf_reasons.items()
            )
            log.append(
                f"S2 performance rejection: dropped {n_perf} "
                f"({perf_str})"
            )
        log.append(
            f"S2 epochs: {ip_result['n_s2_epochs_before_ar']} → "
            f"{ip_result['n_s2_epochs_after_ar']} "
            f"(S1-style rejection: {s1_rej})"
        )
        log.append(
            f"Individual window: {p300_tmin*1000:.0f}–{p300_tmax*1000:.0f} ms "
            f"(peak {ip_result['peak_time']*1000:.0f} ms on "
            f"{ip_result['peak_channel']})"
        )
    else:
        p300_tmin = cfg.get('manual_tmin', 0.3)
        p300_tmax = cfg.get('manual_tmax', 0.6)
        log.append(
            f"Individual window: disabled → manual "
            f"{p300_tmin*1000:.0f}–{p300_tmax*1000:.0f} ms"
        )

    # ------------------------------------------------------------------
    # Step 4 — S1 Probe/Irrelevant epochs + rejection
    # ------------------------------------------------------------------
    s1_tmin = cfg.get('s1_tmin', -0.2)
    s1_tmax = cfg.get('s1_tmax', 0.8)
    s1_baseline = (s1_tmin, 0) if cfg.get('s1_baseline', True) else None
    s1_detrend = s2_detrend_map.get(cfg.get('s1_detrend', 'DC offset (0)'), 0)

    s1_static_thresh = (
        cfg.get('s1_threshold_uv') if s1_rej == 'static' else None
    )
    epochs = create_epochs(
        raw_f, events, event_id,
        tmin=s1_tmin, tmax=s1_tmax,
        baseline=s1_baseline,
        reject_threshold_uv=s1_static_thresh,
        detrend=s1_detrend,
    )
    n_s1_before = len(epochs)

    # S2-performance-based rejection (drop S1 trials with incorrect /
    # too-slow S2 responses) — applied before artifact rejection so that
    # only behaviourally valid trials enter the analysis.
    s2_reject = cfg.get('s2_trial_rejection', True)
    s2_max_rt = cfg.get('s2_max_rt')
    n_s2_dropped = 0
    s2_drop_log = []
    if s2_reject:
        epochs, n_s2_dropped, s2_drop_log = reject_s1_by_s2_performance(
            epochs, raw, max_rt=s2_max_rt,
        )
        n_after_s2 = len(epochs)
        reasons = {}
        for _, _, reason in s2_drop_log:
            reasons[reason] = reasons.get(reason, 0) + 1
        reason_str = ', '.join(f'{r}: {c}' for r, c in reasons.items())
        log.append(
            f"S2 trial rejection: {n_s1_before} → {n_after_s2} "
            f"(dropped {n_s2_dropped}: {reason_str or 'none'})"
        )
        n_s1_before = n_after_s2

    if s1_rej in ('iqr', 'zscore'):
        epochs, _, _, _ = adaptive_reject_epochs(
            epochs,
            method=s1_rej,
            k=cfg.get('s1_adaptive_k', 3.0),
        )
    elif s1_rej == 'autoreject' and _AUTOREJECT_AVAILABLE:
        epochs, _, _ = autoreject_clean_epochs(
            epochs, n_jobs=cfg.get('ar_n_jobs', 1),
        )

    n_after_art = len(epochs)

    epochs, n_bl_dropped, _ = reject_noisy_baseline(
        epochs, k=cfg.get('s1_adaptive_k', 3.0),
    )

    n_s1_after = len(epochs)
    log.append(
        f"S1 epochs: {n_s1_before} → {n_after_art} (rejection: {s1_rej})"
        f" → {n_s1_after} (baseline QC: dropped {n_bl_dropped})"
    )

    if n_s1_after == 0:
        raise ValueError("No S1 epochs remaining after rejection.")

    # Compute S1 probe / irrelevant ERPs for result visualisation
    available_keys = list(epochs.event_id.keys())
    _tgt_stim = cfg.get('target_stim')
    if _tgt_stim is None or _tgt_stim == 'probe':
        _probe_keys = [
            k for k in available_keys
            if 'probe' in k.lower() and 's1_onset' in k.lower()
        ]
    else:
        _probe_keys = [
            k for k in available_keys
            if f'stim_id={_tgt_stim}' in k and 's1_onset' in k.lower()
        ]
    _bl_stims = cfg.get('baseline_stims')
    if not _bl_stims:
        _irr_keys = [
            k for k in available_keys
            if 'irrelevant' in k.lower() and 's1_onset' in k.lower()
        ]
    else:
        _irr_keys = [
            k for k in available_keys
            for s in _bl_stims if f'stim_id={s}' in k
            and 's1_onset' in k.lower()
        ]
    n_probe = len(epochs[_probe_keys]) if _probe_keys else 0
    min_probe = cfg.get('min_probe_epochs', _MIN_PROBE_EPOCHS)
    if n_probe < min_probe:
        raise ValueError(
            f"Only {n_probe} probe epochs remaining (minimum {min_probe}). "
            f"P300 SNR too low — recording rejected."
        )

    probe_erp = (
        epochs[_probe_keys].average() if _probe_keys else None
    )
    irrelevant_erp = (
        epochs[_irr_keys].average() if _irr_keys else None
    )

    # ------------------------------------------------------------------
    # Step 5 — CTP-BAD bootstrap
    # ------------------------------------------------------------------
    amp_method_map = {
        'Mean': 'mean',
        'Peak-to-peak (Rosenfeld)': 'peak_to_peak',
        'Peak-to-Peak (Peak-Valley)': 'peak_valley',
        'Baseline-to-peak': 'baseline_to_peak',
    }
    amp_label = cfg.get('amplitude_method', 'Peak-to-peak (Rosenfeld)')
    amp_method = amp_method_map.get(amp_label, amp_label)

    smooth_map = {
        'None': None,
        'Low-pass (Butterworth)': 'lowpass',
        'Moving average': 'moving_average',
    }
    smooth_label = cfg.get('smoothing_method', 'Low-pass (Butterworth)')
    smooth_method = smooth_map.get(smooth_label, smooth_label)

    bad_channels = cfg.get(
        'bad_channels',
        cfg.get('peak_channels', ['Pz']),
    )

    bad_results = ctp_bad_analysis(
        epochs,
        tmin=p300_tmin,
        tmax=p300_tmax,
        n_bootstrap=cfg.get('n_bootstrap', 10000),
        threshold=cfg.get('guilty_threshold', 0.75),
        channels=bad_channels,
        target_stim=cfg.get('target_stim'),
        baseline_stims=cfg.get('baseline_stims'),
        amplitude_method=amp_method,
        p2p_tmax_negative=cfg.get('p2p_tmax_negative', 0.9),
        smoothing_method=smooth_method,
        smoothing_lowpass_hz=cfg.get('smoothing_lp_hz', 12.0),
        smoothing_window_ms=cfg.get('smoothing_ma_ms', 100.0),
    )

    log.append(
        f"CTP-BAD: {bad_results['overall_classification']} "
        f"(p={bad_results['p_value']:.4f}, "
        f"method={amp_label})"
    )

    return {
        'raw': raw,
        'raw_filtered': raw_f,
        'epochs': epochs,
        'individual_p300_window': ip_result,
        'probe_erp': probe_erp,
        'irrelevant_erp': irrelevant_erp,
        'bad_results': bad_results,
        'p300_tmin': p300_tmin,
        'p300_tmax': p300_tmax,
        'n_s1_before': n_s1_before,
        'n_s1_after': n_s1_after,
        'n_s2_dropped': n_s2_dropped,
        's2_drop_log': s2_drop_log,
        'log': log,
    }


# ============================================================================
# Pipeline Result Display Helpers
# ============================================================================

def _display_pipeline_results(result):
    """Render pipeline results inside Streamlit."""
    bad = result['bad_results']
    ip = result.get('individual_p300_window')

    # --- Verdict ---
    st.markdown("---")
    st.subheader("Classification Result")

    v1, v2, v3 = st.columns([1, 1, 2])
    with v1:
        st.metric("Classification", bad['overall_classification'])
    with v2:
        st.metric("p-value", f"{bad['p_value']:.4f}")
    with v3:
        verdict_color = (
            "red" if bad['overall_classification'] == "GUILTY" else "green"
        )
        st.markdown(
            f"**Verdict:** :{verdict_color}[{bad['verdict']}]"
        )

    # --- Pipeline log ---
    with st.expander("Pipeline log"):
        for msg in result.get('log', []):
            st.write(f"• {msg}")

    # --- S2 ERP plot (if individual window was used) ---
    if ip is not None:
        with st.expander("S2 ERP — Individual P300 Window"):
            s2_erp = ip['s2_erp']
            s2_erp_smooth = ip.get('s2_erp_smooth')
            fig_s2, ax_s2 = plt.subplots(figsize=(10, 4))
            s2_times_ms = s2_erp.times * 1000

            _used = ip.get('peak_channels_used', [ip['peak_channel']])
            _idxs = [
                s2_erp.ch_names.index(c)
                for c in _used if c in s2_erp.ch_names
            ] or [0]
            s2_data = s2_erp.data[_idxs].mean(axis=0) * 1e6

            if s2_erp_smooth is not None:
                ax_s2.plot(
                    s2_times_ms, s2_data, color='#1f77b4',
                    linewidth=1, alpha=0.3,
                    label=f"S2 ERP orig ({ip['peak_channel']})",
                )
                sm_data = (
                    s2_erp_smooth.data[_idxs].mean(axis=0) * 1e6
                )
                ax_s2.plot(
                    s2_times_ms, sm_data, color='#1f77b4',
                    linewidth=2, alpha=0.9,
                    label=f"S2 ERP smoothed ({ip['peak_channel']})",
                )
            else:
                ax_s2.plot(
                    s2_times_ms, s2_data, color='#1f77b4',
                    linewidth=2,
                    label=f"S2 target ERP ({ip['peak_channel']})",
                )

            ax_s2.axvline(
                ip['peak_time'] * 1000, color='red', linestyle='--',
                linewidth=1.5,
                label=f"Peak ({ip['peak_time']*1000:.0f} ms)",
            )
            ax_s2.axvspan(
                ip['window_start'] * 1000, ip['window_end'] * 1000,
                alpha=0.15, color='orange', label='Individual window',
            )
            ax_s2.axhline(0, color='black', linewidth=0.5)
            ax_s2.axvline(0, color='black', linestyle='--', linewidth=1)
            ax_s2.set_xlabel('Time (ms)')
            ax_s2.set_ylabel('Amplitude (µV)')
            ax_s2.set_title('S2 Target ERP — Individual P300 Window')
            ax_s2.legend(loc='best', fontsize=9)
            ax_s2.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_s2)
            plt.close()

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("S2 epochs (before)", ip['n_s2_epochs_before_ar'])
            mc2.metric("S2 epochs (after)", ip['n_s2_epochs_after_ar'])
            mc3.metric(
                "Window",
                f"{ip['window_start']*1000:.0f}–"
                f"{ip['window_end']*1000:.0f} ms",
            )

    # --- S1 ERP plot (Probe vs Irrelevant) ---
    _probe = result.get('probe_erp')
    _irr = result.get('irrelevant_erp')
    if _probe is not None and _irr is not None:
        with st.expander("S1 ERP — Probe vs Irrelevant (P300 window)"):
            p300_w = (result.get('p300_tmin', 0.3),
                      result.get('p300_tmax', 0.6))
            fig_erp = plot_erps(_probe, _irr, p300_window=p300_w)
            st.pyplot(fig_erp)
            plt.close()

    # --- PSD comparison (Raw vs Filtered) ---
    _raw_orig = result.get('raw')
    _raw_filt = result.get('raw_filtered')
    if _raw_orig is not None and _raw_filt is not None:
        with st.expander("PSD Comparison (Raw vs Filtered)"):
            fig_psd, axes_psd = plt.subplots(1, 2, figsize=(14, 4))
            _raw_orig.compute_psd(fmax=100).plot(
                axes=axes_psd[0], show=False,
            )
            axes_psd[0].set_title("Raw", fontweight='bold')
            _raw_filt.compute_psd(fmax=100).plot(
                axes=axes_psd[1], show=False,
            )
            axes_psd[1].set_title("Filtered", fontweight='bold')
            fig_psd.suptitle(
                "PSD Comparison", fontsize=14, fontweight='bold',
            )
            plt.tight_layout()
            st.pyplot(fig_psd)
            plt.close()

            # Per-electrode PSD
            _uid = result.get('filename', id(result))
            psd_ch_pipe = st.selectbox(
                "Per-electrode PSD",
                options=_raw_orig.ch_names,
                key=f"pipe_psd_ch_{_uid}",
            )
            fig_ch_psd = plot_psd_single_channel(
                _raw_orig, psd_ch_pipe, raw_filtered=_raw_filt,
            )
            st.pyplot(fig_ch_psd)
            plt.close()

    # --- Per-channel results ---
    with st.expander("Per-channel CTP-BAD results", expanded=True):
        st.dataframe(
            bad['channel_results'], width='stretch',
        )

    # --- Bootstrap chart ---
    with st.expander("Bootstrap proportions chart"):
        fig, ax = plt.subplots(figsize=(10, 5))
        channels = list(bad['channel_results']['Channel'])
        proportions = [p * 100 for p in bad['bootstrap_proportions']]
        colors = [
            '#d62728' if p >= bad['threshold'] * 100 else '#2ca02c'
            for p in proportions
        ]
        bars = ax.bar(
            channels, proportions, color=colors, alpha=0.7,
            edgecolor='black', linewidth=1.5,
        )
        ax.axhline(
            bad['threshold'] * 100, color='black', linestyle='--',
            linewidth=2,
            label=f"Threshold ({bad['threshold']*100:.0f}%)",
        )
        ax.axhline(50, color='gray', linestyle=':', linewidth=1,
                    label='Chance (50%)')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Bootstrap Proportion (%)')
        ax.set_title('CTP-BAD: Probe > Irrelevant (%)')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        for bar, prop in zip(bars, proportions):
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 1,
                f'{prop:.1f}%', ha='center', va='bottom',
                fontweight='bold',
            )
        st.pyplot(fig)
        plt.close()

    # --- Summary metrics ---
    with st.expander("Analysis details"):
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Probe epochs", bad['n_probe_epochs'])
        d2.metric("Irrelevant epochs", bad['n_irrelevant_epochs'])
        d3.metric("Bootstrap N", bad['n_bootstrap'])
        d4.metric("Best channel", bad['max_channel'])


def _display_batch_results(batch_results):
    """Render batch pipeline summary and per-file details."""
    st.markdown("---")
    st.subheader("Batch Results Summary")

    # Build summary table
    rows = []
    for r in batch_results:
        if 'error' in r:
            rows.append({
                'File': r['filename'],
                'Classification': 'ERROR',
                'p-value': None,
                'Max Proportion (%)': None,
                'Best Channel': None,
                'S2 Epochs': None,
                'Ind. Window (ms)': None,
                'S1 Epochs': None,
                'Error': r['error'],
            })
            continue
        bad = r['bad_results']
        ip = r.get('individual_p300_window')
        rows.append({
            'File': r['filename'],
            'Classification': bad['overall_classification'],
            'p-value': round(bad['p_value'], 4),
            'Max Proportion (%)': round(bad['max_proportion'] * 100, 1),
            'Best Channel': bad['max_channel'],
            'S2 Epochs': (
                ip['n_s2_epochs_after_ar'] if ip else '—'
            ),
            'Ind. Window (ms)': (
                f"{ip['window_start']*1000:.0f}–"
                f"{ip['window_end']*1000:.0f}" if ip else '—'
            ),
            'S1 Epochs': r['n_s1_after'],
            'Error': '',
        })

    df_summary = pd.DataFrame(rows)
    st.dataframe(df_summary, width='stretch')

    # Counts
    n_guilty = sum(
        1 for r in batch_results
        if r.get('bad_results', {}).get('overall_classification') == 'GUILTY'
    )
    n_innocent = sum(
        1 for r in batch_results
        if r.get('bad_results', {}).get('overall_classification') == 'INNOCENT'
    )
    n_error = sum(1 for r in batch_results if 'error' in r)

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Guilty", n_guilty)
    sc2.metric("Innocent", n_innocent)
    if n_error:
        sc3.metric("Errors", n_error)

    # CSV export
    csv_buf = io.StringIO()
    csv_buf.write(f"# Batch CTP-BAD Results — {pd.Timestamp.now()}\n")
    df_summary.to_csv(csv_buf, index=False)
    st.download_button(
        "Download batch summary (CSV)",
        data=csv_buf.getvalue(),
        file_name="batch_ctp_bad_results.csv",
        mime="text/csv",
        key="batch_csv_dl",
    )

    # Per-file details
    st.subheader("Per-file Details")
    for r in batch_results:
        fname = r['filename']
        if 'error' in r:
            with st.expander(f"❌ {fname} — ERROR"):
                st.error(r['error'])
            continue
        bad = r['bad_results']
        status_icon = (
            "🔴" if bad['overall_classification'] == "GUILTY" else "🟢"
        )
        with st.expander(
            f"{status_icon} {fname} — "
            f"{bad['overall_classification']} "
            f"(p={bad['p_value']:.4f})"
        ):
            _display_pipeline_results(r)


def _display_grouped_batch_results(
    batch_results,
    loocv_evaluation=False,
    fbeta_weight=1.0,
):
    """Render grouped batch results with ground-truth-based metrics.

    Each element in *batch_results* must carry a ``ground_truth`` key
    (``'guilty'`` or ``'innocent'``).

    Parameters
    ----------
    batch_results
        Per-file pipeline outputs including ``bad_results`` and labels.
    loocv_evaluation
        When True, leave-one-out thresholding on ``max_proportion``
        (same scheme as Grid Search LOOCV).
    fbeta_weight
        β for F-score when selecting cutoffs on training folds.
    """
    st.markdown("---")
    loocv_metrics = None
    if loocv_evaluation:
        st.subheader("LOOCV evaluation")
        st.caption(
            "Each fold: choose a cutoff on the other subjects' "
            "**max_proportion** scores (maximize F-β), apply it to the "
            "held-out subject. The pipeline **Guilty threshold** is not "
            "used for these metrics. Requires more than two valid files."
        )
        _loo_valid = [r for r in batch_results if 'error' not in r]
        _loo_failed = len(batch_results) - len(_loo_valid)
        if _loo_failed:
            st.caption(
                f"{_loo_failed} file(s) skipped by LOOCV (pipeline error)."
            )
        if len(_loo_valid) <= 2:
            st.warning(
                "LOOCV needs more than two successful recordings."
            )
        else:
            loocv_metrics = _score_combo(
                batch_results,
                use_loocv=True,
                fbeta_weight=fbeta_weight,
            )
            lv1, lv2, lv3, lv4, lv5 = st.columns(5)
            lv1.metric("ROC-AUC", f"{loocv_metrics['auc']:.4f}")
            lv2.metric("F-β (LOOCV preds)", f"{loocv_metrics['fbeta']:.4f}")
            lv3.metric("Accuracy", f"{loocv_metrics['accuracy']:.1%}")
            lv4.metric("Sensitivity", f"{loocv_metrics['sensitivity']:.1%}")
            lv5.metric("Specificity", f"{loocv_metrics['specificity']:.1%}")
            st.caption(
                "Reference: threshold maximizing F-β on **all** scores "
                f"jointly = {loocv_metrics['best_threshold']:.4f} "
                "(ROC marker); F-β at that threshold on all subjects = "
                f"{loocv_metrics.get('fbeta_resubstitution', 0):.4f}."
            )
            ltp = loocv_metrics['tp']
            lfn = loocv_metrics['fn']
            lfp = loocv_metrics['fp']
            ltn = loocv_metrics['tn']
            st.markdown("**LOOCV confusion matrix**")
            st.dataframe(
                pd.DataFrame(
                    [[ltp, lfn], [lfp, ltn]],
                    index=['Actual GUILTY', 'Actual INNOCENT'],
                    columns=['Pred. GUILTY', 'Pred. INNOCENT'],
                ),
                width='stretch',
            )

            _loo_detail = loocv_metrics.get('loocv_per_subject') or []
            if _loo_detail:
                n_ok = sum(1 for row in _loo_detail if row['correct'])
                st.markdown(
                    f"**Fold summary:** {n_ok}/{len(_loo_detail)} participants "
                    "correct under LOOCV; "
                    f"F-β weight for threshold search on training folds = "
                    f"**{fbeta_weight:.2f}**."
                )
                st.markdown(
                    "_For documentation:_ One row = one LOOCV fold. "
                    "**Training threshold** maximizes F-β using only the "
                    "other participants' **max_proportion** scores. "
                    "**Predicted** = GUILTY if held-out "
                    "max_proportion ≥ that threshold. "
                    "**Margin** = score − threshold (distance to boundary; "
                    "negative ⇒ below threshold)."
                )
                _df_loo = pd.DataFrame([
                    {
                        'File': row['filename'],
                        'Ground truth': row['ground_truth'],
                        'Max proportion': round(row['max_proportion'], 6),
                        'Max prop. (%)': round(row['max_proportion'] * 100, 2),
                        'Training threshold': round(
                            row['train_selected_threshold'], 6,
                        ),
                        'Margin (score − thr.)': round(
                            row['score_minus_threshold'], 6,
                        ),
                        'Predicted': row['predicted'],
                        'Correct': '\u2713' if row['correct'] else '\u2717',
                    }
                    for row in _loo_detail
                ]).sort_values('File', kind='stable')
                with st.expander(
                    "Per-participant LOOCV table (thesis / appendix)",
                    expanded=False,
                ):
                    st.dataframe(_df_loo, width='stretch')
                    st.download_button(
                        "Download LOOCV per-participant table (CSV)",
                        data=_df_loo.to_csv(index=False),
                        file_name="loocv_per_participant_details.csv",
                        mime="text/csv",
                        key="qp_loocv_participant_csv_dl",
                    )

                _wrong = [row for row in _loo_detail if not row['correct']]
                if _wrong:
                    with st.expander(
                        f"Misclassified under LOOCV ({len(_wrong)} file(s))",
                        expanded=True,
                    ):
                        st.markdown(
                            "These recordings received the wrong binary label "
                            "when their score was compared to the threshold "
                            "learned without them."
                        )
                        for w in sorted(_wrong, key=lambda x: x['filename']):
                            st.markdown(
                                f"- **`{w['filename']}`** — ground truth "
                                f"**{w['ground_truth']}**, predicted "
                                f"**{w['predicted']}** "
                                f"(max_prop={w['max_proportion']:.4f}, "
                                f"training_thr={w['train_selected_threshold']:.4f}, "
                                f"margin={w['score_minus_threshold']:+.4f})"
                            )
                else:
                    st.success(
                        "No LOOCV misclassifications "
                        f"({len(_loo_detail)} folds)."
                    )
        st.markdown("---")

    st.subheader("Grouped Batch Results")

    rows = []
    for r in batch_results:
        gt = r['ground_truth'].upper()
        if 'error' in r:
            rows.append({
                'File': r['filename'],
                'Ground Truth': gt,
                'Classification': 'ERROR',
                'p-value': None,
                'Max Prop. (%)': None,
                'Best Channel': None,
                'S2 Epochs': None,
                'Ind. Window (ms)': None,
                'S1 Epochs': None,
                'Correct': None,
                'Error': r['error'],
            })
            continue
        bad = r['bad_results']
        ip = r.get('individual_p300_window')
        classification = bad['overall_classification']
        correct = (
            (gt == 'GUILTY' and classification == 'GUILTY')
            or (gt == 'INNOCENT' and classification == 'INNOCENT')
        )
        rows.append({
            'File': r['filename'],
            'Ground Truth': gt,
            'Classification': classification,
            'p-value': round(bad['p_value'], 4),
            'Max Prop. (%)': round(bad['max_proportion'] * 100, 1),
            'Best Channel': bad['max_channel'],
            'S2 Epochs': (
                ip['n_s2_epochs_after_ar'] if ip else '—'
            ),
            'Ind. Window (ms)': (
                f"{ip['window_start']*1000:.0f}–"
                f"{ip['window_end']*1000:.0f}" if ip else '—'
            ),
            'S1 Epochs': r['n_s1_after'],
            'Correct': '\u2713' if correct else '\u2717',
            'Error': '',
        })

    df_summary = pd.DataFrame(rows)
    if loocv_evaluation:
        st.caption(
            "Per-file Classification uses the pipeline Guilty threshold "
            "(Step 5); LOOCV above uses score-based cutoffs instead."
        )
    st.dataframe(df_summary, width='stretch')

    # ------------------------------------------------------------------
    # Confusion matrix & classification metrics
    # ------------------------------------------------------------------
    valid = [r for r in batch_results if 'error' not in r]
    n_error = len(batch_results) - len(valid)

    if valid:
        tp = sum(
            1 for r in valid
            if r['ground_truth'] == 'guilty'
            and r['bad_results']['overall_classification'] == 'GUILTY'
        )
        fn = sum(
            1 for r in valid
            if r['ground_truth'] == 'guilty'
            and r['bad_results']['overall_classification'] == 'INNOCENT'
        )
        fp = sum(
            1 for r in valid
            if r['ground_truth'] == 'innocent'
            and r['bad_results']['overall_classification'] == 'GUILTY'
        )
        tn = sum(
            1 for r in valid
            if r['ground_truth'] == 'innocent'
            and r['bad_results']['overall_classification'] == 'INNOCENT'
        )

        n_total = tp + tn + fp + fn
        accuracy = (tp + tn) / n_total if n_total else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (
            2 * precision * sensitivity / (precision + sensitivity)
            if (precision + sensitivity) else 0.0
        )

        st.subheader(
            "Classification metrics (pipeline guilty threshold)",
        )
        cm_col, metric_col = st.columns(2)

        with cm_col:
            st.markdown("**Confusion Matrix**")
            cm_df = pd.DataFrame(
                [[tp, fn], [fp, tn]],
                index=['Actual GUILTY', 'Actual INNOCENT'],
                columns=['Pred. GUILTY', 'Pred. INNOCENT'],
            )
            st.dataframe(cm_df, width='stretch')

        with metric_col:
            st.markdown("**Performance Metrics**")
            m1, m2 = st.columns(2)
            m1.metric("Accuracy", f"{accuracy:.1%}")
            m2.metric("F1 Score", f"{f1:.1%}")
            m3, m4 = st.columns(2)
            m3.metric("Sensitivity (TPR)", f"{sensitivity:.1%}")
            m4.metric("Specificity (TNR)", f"{specificity:.1%}")
            m5, m6 = st.columns(2)
            m5.metric("Precision (PPV)", f"{precision:.1%}")
            m6.metric("N (valid)", n_total)

        st.subheader("Group Breakdown")
        g1, g2, g3 = st.columns(3)
        g1.metric("Guilty files", tp + fn)
        g2.metric("Innocent files", tn + fp)
        if n_error:
            g3.metric("Errors", n_error)
        else:
            g3.metric("Total valid", n_total)

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------
    csv_buf = io.StringIO()
    csv_buf.write(
        f"# Grouped Batch CTP-BAD Results — {pd.Timestamp.now()}\n"
    )
    df_summary.to_csv(csv_buf, index=False)

    if loocv_metrics is not None:
        csv_buf.write("\n# LOOCV evaluation (max_proportion scores)\n")
        pd.DataFrame([{
            'ROC_AUC': f"{loocv_metrics['auc']:.6f}",
            'Fbeta_LOOCV_predictions': f"{loocv_metrics['fbeta']:.6f}",
            'Fbeta_resubstitution_at_full_sample_threshold': (
                f"{loocv_metrics.get('fbeta_resubstitution', 0):.6f}"
            ),
            'Full_sample_Fbeta_threshold': (
                f"{loocv_metrics['best_threshold']:.6f}"
            ),
            'Accuracy': f"{loocv_metrics['accuracy']:.6f}",
            'Sensitivity': f"{loocv_metrics['sensitivity']:.6f}",
            'Specificity': f"{loocv_metrics['specificity']:.6f}",
            'TP': loocv_metrics['tp'],
            'FN': loocv_metrics['fn'],
            'FP': loocv_metrics['fp'],
            'TN': loocv_metrics['tn'],
            'Fbeta_weight': fbeta_weight,
        }]).to_csv(csv_buf, index=False)
        _loo_rows = loocv_metrics.get('loocv_per_subject')
        if _loo_rows:
            csv_buf.write(
                "\n# LOOCV per-participant fold details "
                "(same order as folds in UI table; sorted by file "
                "name in downloadable snippet)\n",
            )
            pd.DataFrame(_loo_rows).sort_values(
                'filename', kind='stable',
            ).to_csv(csv_buf, index=False)

    if valid:
        csv_buf.write(
            "\n# Classification metrics (pipeline guilty threshold)\n",
        )
        metrics_df = pd.DataFrame([{
            'Accuracy': f"{accuracy:.4f}",
            'Sensitivity_TPR': f"{sensitivity:.4f}",
            'Specificity_TNR': f"{specificity:.4f}",
            'Precision_PPV': f"{precision:.4f}",
            'F1_Score': f"{f1:.4f}",
            'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn,
            'N_valid': n_total,
        }])
        metrics_df.to_csv(csv_buf, index=False)

    st.download_button(
        "Download grouped batch summary (CSV)",
        data=csv_buf.getvalue(),
        file_name="grouped_batch_ctp_bad_results.csv",
        mime="text/csv",
        key="grouped_batch_csv_dl",
    )

    # ------------------------------------------------------------------
    # Per-file details
    # ------------------------------------------------------------------
    st.subheader("Per-file Details")
    for r in batch_results:
        fname = r['filename']
        gt = r['ground_truth'].upper()
        if 'error' in r:
            with st.expander(f"\u274c {fname} [{gt}] — ERROR"):
                st.error(r['error'])
            continue
        bad = r['bad_results']
        classification = bad['overall_classification']
        correct = (
            (gt == 'GUILTY' and classification == 'GUILTY')
            or (gt == 'INNOCENT' and classification == 'INNOCENT')
        )
        icon = "\u2705" if correct else "\u26a0\ufe0f"
        with st.expander(
            f"{icon} {fname} [{gt}] \u2192 "
            f"{classification} (p={bad['p_value']:.4f})"
        ):
            _display_pipeline_results(r)


# ============================================================================
# Grid Search / Optuna Helpers
# ============================================================================

_MIN_PROBE_EPOCHS = 25
_ROC_THRESHOLDS = np.linspace(0.0, 1.0, 201)

_FILTER_PRESET_TO_WIDGET = {
    'skip': 'Skip',
    'aggressive': 'Aggressive (data rescue)',
    'custom': 'Custom',
}
_REJ_TO_WIDGET = {
    'none': 'None',
    'static': 'Static threshold',
    'iqr': 'Adaptive: IQR',
    'zscore': 'Adaptive: Z-score',
    'autoreject': 'Autoreject (ML-based)',
}


def _cfg_to_json(cfg):
    """Serialize a pipeline config dict to a JSON string.

    Skips keys whose values are not JSON-serializable.
    """
    safe = {}
    for k, v in cfg.items():
        if callable(v):
            continue
        try:
            json.dumps(v)
            safe[k] = v
        except (TypeError, ValueError):
            continue
    return json.dumps(safe, indent=2, ensure_ascii=False)


def _apply_config_to_widgets(cfg):
    """Map a pipeline config dict to Quick Pipeline widget session-state keys.

    Must be called **before** the widgets render so that Streamlit picks
    up the pre-set values.  Caller should ``st.rerun()`` afterwards.
    """
    ss = st.session_state

    fp = cfg.get('filter_preset', 'aggressive')
    ss['pipe_filter_preset'] = _FILTER_PRESET_TO_WIDGET.get(fp, fp)

    hp = cfg.get('hp_cutoff', 0.7)
    lp = cfg.get('lp_cutoff', 30.0)
    order = cfg.get('iir_order', 4)
    notch = cfg.get('notch_freqs') or []
    for key in ('pipe_hp_agg', 'pipe_hp'):
        ss[key] = float(hp)
    for key in ('pipe_lp_agg', 'pipe_lp'):
        ss[key] = float(lp)
    for key in ('pipe_iir_order_agg', 'pipe_iir_order'):
        ss[key] = int(order)
    for key in ('pipe_notch_agg', 'pipe_notch'):
        ss[key] = notch
    ss['pipe_fmethod'] = cfg.get('filter_method', 'iir')

    # S2 epochs
    ss['pipe_s2_tmin'] = float(cfg.get('s2_tmin', -0.2))
    ss['pipe_s2_tmax'] = float(cfg.get('s2_tmax', 0.8))
    ss['pipe_s2_bl'] = bool(cfg.get('s2_baseline', True))
    ss['pipe_s2_detrend'] = cfg.get('s2_detrend', 'DC offset (0)')
    s2r = cfg.get('s2_rejection', 'autoreject')
    ss['pipe_s2_rej'] = _REJ_TO_WIDGET.get(s2r, s2r)
    if cfg.get('s2_threshold_uv') is not None:
        ss['pipe_s2_thresh'] = float(cfg['s2_threshold_uv'])

    # Individual P300 window
    ss['pipe_use_iw'] = bool(cfg.get('use_individual_window', True))
    ss['pipe_search_min'] = float(cfg.get('peak_search_tmin', 0.25))
    ss['pipe_search_max'] = float(cfg.get('peak_search_tmax', 0.60))
    ss['pipe_margin'] = float(cfg.get('window_margin', 0.15))
    erp_lp = cfg.get('s2_erp_lowpass_hz')
    ss['pipe_s2_erp_lp_en'] = erp_lp is not None
    if erp_lp is not None:
        ss['pipe_s2_erp_lp_hz'] = float(erp_lp)
    if cfg.get('peak_channels'):
        ss['pipe_peak_chs'] = list(cfg['peak_channels'])
    ss['pipe_man_tmin'] = float(cfg.get('manual_tmin', 0.3))
    ss['pipe_man_tmax'] = float(cfg.get('manual_tmax', 0.6))

    # S1 epochs
    ss['pipe_s1_tmin'] = float(cfg.get('s1_tmin', -0.2))
    ss['pipe_s1_tmax'] = float(cfg.get('s1_tmax', 0.8))
    ss['pipe_s1_bl'] = bool(cfg.get('s1_baseline', True))
    ss['pipe_s1_detrend'] = cfg.get('s1_detrend', 'DC offset (0)')
    s1r = cfg.get('s1_rejection', 'autoreject')
    ss['pipe_s1_rej'] = _REJ_TO_WIDGET.get(s1r, s1r)
    if cfg.get('s1_threshold_uv') is not None:
        ss['pipe_s1_thresh'] = float(cfg['s1_threshold_uv'])
    ss['pipe_s1_k'] = float(cfg.get('s1_adaptive_k', 3.0))

    # S2 trial rejection
    ss['pipe_s2_trial_rej'] = bool(cfg.get('s2_trial_rejection', True))
    s2_max_rt = cfg.get('s2_max_rt')
    ss['pipe_s2_max_rt_en'] = s2_max_rt is not None
    if s2_max_rt is not None:
        ss['pipe_s2_max_rt'] = float(s2_max_rt)

    # BAD channels
    peak_chs = cfg.get('peak_channels', ['Pz'])
    bad_chs = cfg.get('bad_channels')
    if bad_chs and bad_chs != peak_chs:
        ss['pipe_bad_chs_override'] = True
        ss['pipe_bad_chs'] = list(bad_chs)
    else:
        ss['pipe_bad_chs_override'] = False

    # Amplitude / smoothing / bootstrap
    ss['pipe_amp_method'] = cfg.get(
        'amplitude_method', 'Peak-to-Peak (Peak-Valley)',
    )
    if cfg.get('p2p_tmax_negative') is not None:
        ss['pipe_p2p_neg'] = float(cfg['p2p_tmax_negative'])
    ss['pipe_smooth_method'] = cfg.get(
        'smoothing_method', 'Low-pass (Butterworth)',
    )
    if cfg.get('smoothing_lp_hz') is not None:
        ss['pipe_smooth_lp'] = float(cfg['smoothing_lp_hz'])
    if cfg.get('smoothing_ma_ms') is not None:
        ss['pipe_smooth_ma'] = float(cfg['smoothing_ma_ms'])
    ss['pipe_n_boot'] = int(cfg.get('n_bootstrap', 10000))
    ss['pipe_threshold'] = float(cfg.get('guilty_threshold', 0.75))
    ss['pipe_ar_jobs'] = int(cfg.get('ar_n_jobs', 1))


def _grid_space_size(ranges):
    """Return the number of combinations without materializing the grid."""
    if not ranges:
        return 0
    # Adaptive k applies only to IQR / Z-score rejection; count accordingly.
    if 's1_adaptive_k' in ranges and 's1_rejection' in ranges:
        return len(_build_grid_space(ranges))
    n = 1
    for vals in ranges.values():
        n *= len(vals)
    return n


def _build_grid_space(ranges):
    """Cartesian product of *ranges* → ``list[dict]``.

    When both ``s1_rejection`` and ``s1_adaptive_k`` are present, values of
    *s1_adaptive_k* are crossed in only for ``iqr`` / ``zscore``. For
    ``autoreject`` (or other methods), *k* is omitted so ``base_cfg`` applies.
    """
    if not ranges:
        return []
    k_key = 's1_adaptive_k'
    rej_key = 's1_rejection'
    if k_key in ranges and rej_key in ranges:
        k_vals = ranges[k_key]
        sub_ranges = {k: v for k, v in ranges.items() if k != k_key}
        keys = list(sub_ranges.keys())
        values = [sub_ranges[k] for k in keys]
        out = []
        for combo in _iter_product(*values):
            row = dict(zip(keys, combo))
            rej = row.get(rej_key)
            if rej in ('iqr', 'zscore'):
                for kv in k_vals:
                    out.append({**row, k_key: kv})
            else:
                out.append(dict(row))
        return out
    keys = list(ranges.keys())
    values = [ranges[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in _iter_product(*values)]


def _parse_custom_floats(text):
    """Parse comma-separated floats from a text input string."""
    out = []
    for tok in text.replace(';', ',').split(','):
        tok = tok.strip()
        if tok:
            try:
                out.append(float(tok))
            except ValueError:
                pass
    return sorted(set(out))


def _parse_custom_ints(text):
    """Parse comma-separated ints from a text input string."""
    out = []
    for tok in text.replace(';', ',').split(','):
        tok = tok.strip()
        if tok:
            try:
                out.append(int(tok))
            except ValueError:
                pass
    return sorted(set(out))


def _compute_roc_auc(labels, scores):
    """Compute ROC-AUC from ground-truth *labels* (1=guilty) and *scores*.

    Sweeps ``_ROC_THRESHOLDS``, deduplicates by keeping the maximum
    TPR at each unique FPR level, then applies the trapezoidal rule.

    :returns: ``(auc, tpr_list, fpr_list, thresholds)``
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5, [], [], _ROC_THRESHOLDS

    tpr_raw, fpr_raw = [], []
    for thr in _ROC_THRESHOLDS:
        pred = (scores >= thr).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        tpr_raw.append(tp / n_pos)
        fpr_raw.append(fp / n_neg)

    fpr_arr = np.array(fpr_raw)
    tpr_arr = np.array(tpr_raw)

    unique_fprs = np.unique(fpr_arr)
    unique_tprs = np.array([tpr_arr[fpr_arr == f].max() for f in unique_fprs])

    order = np.argsort(unique_fprs)
    unique_fprs = unique_fprs[order]
    unique_tprs = unique_tprs[order]

    auc = float(np.trapz(unique_tprs, unique_fprs))
    return auc, unique_tprs.tolist(), unique_fprs.tolist(), _ROC_THRESHOLDS


def _fbeta_score(labels, preds, beta=1.0):
    """Compute F-beta score from arrays of ground-truth *labels* and *preds*.

    :returns: ``float`` in [0, 1].
    """
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    b2 = beta * beta
    denom = (1 + b2) * tp + b2 * fn + fp
    if denom == 0:
        return 0.0
    return float((1 + b2) * tp / denom)


def _score_combo(batch_results, use_loocv=False, fbeta_weight=1.0):
    """Score a single grid-search combo.

    Computes ROC-AUC by sweeping thresholds over ``max_proportion``.
    Threshold selection uses the **F-beta** metric (with *fbeta_weight*)
    instead of raw accuracy.

    When *use_loocv* is ``True``, accuracy / sensitivity / specificity /
    ``fbeta`` use leave-one-out predictions (threshold chosen on N−1
    subjects via F-beta, tested on the held-out one).
    ``fbeta_resubstitution`` is F-beta at ``best_threshold`` on all subjects.

    :returns: dict with metric keys (includes ``fbeta_resubstitution`` and,
        when *use_loocv* with >2 subjects, ``loocv_per_subject`` fold table).
    """
    valid = [r for r in batch_results if 'error' not in r]
    if not valid:
        return {
            'auc': 0.0, 'best_threshold': 0.5,
            'accuracy': 0.0, 'sensitivity': 0.0, 'specificity': 0.0,
            'fbeta': 0.0,
            'tp': 0, 'fn': 0, 'fp': 0, 'tn': 0,
            'roc_tprs': [], 'roc_fprs': [],
            'best_fpr': 0.0, 'best_tpr': 0.0,
            'fbeta_resubstitution': 0.0,
            'loocv_per_subject': None,
        }

    labels = np.array([
        1 if r['ground_truth'] == 'guilty' else 0 for r in valid
    ])
    scores = np.array([
        r['bad_results']['max_proportion'] for r in valid
    ])

    auc, tprs, fprs, thresholds = _compute_roc_auc(labels, scores)

    loo_preds = None
    loocv_per_subject = None
    if use_loocv and len(valid) > 2:
        loo_preds = np.zeros(len(valid), dtype=int)
        loocv_per_subject = []
        for i in range(len(valid)):
            train_mask = np.ones(len(valid), dtype=bool)
            train_mask[i] = False
            train_labels = labels[train_mask]
            train_scores = scores[train_mask]
            best_thr_i, best_fb_i = 0.5, -1.0
            for thr in _ROC_THRESHOLDS:
                pred = (train_scores >= thr).astype(int)
                fb = _fbeta_score(train_labels, pred, beta=fbeta_weight)
                if fb > best_fb_i:
                    best_fb_i, best_thr_i = fb, thr
            loo_preds[i] = int(scores[i] >= best_thr_i)
            vr = valid[i]
            pred_guilty = bool(loo_preds[i] == 1)
            gt_guilty = vr['ground_truth'] == 'guilty'
            loocv_per_subject.append({
                'filename': vr['filename'],
                'ground_truth': str(vr['ground_truth']).upper(),
                'max_proportion': float(scores[i]),
                'train_selected_threshold': float(best_thr_i),
                'predicted': 'GUILTY' if pred_guilty else 'INNOCENT',
                'score_minus_threshold': float(scores[i] - best_thr_i),
                'correct': pred_guilty == gt_guilty,
            })
        tp = int(((loo_preds == 1) & (labels == 1)).sum())
        fn = int(((loo_preds == 0) & (labels == 1)).sum())
        fp = int(((loo_preds == 1) & (labels == 0)).sum())
        tn = int(((loo_preds == 0) & (labels == 0)).sum())
        best_thr_final = 0.5
        best_fb_final = -1.0
        for thr in _ROC_THRESHOLDS:
            pred = (scores >= thr).astype(int)
            fb = _fbeta_score(labels, pred, beta=fbeta_weight)
            if fb > best_fb_final:
                best_fb_final, best_thr_final = fb, thr
    else:
        best_thr_final, best_fb_final = 0.5, -1.0
        for thr in thresholds:
            pred = (scores >= thr).astype(int)
            fb = _fbeta_score(labels, pred, beta=fbeta_weight)
            if fb > best_fb_final:
                best_fb_final, best_thr_final = fb, thr
        pred_best = (scores >= best_thr_final).astype(int)
        tp = int(((pred_best == 1) & (labels == 1)).sum())
        fn = int(((pred_best == 0) & (labels == 1)).sum())
        fp = int(((pred_best == 1) & (labels == 0)).sum())
        tn = int(((pred_best == 0) & (labels == 0)).sum())

    n = tp + fn + fp + tn

    pred_at_best = (scores >= best_thr_final).astype(int)
    _n_neg = int((labels == 0).sum())
    _n_pos = int((labels == 1).sum())
    best_fpr = float(
        ((pred_at_best == 1) & (labels == 0)).sum() / max(_n_neg, 1)
    )
    best_tpr = float(
        ((pred_at_best == 1) & (labels == 1)).sum() / max(_n_pos, 1)
    )
    fbeta_resub = _fbeta_score(labels, pred_at_best, beta=fbeta_weight)
    if loo_preds is not None:
        fbeta_val = _fbeta_score(labels, loo_preds, beta=fbeta_weight)
    else:
        fbeta_val = fbeta_resub

    return {
        'auc': auc,
        'best_threshold': float(best_thr_final),
        'accuracy': (tp + tn) / n if n else 0.0,
        'sensitivity': tp / (tp + fn) if (tp + fn) else 0.0,
        'specificity': tn / (tn + fp) if (tn + fp) else 0.0,
        'fbeta': fbeta_val,
        'fbeta_resubstitution': fbeta_resub,
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'roc_tprs': tprs, 'roc_fprs': fprs,
        'best_fpr': best_fpr, 'best_tpr': best_tpr,
        'loocv_per_subject': loocv_per_subject,
    }


_PEAK_AMPLITUDE_METHODS = frozenset({
    'Peak-to-peak (Rosenfeld)',
    'Peak-to-Peak (Peak-Valley)',
    'Baseline-to-peak',
})


def _enforce_smoothing_for_peak_methods(cfg):
    """Force smoothing when a peak-based amplitude method is selected.

    Peak-based extraction is highly sensitive to single-sample noise
    spikes.  Running without smoothing would inflate variance and make
    the bootstrap test unreliable.
    """
    amp = cfg.get('amplitude_method', '')
    smooth = cfg.get('smoothing_method', 'Low-pass (Butterworth)')
    if amp in _PEAK_AMPLITUDE_METHODS and smooth == 'None':
        cfg = {**cfg, 'smoothing_method': 'Low-pass (Butterworth)'}
    return cfg


_HEAVY_PIPELINE_KEYS = frozenset({
    'raw', 'raw_filtered', 'epochs', 'probe_erp', 'irrelevant_erp',
})


def _strip_heavy_objects(result):
    """Remove large MNE objects from a pipeline result dict.

    Keeps only lightweight data needed for scoring and display
    (``bad_results``, ``individual_p300_window``, metrics, log, etc.).
    """
    return {k: v for k, v in result.items() if k not in _HEAVY_PIPELINE_KEYS}


def _run_single_file(raw, events, event_id, fname, ground_truth, cfg):
    """Run pipeline for one file. Thread-safe wrapper for parallel execution."""
    try:
        r = run_pipeline(raw, events, event_id, cfg)
        r['filename'] = fname
        r['ground_truth'] = ground_truth
        return r
    except Exception as exc:
        return {
            'filename': fname,
            'ground_truth': ground_truth,
            'error': str(exc),
        }


def _run_combo(innocent_data, guilty_data, cfg, max_workers=1):
    """Run pipeline for all files under a single config.

    :param max_workers:
        Number of parallel threads. MNE releases the GIL for most
        NumPy/SciPy work, so threads give real speedup here.
    :returns: ``list[dict]`` — per-file pipeline results.
    """
    all_jobs = [
        (raw, ev, eid, fn, 'innocent', cfg)
        for raw, ev, eid, fn in innocent_data
    ] + [
        (raw, ev, eid, fn, 'guilty', cfg)
        for raw, ev, eid, fn in guilty_data
    ]

    if max_workers <= 1 or len(all_jobs) <= 1:
        return [_run_single_file(*args) for args in all_jobs]

    batch = [None] * len(all_jobs)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(_run_single_file, *args): i
            for i, args in enumerate(all_jobs)
        }
        for future in as_completed(future_to_idx):
            batch[future_to_idx[future]] = future.result()
    return batch


def _run_grid_search(
    innocent_data,
    guilty_data,
    base_cfg,
    grid_space,
    progress_bar,
    status_text,
    use_loocv=False,
    fbeta_weight=1.0,
    max_workers=1,
):
    """Execute a full grid search with ROC-AUC scoring and ETA.

    Partial results are saved to ``st.session_state`` after every
    iteration so that they survive page refreshes.
    """
    results = st.session_state.get('_gs_partial', [])
    start_idx = len(results)
    n_combos = len(grid_space)
    cum_time = sum(r['elapsed_s'] for r in results)

    for idx in range(start_idx, n_combos):
        if st.session_state.get('_gs_stop'):
            status_text.text(
                f"Stopped after {idx}/{n_combos} combos."
            )
            break

        overrides = grid_space[idx]
        t0 = _time.time()
        cfg = _enforce_smoothing_for_peak_methods(
            {**base_cfg, **overrides},
        )

        eta_str = ""
        if idx > 0 and cum_time > 0:
            avg = cum_time / idx
            remaining = avg * (n_combos - idx)
            eta_str = f" | ETA {remaining:.0f}s"

        status_text.text(
            f"Combo {idx + 1}/{n_combos}{eta_str}: "
            + ", ".join(f"{k}={v}" for k, v in overrides.items())
        )
        progress_bar.progress(
            int(idx / n_combos * 100),
            text=f"Grid search {idx + 1}/{n_combos}{eta_str}",
        )

        batch = _run_combo(
            innocent_data, guilty_data, cfg,
            max_workers=max_workers,
        )
        metrics = _score_combo(
            batch, use_loocv=use_loocv, fbeta_weight=fbeta_weight,
        )
        elapsed = _time.time() - t0
        cum_time += elapsed

        lightweight_batch = [_strip_heavy_objects(r) for r in batch]
        del batch

        results.append({
            'overrides': overrides,
            'cfg': cfg,
            **metrics,
            'n_errors': sum(
                1 for r in lightweight_batch if 'error' in r
            ),
            'elapsed_s': elapsed,
            'batch_results': lightweight_batch,
        })
        st.session_state['_gs_partial'] = results

    done = len(results) >= n_combos
    label = "Grid search complete!" if done else "Grid search stopped."
    progress_bar.progress(100 if done else int(len(results) / n_combos * 100), text=label)
    st.session_state.pop('_gs_partial', None)
    return results


def _run_optuna_search(
    innocent_data,
    guilty_data,
    base_cfg,
    param_defs,
    n_trials,
    progress_bar,
    status_text,
    use_loocv=False,
    fbeta_weight=1.0,
    study=None,
    max_workers=1,
    adaptive_k_float_spec=None,
):
    """Run Optuna TPE optimisation.

    Mixed search space: categorical lists from the Grid Search UI,
    stepped integers/floats for selected LP/P300/BAD cutoffs, and optional
    conditional ``s1_adaptive_k`` (only when rejection is IQR/Z-score).

    :param param_defs:
        ``list[(key, kind, lo, hi, step_or_choices)]``.
        *kind* is ``'cat'`` (*step_or_choices* = choice list), ``'int'``, or
        ``'float'`` (*step_or_choices* = step).
    :param adaptive_k_float_spec:
        ``(low, high, step)`` for ``trial.suggest_float`` on
        ``s1_adaptive_k`` when ``s1_rejection`` is ``iqr`` or ``zscore``;
        ``None`` to omit.
    :param study:
        Optional existing ``optuna.Study`` to resume.
    :returns: ``list[dict]`` same shape as grid search results.
    """
    results = st.session_state.get('_gs_partial', [])
    start_trial = len(results)
    cum_time = sum(r['elapsed_s'] for r in results)
    stopped = False
    trial_counter = start_trial

    def objective(trial):
        nonlocal cum_time, stopped, trial_counter
        if st.session_state.get('_gs_stop'):
            stopped = True
            raise optuna.TrialPruned()

        t0 = _time.time()
        overrides = {}
        for key, kind, lo, hi, step_or_choices in param_defs:
            if kind == 'int':
                overrides[key] = trial.suggest_int(
                    key, lo, hi, step=step_or_choices,
                )
            elif kind == 'float':
                overrides[key] = trial.suggest_float(
                    key, lo, hi, step=step_or_choices,
                )
            else:
                overrides[key] = trial.suggest_categorical(
                    key, step_or_choices,
                )

        rej = overrides.get(
            's1_rejection', base_cfg.get('s1_rejection'),
        )
        if adaptive_k_float_spec is not None and rej in ('iqr', 'zscore'):
            k_lo, k_hi, k_step = adaptive_k_float_spec
            overrides['s1_adaptive_k'] = trial.suggest_float(
                's1_adaptive_k', k_lo, k_hi, step=k_step,
            )

        cfg = _enforce_smoothing_for_peak_methods(
            {**base_cfg, **overrides},
        )
        idx = trial_counter

        eta_str = ""
        if idx > 0 and cum_time > 0:
            avg = cum_time / idx
            eta_remaining = avg * (n_trials - idx)
            eta_str = f" | ETA {eta_remaining:.0f}s"

        status_text.text(
            f"Trial {idx + 1}/{n_trials}{eta_str}: "
            + ", ".join(f"{k}={v}" for k, v in overrides.items())
        )
        progress_bar.progress(
            min(int(idx / n_trials * 100), 100),
            text=f"Optuna {idx + 1}/{n_trials}{eta_str}",
        )

        batch = _run_combo(
            innocent_data, guilty_data, cfg,
            max_workers=max_workers,
        )
        metrics = _score_combo(
            batch, use_loocv=use_loocv, fbeta_weight=fbeta_weight,
        )
        elapsed = _time.time() - t0
        cum_time += elapsed

        lightweight_batch = [_strip_heavy_objects(r) for r in batch]
        del batch

        results.append({
            'overrides': {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in overrides.items()
            },
            'cfg': cfg,
            **metrics,
            'n_errors': sum(
                1 for r in lightweight_batch if 'error' in r
            ),
            'elapsed_s': elapsed,
            'batch_results': lightweight_batch,
        })
        st.session_state['_gs_partial'] = results
        trial_counter += 1
        return metrics['auc']

    if study is None:
        study = optuna.create_study(direction='maximize')

    def _stop_callback(study_obj, trial):
        if st.session_state.get('_gs_stop'):
            study_obj.stop()

    remaining = max(0, n_trials - start_trial)
    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            show_progress_bar=False,
            callbacks=[_stop_callback],
        )

    st.session_state['_optuna_study'] = study

    done = len(results) >= n_trials
    label = "Optuna search complete!" if done else "Optuna search stopped."
    progress_bar.progress(100 if done else int(len(results) / n_trials * 100), text=label)
    st.session_state.pop('_gs_partial', None)
    return results


def _display_grid_search_results(gs_results, use_loocv=False):
    """Render search results table, ROC curve, best config, and details."""
    st.markdown("---")
    n_completed = len(gs_results)
    validation_tag = " (LOOCV)" if use_loocv else " (resubstitution)"
    st.subheader(f"Search Results \u2014 {n_completed} combos{validation_tag}")

    rows = []
    for i, r in enumerate(gs_results):
        row = {'#': i + 1}
        row.update(r['overrides'])
        row['AUC'] = f"{r['auc']:.4f}"
        row['F-beta'] = f"{r.get('fbeta', 0.0):.4f}"
        row['Best Thr.'] = f"{r['best_threshold']:.2f}"
        row['Accuracy'] = f"{r['accuracy']:.1%}"
        row['Sensitivity'] = f"{r['sensitivity']:.1%}"
        row['Specificity'] = f"{r['specificity']:.1%}"
        row['TP'] = r['tp']
        row['FN'] = r['fn']
        row['FP'] = r['fp']
        row['TN'] = r['tn']
        row['Errors'] = r['n_errors']
        row['Time (s)'] = f"{r['elapsed_s']:.1f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.highlight_max(subset=['AUC'], color='#c6efce'),
        width='stretch',
    )

    best = max(gs_results, key=lambda r: r['auc'])
    best_idx = gs_results.index(best) + 1

    st.subheader("Best configuration (by ROC-AUC)")
    b1, b2, b3, b4, b5, b6 = st.columns(6)
    b1.metric("AUC", f"{best['auc']:.4f}")
    b2.metric("F-beta", f"{best.get('fbeta', 0.0):.4f}")
    b3.metric("Best Thr.", f"{best['best_threshold']:.2f}")
    b4.metric("Accuracy", f"{best['accuracy']:.1%}")
    b5.metric("Sensitivity", f"{best['sensitivity']:.1%}")
    b6.metric("Specificity", f"{best['specificity']:.1%}")

    if best.get('roc_fprs') and best.get('roc_tprs'):
        with st.expander("ROC Curve (best config)", expanded=True):
            fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
            ax_roc.plot(
                best['roc_fprs'], best['roc_tprs'],
                color='#1f77b4', linewidth=2,
                label=f"AUC = {best['auc']:.4f}",
            )
            ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)

            b_fpr = best.get('best_fpr')
            b_tpr = best.get('best_tpr')
            b_thr = best.get('best_threshold')
            if b_fpr is not None and b_tpr is not None:
                ax_roc.plot(
                    b_fpr, b_tpr, 'ro', markersize=8, zorder=5,
                    label=f"Thr = {b_thr:.2f}",
                )
                ax_roc.annotate(
                    f"thr={b_thr:.2f}\nTPR={b_tpr:.2f}, FPR={b_fpr:.2f}",
                    xy=(b_fpr, b_tpr),
                    xytext=(b_fpr + 0.12, b_tpr - 0.15),
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
                    bbox=dict(
                        boxstyle='round,pad=0.3', fc='#fff3cd',
                        ec='red', alpha=0.9,
                    ),
                )

            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('ROC Curve')
            ax_roc.legend(loc='lower right')
            ax_roc.grid(True, alpha=0.3)
            ax_roc.set_xlim(-0.02, 1.02)
            ax_roc.set_ylim(-0.02, 1.02)
            plt.tight_layout()
            st.pyplot(fig_roc)
            plt.close()

    st.markdown("**Optimal parameter overrides:**")
    st.json(best['overrides'])

    with st.expander("Full optimal config dict"):
        st.json({
            k: v for k, v in best['cfg'].items()
            if not callable(v)
        })

    with st.expander("Per-file results for best config"):
        _display_grouped_batch_results(best['batch_results'])

    csv_buf = io.StringIO()
    csv_buf.write(
        f"# Search Results \u2014 {pd.Timestamp.now()}\n"
    )
    df.to_csv(csv_buf, index=False)
    csv_buf.write(f"\n# Best combo: #{best_idx}\n")
    csv_buf.write(f"# AUC: {best['auc']:.4f}\n")
    csv_buf.write(
        f"# Best threshold: {best['best_threshold']:.2f}\n"
    )
    for k, v in best['overrides'].items():
        csv_buf.write(f"# {k} = {v}\n")

    dl_c1, dl_c2, dl_c3 = st.columns(3)
    with dl_c1:
        st.download_button(
            "Download search results (CSV)",
            data=csv_buf.getvalue(),
            file_name="grid_search_results.csv",
            mime="text/csv",
            key="gs_csv_dl",
        )
    with dl_c2:
        _best_cfg_json = _cfg_to_json(best['cfg'])
        st.download_button(
            "Download best config (JSON)",
            data=_best_cfg_json,
            file_name="best_pipeline_config.json",
            mime="application/json",
            key="gs_json_dl",
        )
    with dl_c3:
        if st.button(
            "Apply best config to Quick Pipeline",
            key="gs_apply_to_qp",
        ):
            _apply_config_to_widgets(best['cfg'])
            st.success(
                "Config applied! Switch to **Quick Pipeline** to use it."
            )


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application."""
    
    # Header
    st.title("🧠 EEG FIF Analyzer")
    st.markdown("Interactive analysis of EEG data in FIF format using MNE-Python")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📂 Load Data", "📊 Signal Quality", "🔧 Preprocessing",
         "🔍 Block Viewer", "📈 Epoching", "🎯 ERP Analysis",
         "⚡ Quick Pipeline", "🔎 Grid Search", "📉 Export Results"]
    )
    
    # ========================================================================
    # Page 1: Load Data
    # ========================================================================
    
    if page == "📂 Load Data":
        st.header("📂 Load FIF File")
        
        # File upload
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload FIF file",
                type=['fif'],
                help="Select a .fif file containing EEG data"
            )
        
        with col2:
            st.markdown("### Quick Info")
            st.markdown("""
            **Supported format:**
            - `.fif` (MNE-Python)
            
            **Expected content:**
            - 3 channels (Fz, Cz, Pz)
            - 250 Hz sampling rate
            - Event annotations
            """)
        
        if uploaded_file is not None:
            try:
                with st.spinner("Loading FIF file..."):
                    # Save to temporary location
                    temp_path = Path("temp_uploaded.fif")
                    temp_path.write_bytes(uploaded_file.read())
                    
                    # Load file
                    raw, events, event_id = load_fif_file(str(temp_path))
                    
                    # Store in session state
                    st.session_state.raw = raw
                    st.session_state.events = events
                    st.session_state.event_id = event_id
                    st.session_state.raw_filtered = None  # Reset filtered data
                    
                    # Clean up temp file
                    temp_path.unlink()
                
                st.success("✅ File loaded successfully!")
                
                # Display file information
                st.subheader("File Information")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Channels", len(raw.ch_names))
                with col2:
                    st.metric("Sampling Rate", f"{raw.info['sfreq']:.0f} Hz")
                with col3:
                    st.metric("Duration", f"{raw.times[-1]:.1f} s")
                with col4:
                    st.metric("Samples", len(raw.times))
                
                # Channel names
                st.markdown("**Channels:**")
                st.code(", ".join(raw.ch_names))
                
                # Annotations
                if events is not None:
                    st.subheader("Event Markers")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Events", len(raw.annotations))
                    
                    with col2:
                        st.metric("Event Types", len(event_id))
                    
                    # Event summary
                    with st.expander("View Event Types"):
                        event_counts = {}
                        for desc in raw.annotations.description:
                            event_type = desc.split('|')[0]  # Get base event type
                            event_counts[event_type] = event_counts.get(event_type, 0) + 1
                        
                        df_events = pd.DataFrame([
                            {'Event Type': k, 'Count': v}
                            for k, v in sorted(event_counts.items())
                        ])
                        st.dataframe(df_events, width='stretch')
                    
                    # First few annotations
                    with st.expander("View First 10 Annotations"):
                        annotations_data = []
                        for i in range(min(10, len(raw.annotations))):
                            annotations_data.append({
                                'Time (s)': f"{raw.annotations.onset[i]:.3f}",
                                'Description': raw.annotations.description[i]
                            })
                        st.dataframe(pd.DataFrame(annotations_data), 
                                   width='stretch')
                
                else:
                    st.warning("⚠️ No event markers found in file")
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                st.code(traceback.format_exc())
        
        elif st.session_state.raw is not None:
            st.info("ℹ️ File already loaded. Upload a new file to replace it.")
    
    # ========================================================================
    # Page 2: Signal Quality
    # ========================================================================
    
    elif page == "📊 Signal Quality":
        st.header("📊 Signal Quality Assessment")
        
        if st.session_state.raw is None:
            st.warning("⚠️ Please load a FIF file first (Load Data page)")
            return
        
        raw = st.session_state.raw
        
        # Duration selection
        duration = st.slider(
            "Analysis duration (seconds)",
            min_value=1.0,
            max_value=min(30.0, raw.times[-1]),
            value=10.0,
            step=1.0
        )
        
        # Check quality
        with st.spinner("Analyzing signal quality..."):
            quality_df = check_signal_quality(raw, duration=duration)
        
        # Display results
        st.subheader("Quality Metrics")
        st.dataframe(quality_df, width='stretch')
        
        # Interpretation guide
        with st.expander("📖 Interpretation Guide"):
            st.markdown("""
            **Quality Status:**
            - 🟢 **GOOD**: Signal quality is acceptable for analysis
            - 🟡 **POOR**: High noise, some epochs may be rejected
            - 🔴 **FLAT**: No signal detected - check electrode connection
            - 🔴 **VERY NOISY**: Excessive artifacts - check electrode placement
            
            **Typical Values:**
            - **Std Dev**: 10-50 µV (good), >100 µV (poor)
            - **Peak-to-Peak**: 50-200 µV (good), >500 µV (poor)
            """)
        
        # Plot raw data
        st.subheader("Raw Data Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            plot_duration = st.slider(
                "Plot duration (seconds)",
                min_value=1.0,
                max_value=min(20.0, raw.times[-1]),
                value=5.0,
                step=1.0
            )
        
        with col2:
            start_time = st.slider(
                "Start time (seconds)",
                min_value=0.0,
                max_value=max(0.0, raw.times[-1] - plot_duration),
                value=0.0,
                step=1.0
            )
        
        fig = plot_raw_data(raw, duration=plot_duration, start=start_time)
        st.pyplot(fig)
        plt.close()
        
        # Power spectral density
        st.subheader("Power Spectral Density")
        
        with st.spinner("Computing PSD..."):
            fig_psd = plot_psd(raw)
            st.pyplot(fig_psd)
            plt.close()
    
    # ========================================================================
    # Page 3: Preprocessing
    # ========================================================================
    
    elif page == "🔧 Preprocessing":
        st.header("🔧 Preprocessing & Filtering")

        if st.session_state.raw is None:
            st.warning("⚠️ Please load a FIF file first (Load Data page)")
            return

        raw = st.session_state.raw

        # --- Preset selector ---
        preset = st.radio(
            "Filter preset",
            options=["Custom", "Aggressive (data rescue)"],
            index=1,
            horizontal=True,
            help=(
                "**Custom** — full manual control.  \n"
                "**Aggressive** — IIR Butterworth 0.5–30 Hz (order 4) "
                "with steep rolloff. Designed to remove large slow-wave "
                "drift while preserving the P300 band."
            ),
        )
        is_aggressive = preset == "Aggressive (data rescue)"

        # --- Notch filter ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Notch Filter")
            apply_notch = st.checkbox("Apply notch filter", value=True)

            if apply_notch:
                notch_freqs = st.multiselect(
                    "Frequencies (Hz)",
                    options=[50, 60],
                    default=[50],
                    help="Remove powerline noise",
                )
            else:
                notch_freqs = None

        # --- Bandpass filter ---
        with col2:
            st.subheader("Bandpass Filter")
            apply_bandpass = st.checkbox("Apply bandpass filter", value=True)

            if apply_bandpass:
                if is_aggressive:
                    st.info(
                        "**Aggressive preset:** IIR Butterworth — "
                        "defaults tuned for drift removal. "
                        "Adjust parameters below if needed."
                    )
                    col_agg1, col_agg2 = st.columns(2)
                    with col_agg1:
                        lowcut = st.number_input(
                            "High-pass (Hz)",
                            min_value=0.1, max_value=50.0,
                            value=0.7, step=0.1,
                            key="preproc_agg_hp",
                        )
                    with col_agg2:
                        highcut = st.number_input(
                            "Low-pass (Hz)",
                            min_value=1.0, max_value=125.0,
                            value=30.0, step=1.0,
                            key="preproc_agg_lp",
                        )
                    filter_method = 'iir'
                    iir_order = st.slider(
                        "IIR Butterworth order",
                        min_value=2, max_value=8, value=4, step=1,
                        key="preproc_agg_order",
                        help="Higher order = steeper rolloff but more ringing.",
                    )
                else:
                    col_low, col_high = st.columns(2)
                    with col_low:
                        lowcut = st.number_input(
                            "High-pass (Hz)",
                            min_value=0.1,
                            max_value=50.0,
                            value=0.1,
                            step=0.1,
                        )
                    with col_high:
                        highcut = st.number_input(
                            "Low-pass (Hz)",
                            min_value=1.0,
                            max_value=125.0,
                            value=30.0,
                            step=1.0,
                        )
            else:
                lowcut = None
                highcut = None

        # --- Filter method (custom mode) ---
        if not is_aggressive:
            with st.expander("⚙️ Advanced: filter method"):
                filter_method = st.selectbox(
                    "Method",
                    options=["fir", "iir"],
                    index=0,
                    help=(
                        "**FIR** (default): linear-phase, no phase distortion.  \n"
                        "**IIR** (Butterworth): steeper rolloff, better for "
                        "aggressive drift removal. Uses zero-phase (forward-"
                        "backward) filtering to avoid P300 latency shifts."
                    ),
                )
                if filter_method == "iir":
                    iir_order = st.slider(
                        "Butterworth order",
                        min_value=2, max_value=8, value=4, step=1,
                        help="Higher order = steeper rolloff but more ringing.",
                    )
                else:
                    iir_order = 4

        # --- Apply ---
        if st.button("Apply Filters", type="primary"):
            with st.spinner("Applying filters..."):
                raw_filtered = apply_filters(
                    raw,
                    notch_freqs=notch_freqs,
                    lowcut=lowcut,
                    highcut=highcut,
                    method=filter_method if apply_bandpass else 'fir',
                    iir_order=iir_order if apply_bandpass else 4,
                )
                st.session_state.raw_filtered = raw_filtered

            st.success("✅ Filters applied successfully!")
        
        # Show comparison if filtered data exists
        if st.session_state.raw_filtered is not None:
            st.subheader("Before / After Comparison")
            
            plot_duration = st.slider(
                "Duration (seconds)",
                min_value=1.0,
                max_value=min(10.0, raw.times[-1]),
                value=5.0,
                step=1.0,
                key="filter_compare_duration"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original**")
                fig_orig = plot_raw_data(raw, duration=plot_duration)
                st.pyplot(fig_orig)
                plt.close()
            
            with col2:
                st.markdown("**Filtered**")
                fig_filt = plot_raw_data(st.session_state.raw_filtered, 
                                        duration=plot_duration)
                st.pyplot(fig_filt)
                plt.close()
            
            # PSD comparison
            st.subheader("PSD Comparison")
            
            with st.spinner("Computing PSDs..."):
                fig, axes = plt.subplots(1, 2, figsize=(14, 4))
                
                raw.compute_psd(fmax=100).plot(axes=axes[0], show=False)
                axes[0].set_title("Original", fontweight='bold')
                
                st.session_state.raw_filtered.compute_psd(fmax=100).plot(
                    axes=axes[1], show=False)
                axes[1].set_title("Filtered", fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Per-electrode PSD
            with st.expander("🔎 Per-electrode PSD"):
                psd_ch = st.selectbox(
                    "Channel",
                    options=raw.ch_names,
                    key="preproc_psd_ch",
                )
                with st.spinner(f"Computing PSD for {psd_ch}..."):
                    fig_ch_psd = plot_psd_single_channel(
                        raw, psd_ch,
                        raw_filtered=st.session_state.raw_filtered,
                    )
                    st.pyplot(fig_ch_psd)
                    plt.close()

    # ========================================================================
    # Page 3b: Block Signal Viewer
    # ========================================================================

    elif page == "🔍 Block Viewer":
        st.header("🔍 Block Signal Viewer")

        if st.session_state.raw is None:
            st.warning("⚠️ Please load a FIF file first (Load Data page)")
            return

        raw = st.session_state.raw

        # Extract block boundaries
        blocks = extract_block_boundaries(raw)
        if not blocks:
            st.warning(
                "⚠️ No block markers found in annotations. "
                "Expected `block_start|block=N` / `block_end|block=N`."
            )
            return

        sorted_block_nums = sorted(blocks)

        # Block selector
        col_sel, col_info = st.columns([1, 2])

        with col_sel:
            selected_block = st.selectbox(
                "Select block",
                options=sorted_block_nums,
                format_func=lambda b: (
                    f"Block {b}  "
                    f"({blocks[b]['start']:.1f}s – {blocks[b]['end']:.1f}s)"
                ),
            )

        blk = blocks[selected_block]
        blk_start, blk_end = blk['start'], blk['end']
        blk_duration = blk_end - blk_start

        with col_info:
            c1, c2, c3 = st.columns(3)
            c1.metric("Start", f"{blk_start:.1f} s")
            c2.metric("End", f"{blk_end:.1f} s")
            c3.metric("Duration", f"{blk_duration:.1f} s")

        # Collect events for the selected block
        block_events = extract_block_events(raw, blk_start, blk_end)
        n_probe = sum(1 for e in block_events if e['type'] == 'probe')
        n_irr = sum(1 for e in block_events if e['type'] == 'irrelevant')

        st.markdown(
            f"**Events in block:** {len(block_events)} total "
            f"({n_probe} probe, {n_irr} irrelevant)"
        )

        show_events = st.checkbox("Show event markers on plots", value=True)
        ev_for_plot = block_events if show_events else None

        # --- Stage 1: Raw (unfiltered) signal ---
        st.subheader("1. Raw signal (before filtering)")
        with st.spinner("Plotting raw block signal..."):
            fig_raw = plot_block_signal(
                raw, blk_start, blk_end,
                events=ev_for_plot,
                title=f"Block {selected_block} — Raw signal",
            )
            st.pyplot(fig_raw)
            plt.close()

        # --- Stage 2: Filtered signal ---
        st.subheader("2. Filtered signal (after filtering)")

        if st.session_state.raw_filtered is not None:
            with st.spinner("Plotting filtered block signal..."):
                fig_filt = plot_block_signal(
                    st.session_state.raw_filtered, blk_start, blk_end,
                    events=ev_for_plot,
                    title=f"Block {selected_block} — Filtered signal",
                )
                st.pyplot(fig_filt)
                plt.close()
        else:
            st.info(
                "ℹ️ No filtered data available. "
                "Apply filters on the Preprocessing page first."
            )

        # --- Stage 3: PSD comparison for the block ---
        st.subheader("3. Power Spectral Density (block segment)")

        with st.spinner("Computing block PSD..."):
            sfreq = raw.info['sfreq']
            s0 = max(0, int(blk_start * sfreq))
            s1 = min(int(blk_end * sfreq), len(raw.times))
            raw_block = mne.io.RawArray(
                raw.get_data()[:, s0:s1], raw.info.copy(), verbose='ERROR'
            )
            raw_filt_block = None

            if st.session_state.raw_filtered is not None:
                raw_filt = st.session_state.raw_filtered
                raw_filt_block = mne.io.RawArray(
                    raw_filt.get_data()[:, s0:s1],
                    raw_filt.info.copy(), verbose='ERROR',
                )
                fig_psd, axes_psd = plt.subplots(1, 2, figsize=(14, 4))
                raw_block.compute_psd(fmax=100).plot(
                    axes=axes_psd[0], show=False
                )
                axes_psd[0].set_title("Raw", fontweight='bold')
                raw_filt_block.compute_psd(fmax=100).plot(
                    axes=axes_psd[1], show=False
                )
                axes_psd[1].set_title("Filtered", fontweight='bold')
            else:
                fig_psd, ax_psd = plt.subplots(figsize=(10, 4))
                raw_block.compute_psd(fmax=100).plot(axes=ax_psd, show=False)
                ax_psd.set_title("Raw", fontweight='bold')

            fig_psd.suptitle(
                f"Block {selected_block} — PSD",
                fontsize=14, fontweight='bold',
            )
            plt.tight_layout()
            st.pyplot(fig_psd)
            plt.close()

            # Per-electrode PSD for the block
            with st.expander("🔎 Per-electrode PSD (block)"):
                blk_psd_ch = st.selectbox(
                    "Channel",
                    options=raw.ch_names,
                    key="blk_psd_ch",
                )
                with st.spinner(f"Computing PSD for {blk_psd_ch}..."):
                    fig_ch = plot_psd_single_channel(
                        raw_block, blk_psd_ch,
                        raw_filtered=raw_filt_block,
                    )
                    st.pyplot(fig_ch)
                    plt.close()

        # --- Stage 4: Epoched data for the block ---
        st.subheader("4. Epoched data (block epochs)")

        if st.session_state.epochs is not None:
            epochs = st.session_state.epochs

            # Filter epochs belonging to this block via their event samples
            sfreq = raw.info['sfreq']
            blk_start_samp = int(blk_start * sfreq) + raw.first_samp
            blk_end_samp = int(blk_end * sfreq) + raw.first_samp
            epoch_mask = (
                (epochs.events[:, 0] >= blk_start_samp)
                & (epochs.events[:, 0] <= blk_end_samp)
            )
            block_epoch_indices = np.nonzero(epoch_mask)[0]

            if len(block_epoch_indices) == 0:
                st.info("No epochs found for this block.")
            else:
                block_epochs = epochs[block_epoch_indices]
                st.markdown(
                    f"**Epochs in block {selected_block}:** "
                    f"{len(block_epochs)}"
                )

                data_ep = block_epochs.get_data() * 1e6
                times_ep = block_epochs.times * 1000

                n_show = st.slider(
                    "Epochs to overlay",
                    min_value=1,
                    max_value=min(len(block_epochs), 80),
                    value=min(len(block_epochs), 30),
                    key="block_epoch_slider",
                )

                fig_ep, axes_ep = plt.subplots(
                    len(block_epochs.ch_names), 1,
                    figsize=(14, 3 * len(block_epochs.ch_names)),
                    sharex=True,
                )
                if len(block_epochs.ch_names) == 1:
                    axes_ep = [axes_ep]

                for ch_idx, (ch_name, ax) in enumerate(
                    zip(block_epochs.ch_names, axes_ep)
                ):
                    for ep_idx in range(n_show):
                        ax.plot(
                            times_ep, data_ep[ep_idx, ch_idx, :],
                            alpha=0.25, linewidth=0.5, color='blue',
                        )
                    avg = data_ep[:n_show, ch_idx, :].mean(axis=0)
                    ax.plot(times_ep, avg, color='red', linewidth=2,
                            label='Average')
                    ax.axvline(0, color='black', linestyle='--', linewidth=1)
                    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
                    ax.set_ylabel(f'{ch_name}\n(\u00b5V)', fontsize=10)
                    ax.legend(loc='upper right', fontsize=8)
                    ax.grid(True, alpha=0.3)

                axes_ep[-1].set_xlabel('Time (ms)', fontsize=11)
                fig_ep.suptitle(
                    f"Block {selected_block} — Epochs "
                    f"({len(block_epochs)} total, showing {n_show})",
                    fontsize=14, fontweight='bold',
                )
                plt.tight_layout()
                st.pyplot(fig_ep)
                plt.close()
        else:
            st.info(
                "ℹ️ No epochs available. "
                "Create epochs on the Epoching page first."
            )

        # --- Event table ---
        with st.expander("📋 Block event list"):
            if block_events:
                df_ev = pd.DataFrame(block_events)
                df_ev['time'] = df_ev['time'].map(lambda t: f"{t:.3f}")
                st.dataframe(df_ev, width='stretch')
            else:
                st.info("No events in this block.")

    # ========================================================================
    # Page 4: Epoching
    # ========================================================================
    
    elif page == "📈 Epoching":
        st.header("📈 Epoch Creation")
        
        if st.session_state.raw is None:
            st.warning("⚠️ Please load a FIF file first (Load Data page)")
            return
        
        if st.session_state.events is None:
            st.error("❌ No events found in file. Cannot create epochs.")
            return
        
        # Use filtered data if available
        raw = (st.session_state.raw_filtered 
               if st.session_state.raw_filtered is not None 
               else st.session_state.raw)
        
        events = st.session_state.events
        event_id = st.session_state.event_id
        
        st.markdown("""
        Create epochs (time-locked segments) around stimulus events.
        """)
        
        # Epoch parameters
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            tmin = st.number_input(
                "Start time (s)",
                min_value=-2.0,
                max_value=0.0,
                value=-0.2,
                step=0.1,
                help="Time before stimulus onset",
            )

        with col2:
            tmax = st.number_input(
                "End time (s)",
                min_value=0.0,
                max_value=2.0,
                value=0.8,
                step=0.1,
                help="Time after stimulus onset",
            )

        with col3:
            baseline_mode = st.selectbox(
                "Baseline correction",
                options=["Yes", "No"],
                index=0,
            )

            if baseline_mode == "Yes":
                baseline_start_ms = st.number_input(
                    "Baseline start (ms)",
                    min_value=int(tmin * 1000),
                    max_value=0,
                    value=int(tmin * 1000),
                    step=10,
                    help=(
                        "Start of the baseline window (ms before stimulus). "
                        "Baseline is averaged from this point to 0 ms "
                        "(stimulus onset) and subtracted from the epoch."
                    ),
                )
                baseline = (baseline_start_ms / 1000.0, 0)
            else:
                baseline = None

        with col4:
            detrend_mode = st.selectbox(
                "Detrend",
                options=["None", "DC offset (0)", "Linear (1)"],
                index=1,
                help=(
                    "**None**: no detrending.  \n"
                    "**DC offset**: remove mean from each epoch.  \n"
                    "**Linear**: fit & remove linear trend — straightens "
                    "residual slope left after filtering. "
                    "Recommended for noisy / drift-heavy data."
                ),
            )

        detrend_map = {"None": None, "DC offset (0)": 0, "Linear (1)": 1}
        detrend = detrend_map[detrend_mode]
        
        # Artifact rejection
        st.subheader("Artifact Rejection")

        _rejection_options = [
            "None", "Static threshold",
            "Adaptive: IQR", "Adaptive: Z-score",
        ]
        if _AUTOREJECT_AVAILABLE:
            _rejection_options.append("Autoreject (ML-based)")

        col1, col2 = st.columns(2)

        with col1:
            rejection_method = st.selectbox(
                "Rejection method",
                options=_rejection_options,
                index=len(_rejection_options) - 1,
                help=(
                    "**None**: keep all epochs.  \n"
                    "**Static**: reject if any channel peak-to-peak > threshold.  \n"
                    "**Adaptive IQR**: reject outlier epochs > Q3 + k×IQR — "
                    "robust and scale-independent, recommended for "
                    "variable-quality signals.  \n"
                    "**Adaptive Z-score**: reject epochs with amplitude "
                    "z-score > k.  \n"
                    "**Autoreject (ML-based)**: uses cross-validation to "
                    "find optimal per-channel thresholds automatically — "
                    "fully objective, no manual tuning. "
                    "Requires `pip install autoreject`."
                )
            )

            if not _AUTOREJECT_AVAILABLE:
                st.caption(
                    "ℹ️ Install `autoreject` (`pip install autoreject`) "
                    "to enable ML-based rejection."
                )

        with col2:
            if rejection_method == "Static threshold":
                reject_threshold = st.slider(
                    "Threshold (µV)",
                    min_value=50.0, max_value=1000.0, value=100.0, step=10.0,
                    help="Reject epoch if any channel peak-to-peak exceeds "
                         "this value."
                )
                adaptive_method = None
                adaptive_k = None
            elif rejection_method in ("Adaptive: IQR", "Adaptive: Z-score"):
                reject_threshold = None
                adaptive_method = (
                    "iqr" if "IQR" in rejection_method else "zscore"
                )
                k_label = (
                    "IQR multiplier k"
                    if adaptive_method == "iqr"
                    else "Z-score threshold k"
                )
                k_help = (
                    "Reject epochs with amplitude > Q3 + k×IQR. "
                    "Lower k = stricter (e.g. 2.5 removes more, "
                    "4.0 removes less)."
                    if adaptive_method == "iqr"
                    else "Reject epochs with amplitude z-score > k. "
                    "Lower k = stricter (e.g. 2.5–3.5 typical)."
                )
                adaptive_k = st.slider(
                    k_label, min_value=1.0, max_value=6.0, value=3.0,
                    step=0.1, help=k_help,
                )
            elif rejection_method == "Autoreject (ML-based)":
                reject_threshold = None
                adaptive_method = None
                adaptive_k = None
                st.markdown(
                    "**Autoreject** uses Bayesian optimization and "
                    "cross-validation to determine optimal per-channel "
                    "thresholds. Can also interpolate individual bad "
                    "channels within an epoch."
                )
                ar_n_jobs = st.slider(
                    "Parallel jobs", min_value=1, max_value=8,
                    value=1, step=1,
                    help="Number of parallel jobs for cross-validation.",
                    key="ar_n_jobs",
                )
            else:  # None
                reject_threshold = None
                adaptive_method = None
                adaptive_k = None

        # S2-based trial rejection
        st.subheader("S2-Based Trial Rejection")
        st.caption(
            "Drop S1 epochs whose paired S2 response was incorrect "
            "or exceeded the reaction-time limit. Applied before "
            "artifact rejection."
        )
        s2r_col1, s2r_col2 = st.columns(2)
        with s2r_col1:
            s2_trial_rej_enabled = st.checkbox(
                "Enable S2 trial rejection", value=True,
                key="epoch_s2_trial_rej",
            )
        with s2r_col2:
            s2_max_rt_enabled = st.checkbox(
                "Apply max S2 RT limit", value=True,
                key="epoch_s2_max_rt_en",
                disabled=not s2_trial_rej_enabled,
            )
            s2_max_rt_val = st.number_input(
                "Max S2 RT (s)", 0.1, 10.0, 1.0, 0.1,
                key="epoch_s2_max_rt",
                disabled=not (s2_trial_rej_enabled and s2_max_rt_enabled),
            )

        # Create epochs
        if st.button("Create Epochs", type="primary"):
            try:
                with st.spinner("Creating epochs..."):
                    epochs = create_epochs(
                        raw, events, event_id,
                        tmin=tmin,
                        tmax=tmax,
                        baseline=baseline,
                        reject_threshold_uv=reject_threshold,
                        detrend=detrend,
                    )
                    n_before_s2_rej = len(epochs)
                    n_s2_rej_dropped = 0
                    s2_rej_log = []
                    if s2_trial_rej_enabled:
                        _s2_max = (
                            s2_max_rt_val
                            if s2_max_rt_enabled else None
                        )
                        epochs, n_s2_rej_dropped, s2_rej_log = (
                            reject_s1_by_s2_performance(
                                epochs,
                                st.session_state.raw,
                                max_rt=_s2_max,
                            )
                        )

                    epochs_ch_names = list(epochs.ch_names)
                    ptp_vals = None
                    n_adaptive_dropped = 0
                    autoreject_log = None
                    n_autoreject_dropped = 0

                    if adaptive_method is not None:
                        epochs, n_adaptive_dropped, ptp_vals, _ = (
                            adaptive_reject_epochs(
                                epochs, method=adaptive_method, k=adaptive_k
                            )
                        )

                    ar_info = None
                    if rejection_method == "Autoreject (ML-based)":
                        with st.spinner(
                            "Running autoreject (ML cross-validation)… "
                            "this may take a minute."
                        ):
                            epochs, autoreject_log, ar_info = (
                                autoreject_clean_epochs(
                                    epochs, n_jobs=ar_n_jobs,
                                )
                            )
                            n_autoreject_dropped = ar_info['n_dropped']

                    epochs, n_bl_dropped, _ = reject_noisy_baseline(
                        epochs, k=adaptive_k,
                    )
                    if n_bl_dropped > 0:
                        st.info(
                            f"Baseline QC: dropped {n_bl_dropped} "
                            f"epoch(s) with noisy baseline window."
                        )

                    st.session_state.epochs = epochs

                n_s1_events = len([
                    e for e in event_id.keys()
                    if 'S1_onset' in e or 's1_onset' in e.lower()
                ])
                n_dropped = n_s1_events - len(epochs)
                drop_pct = (
                    (n_dropped / n_s1_events * 100)
                    if n_s1_events > 0 else 0
                )

                st.success(f"✅ Created {len(epochs)} epochs!")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Kept", len(epochs))
                with col2:
                    st.metric("Total dropped", n_dropped)
                with col3:
                    st.metric("Drop rate", f"{drop_pct:.1f}%")
                with col4:
                    if adaptive_method:
                        st.metric("Adaptive dropped", n_adaptive_dropped)
                    elif ar_info is not None:
                        st.metric("Autoreject dropped",
                                  n_autoreject_dropped)

                if ptp_vals is not None:
                    with st.expander("📊 Epoch amplitude distribution"):
                        fig_ptp, ax_ptp = plt.subplots(figsize=(8, 3))
                        ax_ptp.hist(ptp_vals, bins=30, color='#1f77b4',
                                    alpha=0.7, edgecolor='black')
                        q1_v, q3_v = np.percentile(ptp_vals, [25, 75])
                        if adaptive_method == "iqr":
                            upper = q3_v + adaptive_k * (q3_v - q1_v)
                        else:
                            upper = (ptp_vals.mean()
                                     + adaptive_k * ptp_vals.std())
                        ax_ptp.axvline(
                            upper, color='red', linestyle='--',
                            linewidth=2,
                            label=f'Threshold ({upper:.0f} µV)',
                        )
                        ax_ptp.set_xlabel('Peak-to-peak amplitude (µV)')
                        ax_ptp.set_ylabel('Epoch count')
                        ax_ptp.set_title(
                            'Epoch amplitude distribution '
                            '(before adaptive rejection)'
                        )
                        ax_ptp.legend()
                        st.pyplot(fig_ptp)
                        plt.close()

                if ar_info is not None:
                    with st.expander("📊 Autoreject details"):
                        if autoreject_log is not None:
                            labels = autoreject_log.labels
                            n_interp_epochs = int(
                                (labels == 2).any(axis=1).sum()
                            )
                            n_good_epochs = int(
                                (~autoreject_log.bad_epochs).sum()
                            )
                            ar_c1, ar_c2, ar_c3 = st.columns(3)
                            ar_c1.metric("Kept", n_good_epochs)
                            ar_c2.metric("Rejected",
                                         n_autoreject_dropped)
                            ar_c3.metric("Interpolated (partial)",
                                         n_interp_epochs)

                            from matplotlib.colors import ListedColormap
                            ar_cmap = ListedColormap(
                                ['#2ca02c', '#d62728', '#ff7f0e']
                            )
                            fig_ar, ax_ar = plt.subplots(
                                figsize=(12, 3),
                            )
                            im = ax_ar.imshow(
                                labels.T, aspect='auto',
                                cmap=ar_cmap, vmin=0, vmax=2,
                                interpolation='nearest',
                            )
                            ax_ar.set_xlabel('Epoch index')
                            ax_ar.set_ylabel('Channel')
                            ax_ar.set_yticks(
                                range(len(epochs_ch_names))
                            )
                            ax_ar.set_yticklabels(epochs_ch_names)
                            ax_ar.set_title(
                                'Autoreject: Epoch × Channel labels'
                            )
                            cbar = plt.colorbar(
                                im, ax=ax_ar, ticks=[0, 1, 2],
                            )
                            cbar.set_ticklabels(
                                ['Good', 'Bad (rejected)',
                                 'Interpolated']
                            )
                            plt.tight_layout()
                            st.pyplot(fig_ar)
                            plt.close()
                        else:
                            thr = ar_info.get('threshold_uv', 0)
                            st.info(
                                f"**Mode:** global threshold "
                                f"(< 4 channels — full AutoReject "
                                f"requires ≥ 4).  \n"
                                f"**Optimal threshold:** "
                                f"**{thr:.1f} µV** "
                                f"(computed via Bayesian "
                                f"optimization).  \n"
                                f"**Dropped:** "
                                f"{n_autoreject_dropped} epochs"
                            )

                if s2_trial_rej_enabled:
                    with st.expander(
                        "📊 S2 trial rejection details", expanded=True,
                    ):
                        sr1, sr2, sr3 = st.columns(3)
                        with sr1:
                            st.metric(
                                "Before S2 rejection",
                                n_before_s2_rej,
                            )
                        with sr2:
                            st.metric(
                                "Dropped (S2)",
                                n_s2_rej_dropped,
                            )
                        with sr3:
                            s2_rej_pct = (
                                n_s2_rej_dropped / n_before_s2_rej * 100
                                if n_before_s2_rej > 0 else 0
                            )
                            st.metric(
                                "S2 drop rate",
                                f"{s2_rej_pct:.1f}%",
                            )

                        if s2_rej_log:
                            reasons = {}
                            for _, tnum, reason in s2_rej_log:
                                reasons[reason] = (
                                    reasons.get(reason, 0) + 1
                                )
                            reason_df = pd.DataFrame([
                                {'Reason': r, 'Count': c}
                                for r, c in reasons.items()
                            ])
                            st.dataframe(
                                reason_df, width='stretch',
                            )

                        st.markdown("**S2 ERP: Correct vs Incorrect**")
                        with st.spinner(
                            "Computing S2 comparison ERP…"
                        ):
                            fig_s2cmp, s2_stats = (
                                plot_s2_correct_vs_incorrect(
                                    st.session_state.raw,
                                    tmin=tmin,
                                    tmax=tmax,
                                    baseline=baseline,
                                    detrend=detrend,
                                )
                            )
                        sc1, sc2, sc3 = st.columns(3)
                        sc1.metric(
                            "S2 Correct",
                            s2_stats['n_correct'],
                        )
                        sc2.metric(
                            "S2 Incorrect",
                            s2_stats['n_incorrect'],
                        )
                        sc3.metric(
                            "S2 No response",
                            s2_stats['n_no_response'],
                        )
                        if fig_s2cmp is not None:
                            st.pyplot(fig_s2cmp)
                            plt.close(fig_s2cmp)
                        else:
                            st.info(
                                "No S2 onset events found — "
                                "comparison plot unavailable."
                            )

            except Exception as e:
                st.error(f"❌ Error creating epochs: {str(e)}")
                st.code(traceback.format_exc())
        
        # Display epoch information
        if st.session_state.epochs is not None:
            epochs = st.session_state.epochs
            
            st.subheader("Epoch Information")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("N Epochs", len(epochs))
            with col2:
                st.metric("Time Points", epochs.get_data().shape[2])
            with col3:
                st.metric("Channels", len(epochs.ch_names))
            with col4:
                st.metric("Epoch Duration", 
                         f"{epochs.times[-1] - epochs.times[0]:.2f} s")
            
            # Event counts
            with st.expander("Event Type Breakdown"):
                event_counts = []
                for event_type, event_idx in epochs.event_id.items():
                    # Count epochs for this event type
                    count = np.sum(epochs.events[:, 2] == event_idx)
                    base_type = event_type.split('|')[0]
                    event_counts.append({
                        'Event Type': base_type,
                        'Count': count
                    })
                
                df_counts = pd.DataFrame(event_counts)
                # Aggregate by base type
                df_counts = df_counts.groupby('Event Type').sum().reset_index()
                st.dataframe(df_counts, width='stretch')
            
            # Visualize epochs
            st.subheader("Epoch Visualization")
            
            n_epochs_plot = st.slider(
                "Number of epochs to plot",
                min_value=1,
                max_value=min(50, len(epochs)),
                value=min(10, len(epochs)),
                step=1
            )
            
            if st.button("Plot Epochs"):
                with st.spinner("Plotting..."):
                    # Create simplified plot
                    fig, axes = plt.subplots(len(epochs.ch_names), 1,
                                            figsize=(12, 3*len(epochs.ch_names)),
                                            sharex=True)
                    
                    if len(epochs.ch_names) == 1:
                        axes = [axes]
                    
                    data = epochs.get_data()[:n_epochs_plot] * 1e6  # µV
                    times = epochs.times * 1000  # ms
                    
                    for ch_idx, (ch_name, ax) in enumerate(zip(epochs.ch_names, axes)):
                        # Plot each epoch
                        for ep_idx in range(n_epochs_plot):
                            ax.plot(times, data[ep_idx, ch_idx, :], 
                                   alpha=0.3, linewidth=0.5, color='blue')
                        
                        # Plot average
                        avg = data[:, ch_idx, :].mean(axis=0)
                        ax.plot(times, avg, color='red', linewidth=2, 
                               label='Average')
                        
                        ax.axvline(0, color='black', linestyle='--', linewidth=1)
                        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
                        ax.set_ylabel(f'{ch_name}\n(µV)', fontsize=10)
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                    
                    axes[-1].set_xlabel('Time (ms)', fontsize=11)
                    fig.suptitle(f'First {n_epochs_plot} Epochs', 
                                fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    plt.close()
    
    # ========================================================================
    # Page 5: ERP Analysis
    # ========================================================================
    
    elif page == "🎯 ERP Analysis":
        st.header("🎯 Event-Related Potential Analysis")

        if st.session_state.epochs is None:
            st.warning("⚠️ Please create epochs first (Epoching page)")
            return

        epochs = st.session_state.epochs

        # ----------------------------------------------------------------
        # Analysis configuration
        # ----------------------------------------------------------------
        st.subheader("Analysis Configuration")
        conf_col1, conf_col2 = st.columns(2)

        with conf_col1:
            available_channels = list(epochs.ch_names)
            _default_analysis_chs = (
                ['Pz'] if 'Pz' in available_channels else available_channels
            )
            selected_channels = st.multiselect(
                "Channels to include",
                options=available_channels,
                default=_default_analysis_chs,
                help=(
                    "Subset of channels used for ERP waveforms and CTP-BAD. "
                    "Useful when only some electrodes have clean signal."
                )
            )
            if not selected_channels:
                st.warning("No channels selected — using all.")
                selected_channels = available_channels

        with conf_col2:
            stim_ids = extract_stim_ids(st.session_state.event_id or {})
            if stim_ids:
                target_options = ["probe (default)"] + stim_ids
                _tgt_default_idx = (
                    target_options.index('wolf')
                    if 'wolf' in target_options else 0
                )
                target_choice = st.selectbox(
                    "Target stimulus",
                    options=target_options,
                    index=_tgt_default_idx,
                    help=(
                        "Stimulus treated as 'target' in comparison.  \n"
                        "**probe (default)**: the designated probe object.  \n"
                        "**any stim_id**: test recognition of that specific object "
                        "(e.g. select an irrelevant to check for incidental learning)."
                    )
                )
                target_stim = None if target_choice == "probe (default)" else target_choice

                baseline_options = [s for s in stim_ids if s != target_stim]
                baseline_stims = st.multiselect(
                    "Baseline stimuli (compare against)",
                    options=baseline_options,
                    default=baseline_options,
                    help=(
                        "Average these objects as the baseline ERP. "
                        "Empty = all irrelevant events."
                    )
                )
                if not baseline_stims:
                    baseline_stims = None
            else:
                target_stim = None
                baseline_stims = None
                st.info("No stim_id found in events — using default probe vs irrelevant.")

        # ----------------------------------------------------------------
        # ERP Smoothing (low-pass for peak detection & visualization)
        # ----------------------------------------------------------------
        st.subheader("ERP Smoothing (for peak detection & visualization)")

        smooth_col1, smooth_col2 = st.columns(2)
        with smooth_col1:
            erp_smooth_enabled = st.checkbox(
                "Enable ERP low-pass smoothing",
                value=True,
                key="erp_smooth_enabled",
                help=(
                    "Apply a low-pass filter to the averaged ERPs before "
                    "peak detection and on the ERP plot overlay.  \n"
                    "Original epochs and mean-amplitude analysis are **not** "
                    "affected."
                ),
            )
        with smooth_col2:
            erp_lowpass_hz = st.number_input(
                "Low-pass cutoff (Hz)",
                min_value=1.0,
                max_value=30.0,
                value=12.0,
                step=0.5,
                key="erp_lowpass_hz",
                disabled=not erp_smooth_enabled,
                help=(
                    "Cutoff frequency for the zero-phase low-pass filter "
                    "applied to Evoked objects.  \n"
                    "Typical values: **6 Hz** (strong smoothing) to "
                    "**10 Hz** (mild smoothing)."
                ),
            )
        erp_lowpass_hz_value = erp_lowpass_hz if erp_smooth_enabled else None

        st.markdown("---")

        # Compute ERPs
        if st.button("Compute ERPs", type="primary"):
            with st.spinner("Computing ERPs..."):
                target_erp, baseline_erp, _, _ = compute_erps(
                    epochs,
                    target_stim=target_stim,
                    baseline_stims=baseline_stims,
                    channels=selected_channels
                )

                if target_erp is not None and baseline_erp is not None:
                    st.session_state.probe_erp = target_erp
                    st.session_state.irrelevant_erp = baseline_erp

                    st.success("✅ ERPs computed successfully!")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Target Trials", target_erp.nave)
                    with col2:
                        st.metric("Baseline Trials", baseline_erp.nave)

                else:
                    st.error("❌ Could not find target/baseline events.")
                    st.info(f"Available events (first 10): {list(epochs.event_id.keys())[:10]}")
        
        # Display ERP analysis
        if (st.session_state.probe_erp is not None and 
            st.session_state.irrelevant_erp is not None):
            
            probe_erp = st.session_state.probe_erp
            irrelevant_erp = st.session_state.irrelevant_erp
            
            # Plot ERPs
            st.subheader("ERP Waveforms")

            # -------------------------------------------------------
            # Individual P300 window (from S2 targets)
            # -------------------------------------------------------
            with st.expander("🔬 Individual P300 Window (from S2 targets)"):
                st.markdown(
                    "Compute an individualized P300 time window by finding "
                    "the P300 peak latency on **S2 target** responses, then "
                    "applying *peak ± margin* to the S1 analysis."
                )
                st.caption(
                    "S2 epochs use the **same rejection as Quick Pipeline → "
                    "S1** when those widgets exist in this session; otherwise "
                    "the checkbox fallback below applies."
                )

                raw_for_s2 = (
                    st.session_state.raw_filtered
                    if st.session_state.raw_filtered is not None
                    else st.session_state.raw
                )

                ip_col1, ip_col2, ip_col3 = st.columns(3)
                with ip_col1:
                    _ch_list = list(epochs.ch_names)
                    _default_chs = [
                        ch for ch in ['Pz'] if ch in _ch_list
                    ] or _ch_list[:1]
                    ip_channels = st.multiselect(
                        "Peak channel(s)",
                        options=_ch_list,
                        default=_default_chs,
                        key="ip_peak_channels",
                        help=(
                            "Channel(s) used to locate the P300 peak. "
                            "Multiple channels are averaged (higher SNR)."
                        ),
                    )
                    if not ip_channels:
                        ip_channels = _ch_list[:1]
                    ip_use_ar = st.checkbox(
                        "Autoreject S2 epochs",
                        value=_AUTOREJECT_AVAILABLE,
                        key="ip_use_autoreject",
                        disabled=not _AUTOREJECT_AVAILABLE,
                        help=(
                            "Fallback when Quick Pipeline was not opened: "
                            "if checked, use autoreject as S1-style "
                            "global rejection; otherwise skip that stage."
                        ),
                    )
                    ip_s2_trial_rej = st.checkbox(
                        "S2 trial rejection",
                        value=True,
                        key="ip_s2_trial_rej",
                        help=(
                            "Drop S2 target epochs whose behavioural "
                            "response was incorrect or missing."
                        ),
                    )
                    ip_s2_max_rt_en = st.checkbox(
                        "Apply max S2 RT limit",
                        value=True,
                        key="ip_s2_max_rt_en",
                        disabled=not ip_s2_trial_rej,
                    )
                    ip_s2_max_rt_val = st.number_input(
                        "Max S2 RT (s)", 0.1, 10.0, 1.0, 0.1,
                        key="ip_s2_max_rt",
                        disabled=not (ip_s2_trial_rej and ip_s2_max_rt_en),
                    )
                with ip_col2:
                    ip_search_min = st.number_input(
                        "Peak search start (s)", min_value=0.1,
                        max_value=1.0, value=0.25, step=0.05,
                        key="ip_search_min",
                    )
                    ip_search_max = st.number_input(
                        "Peak search end (s)", min_value=0.2,
                        max_value=1.5, value=0.60, step=0.05,
                        key="ip_search_max",
                    )
                with ip_col3:
                    ip_margin = st.number_input(
                        "Window margin ± (s)", min_value=0.05,
                        max_value=0.50, value=0.15, step=0.01,
                        key="ip_margin",
                        help="Half-width around the peak (peak ± margin).",
                    )

                if st.button("Compute Individual P300 Window",
                             key="btn_compute_ip300"):
                    try:
                        with st.spinner("Creating S2 target epochs…"):
                            _ip_s2_max = (
                                ip_s2_max_rt_val
                                if ip_s2_trial_rej and ip_s2_max_rt_en
                                else None
                            )
                            _ip_rej_map = {
                                "None": "none",
                                "Static threshold": "static",
                                "Adaptive: IQR": "iqr",
                                "Adaptive: Z-score": "zscore",
                                "Autoreject (ML-based)": "autoreject",
                            }
                            _pipe_lbl = st.session_state.get('pipe_s1_rej')
                            if _pipe_lbl is not None:
                                _ip_s1_rej = _ip_rej_map.get(
                                    _pipe_lbl, 'autoreject',
                                )
                                _ip_s1_thresh = None
                                _ip_s1_k = 3.0
                                if _pipe_lbl == "Static threshold":
                                    _ip_s1_thresh = float(
                                        st.session_state.get(
                                            'pipe_s1_thresh', 100.0,
                                        ),
                                    )
                                elif _pipe_lbl in (
                                    "Adaptive: IQR",
                                    "Adaptive: Z-score",
                                ):
                                    _ip_s1_k = float(
                                        st.session_state.get(
                                            'pipe_s1_k', 3.0,
                                        ),
                                    )
                            else:
                                _ip_s1_rej = (
                                    'autoreject' if ip_use_ar else 'none'
                                )
                                _ip_s1_thresh = None
                                _ip_s1_k = 3.0
                            _ip_ar_jobs = int(
                                st.session_state.get('pipe_ar_jobs', 1),
                            )
                            ip_result = compute_individual_p300_window(
                                raw=raw_for_s2,
                                events=st.session_state.events,
                                event_id=st.session_state.event_id,
                                peak_channels=ip_channels,
                                peak_search_tmin=ip_search_min,
                                peak_search_tmax=ip_search_max,
                                window_margin=ip_margin,
                                erp_lowpass_hz=erp_lowpass_hz_value,
                                s1_rejection=_ip_s1_rej,
                                s1_threshold_uv=(
                                    _ip_s1_thresh
                                    if _ip_s1_rej == 'static' else None
                                ),
                                s1_adaptive_k=_ip_s1_k,
                                ar_n_jobs=_ip_ar_jobs,
                                s2_trial_rejection=ip_s2_trial_rej,
                                s2_max_rt=_ip_s2_max,
                            )
                            st.session_state.individual_p300_window = ip_result
                        _perf_msg = ""
                        _n_perf = ip_result.get('n_s2_perf_dropped', 0)
                        if _n_perf:
                            _perf_msg = (
                                f" | S2 perf. rejected: {_n_perf}"
                            )
                        st.success(
                            f"Peak at **{ip_result['peak_time']*1000:.0f} ms** "
                            f"({ip_result['peak_amplitude']:.2f} µV) on "
                            f"**{ip_result['peak_channel']}** — "
                            f"Window: **{ip_result['window_start']*1000:.0f}–"
                            f"{ip_result['window_end']*1000:.0f} ms** "
                            f"({ip_result['n_s2_epochs']} S2 target epochs"
                            f"{_perf_msg})"
                        )
                    except Exception as exc:
                        st.error(f"Individual window error: {exc}")
                        st.code(traceback.format_exc())

                if st.session_state.individual_p300_window is not None:
                    iw = st.session_state.individual_p300_window
                    st.info(
                        f"**Stored individual window:** "
                        f"{iw['window_start']*1000:.0f}–"
                        f"{iw['window_end']*1000:.0f} ms "
                        f"(peak {iw['peak_time']*1000:.0f} ms on "
                        f"{iw['peak_channel']}, "
                        f"{iw['n_s2_epochs']} S2 target epochs)"
                    )

                    s2_erp = iw['s2_erp']
                    s2_erp_smooth = iw.get('s2_erp_smooth')
                    fig_s2, ax_s2 = plt.subplots(figsize=(10, 4))
                    s2_times_ms = s2_erp.times * 1000
                    _used = iw.get(
                        'peak_channels_used', [iw['peak_channel']]
                    )
                    _ch_idxs = [
                        s2_erp.ch_names.index(c)
                        for c in _used if c in s2_erp.ch_names
                    ] or [0]
                    s2_data = s2_erp.data[_ch_idxs].mean(axis=0) * 1e6

                    if s2_erp_smooth is not None:
                        ax_s2.plot(
                            s2_times_ms, s2_data, color='#1f77b4',
                            linewidth=1, alpha=0.3,
                            label=f"S2 ERP orig ({iw['peak_channel']})",
                        )
                        s2_smooth_data = (
                            s2_erp_smooth.data[_ch_idxs].mean(axis=0)
                            * 1e6
                        )
                        ax_s2.plot(
                            s2_times_ms, s2_smooth_data, color='#1f77b4',
                            linewidth=2, alpha=0.9,
                            label=(
                                f"S2 ERP LP {erp_lowpass_hz_value} Hz "
                                f"({iw['peak_channel']})"
                            ),
                        )
                    else:
                        ax_s2.plot(
                            s2_times_ms, s2_data, color='#1f77b4',
                            linewidth=2,
                            label=f"S2 target ERP ({iw['peak_channel']})",
                        )

                    ax_s2.axvline(
                        iw['peak_time'] * 1000, color='red',
                        linestyle='--', linewidth=1.5,
                        label=f"Peak ({iw['peak_time']*1000:.0f} ms)",
                    )
                    ax_s2.axvspan(
                        iw['window_start'] * 1000,
                        iw['window_end'] * 1000,
                        alpha=0.15, color='orange',
                        label='Individual window',
                    )
                    ax_s2.axhline(0, color='black', linewidth=0.5)
                    ax_s2.axvline(0, color='black', linestyle='--',
                                  linewidth=1)
                    ax_s2.set_xlabel('Time (ms)')
                    ax_s2.set_ylabel('Amplitude (µV)')
                    title_s2 = 'S2 Target ERP — Individual P300 Peak Detection'
                    if s2_erp_smooth is not None:
                        title_s2 += (
                            f'  [peak on LP {erp_lowpass_hz_value} Hz]'
                        )
                    ax_s2.set_title(title_s2)
                    ax_s2.legend(loc='best', fontsize=9)
                    ax_s2.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_s2)
                    plt.close()

            # -------------------------------------------------------
            # P300 window selection
            # -------------------------------------------------------
            use_individual = False
            if st.session_state.individual_p300_window is not None:
                use_individual = st.checkbox(
                    "Use individual P300 window for analysis",
                    value=True,
                    key="chk_use_individual_p300",
                )

            if use_individual:
                iw = st.session_state.individual_p300_window
                p300_start = iw['window_start']
                p300_end = iw['window_end']
                st.info(
                    f"Using individual window: "
                    f"**{p300_start*1000:.0f}–{p300_end*1000:.0f} ms** "
                    f"(peak at {iw['peak_time']*1000:.0f} ms on "
                    f"{iw['peak_channel']})"
                )
            else:
                col1, col2 = st.columns(2)
                with col1:
                    p300_start = st.number_input(
                        "P300 window start (s)",
                        min_value=0.0,
                        max_value=1.5,
                        value=0.3,
                        step=0.05,
                    )
                with col2:
                    p300_end = st.number_input(
                        "P300 window end (s)",
                        min_value=0.0,
                        max_value=1.5,
                        value=0.6,
                        step=0.05,
                    )
            
            fig_erp = plot_erps(probe_erp, irrelevant_erp,
                               p300_window=(p300_start, p300_end),
                               erp_lowpass_hz=erp_lowpass_hz_value)
            st.pyplot(fig_erp)
            plt.close()
            
            # P300 Analysis
            st.subheader("P300 Amplitude Analysis")
            
            p300_df = analyze_p300(probe_erp, irrelevant_erp, 
                                  tmin=p300_start, tmax=p300_end)
            
            st.dataframe(p300_df, width='stretch')
            
            # Interpretation
            with st.expander("📖 Interpretation Guide"):
                st.markdown("""
                **P300 Effect Strength:**
                - 🟢 **Strong** (>10 µV): Clear recognition effect
                - 🟡 **Moderate** (5-10 µV): Detectable effect
                - 🟡 **Weak** (2-5 µV): Marginal effect
                - 🔴 **None** (<2 µV): No recognition effect
                
                **Expected Pattern:**
                - Probe stimuli should elicit larger P300 than irrelevant stimuli
                - Effect is typically strongest at Pz (parietal) electrode
                - Effect appears 300-600ms after stimulus onset
                """)
            
            # Channel comparison
            st.subheader("Channel Comparison")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Extract P300 differences for each channel
            channels = []
            differences = []
            
            for _, row in p300_df.iterrows():
                channels.append(row['Channel'])
                diff_val = float(row['Difference (µV)'])
                differences.append(diff_val)
            
            colors = ['#2ca02c' if d > 5 else '#ff7f0e' if d > 2 else '#d62728' 
                     for d in differences]
            
            bars = ax.bar(channels, differences, color=colors, alpha=0.7, 
                         edgecolor='black', linewidth=1.5)
            
            ax.axhline(0, color='black', linestyle='-', linewidth=1)
            ax.axhline(5, color='gray', linestyle='--', linewidth=1, 
                      label='Moderate threshold (5 µV)')
            ax.axhline(10, color='green', linestyle='--', linewidth=1, 
                      label='Strong threshold (10 µV)')
            
            ax.set_xlabel('Channel', fontsize=12, fontweight='bold')
            ax.set_ylabel('P300 Difference (µV)', fontsize=12, fontweight='bold')
            ax.set_title('P300 Effect by Channel', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            st.pyplot(fig)
            plt.close()
            
            # CTP-BAD Analysis
            st.markdown("---")
            st.subheader("🎲 CTP-BAD Bootstrap Analysis")
            
            st.markdown("""
            **Bootstrap Amplitude Difference (BAD)** method for classification:
            - Uses bootstrap resampling to test if participant recognized probe
            - More robust than simple t-test (handles non-normal distributions)
            - Provides statistical confidence in guilty/innocent classification
            """)
            
            with st.expander("ℹ️ How CTP-BAD Works"):
                st.markdown("""
                **Method:**
                1. Extract amplitude from each epoch in the analysis window using the
                   selected measure (mean / peak-to-peak / baseline-to-peak).
                2. Bootstrap resample (default: 1000 iterations):
                   - Randomly sample probe epochs with replacement
                   - Randomly sample irrelevant epochs with replacement
                   - Calculate: `diff = measure(probe) - measure(irrelevant)`
                   - Count when `diff > 0` (probe > irrelevant)
                3. Calculate proportion: `p = count(diff > 0) / n_iterations`
                4. Classify:
                   - **Guilty** if `p ≥ 90%` (probe consistently larger)
                   - **Innocent** if `p < 90%` (no consistent difference)

                **Amplitude measures:**
                - **Mean**: signed average amplitude in the P300 window
                  (default, classic BAD). Best for noisy data — random sharp
                  spikes cancel out during averaging.
                - **Peak-to-peak (Rosenfeld)**: finds the maximum positive
                  peak in the P300 window, then the maximum negative trough
                  *after* that peak (up to a configurable end time, default
                  900 ms). Amplitude = peak − trough. Robust against slow
                  baseline drift and CNV, but sensitive to single sharp
                  artifacts.
                - **Peak-to-Peak (Peak-Valley)**: finds the global maximum
                  and minimum in the P300 window regardless of temporal
                  order. Amplitude = max − min. Simple, order-agnostic
                  measure.
                - **Baseline-to-peak**: maximum positive amplitude in the
                  P300 window relative to the zeroed baseline. Classic and
                  simple, but most sensitive to poor baseline quality.

                **Interpretation:**
                - 95-100%: Very strong evidence of recognition
                - 90-95%: Strong evidence of recognition (guilty)
                - 75-90%: Moderate evidence (inconclusive)
                - 50-75%: Weak/no evidence
                - <50%: Irrelevant larger than probe (unusual)
                """)

            # Parameters
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                bad_n_bootstrap = st.number_input(
                    "Bootstrap iterations",
                    min_value=100,
                    max_value=100000,
                    value=10000,
                    step=100,
                    help="More iterations = more stable results (slower)"
                )

            with col2:
                bad_threshold = st.slider(
                    "Guilty threshold",
                    min_value=0.50,
                    max_value=0.99,
                    value=0.75,
                    step=0.01,
                    help="Classification threshold (default: 0.75 = 75%)"
                )

            with col3:
                amp_method_label = st.selectbox(
                    "Amplitude measure",
                    options=[
                        "Mean",
                        "Peak-to-peak (Rosenfeld)",
                        "Peak-to-Peak (Peak-Valley)",
                        "Baseline-to-peak",
                    ],
                    index=2,
                    help=(
                        "**Mean**: signed average in the P300 window — best "
                        "for noisy data (random spikes cancel out).  \n"
                        "**Peak-to-peak (Rosenfeld)**: positive peak in P300 "
                        "window, then negative trough after it — robust "
                        "against baseline drift & CNV, but sensitive to "
                        "single sharp artifacts.  \n"
                        "**Peak-to-Peak (Peak-Valley)**: global max minus "
                        "global min in the window, regardless of temporal "
                        "order. Simple and order-agnostic.  \n"
                        "**Baseline-to-peak**: max positive amplitude "
                        "relative to zeroed baseline."
                    )
                )
                amp_method_map = {
                    "Mean": "mean",
                    "Peak-to-peak (Rosenfeld)": "peak_to_peak",
                    "Peak-to-Peak (Peak-Valley)": "peak_valley",
                    "Baseline-to-peak": "baseline_to_peak",
                }
                bad_amplitude_method = amp_method_map[amp_method_label]

                if bad_amplitude_method == 'peak_to_peak':
                    bad_p2p_tmax_neg = st.number_input(
                        "Neg. trough end (s)",
                        min_value=float(p300_end) + 0.05,
                        max_value=1.5,
                        value=0.9,
                        step=0.05,
                        help=(
                            "Rosenfeld method: after finding the positive "
                            "peak in the P300 window, the negative trough "
                            "is searched up to this time point."
                        ),
                    )
                else:
                    bad_p2p_tmax_neg = 0.9

            with col4:
                st.markdown("**Window:**")
                st.info(f"{p300_start:.2f}s – {p300_end:.2f}s")

            # Smoothing options for peak-based methods
            is_peak_method = bad_amplitude_method != 'mean'
            with st.expander(
                "🔧 Epoch smoothing (peak-based methods)",
                expanded=is_peak_method,
            ):
                if not is_peak_method:
                    st.info(
                        "Smoothing is only used with peak-based amplitude "
                        "measures (Rosenfeld, Peak-Valley, Baseline-to-peak). "
                        "**Mean** does not require smoothing."
                    )
                sm_col1, sm_col2, sm_col3 = st.columns(3)
                with sm_col1:
                    bad_smooth_label = st.selectbox(
                        "Smoothing method",
                        options=[
                            "None",
                            "Low-pass (Butterworth)",
                            "Moving average",
                        ],
                        index=1 if is_peak_method else 0,
                        key="bad_smooth_method",
                        disabled=not is_peak_method,
                        help=(
                            "**Low-pass**: zero-phase 4th-order Butterworth "
                            "via `scipy.signal.filtfilt`.  \n"
                            "**Moving average**: symmetric uniform window."
                        ),
                    )
                    bad_smooth_map = {
                        "None": None,
                        "Low-pass (Butterworth)": "lowpass",
                        "Moving average": "moving_average",
                    }
                    bad_smoothing_method = bad_smooth_map[bad_smooth_label]
                with sm_col2:
                    bad_smooth_lp_hz = st.number_input(
                        "LP cutoff (Hz)",
                        min_value=1.0,
                        max_value=30.0,
                        value=12.0,
                        step=0.5,
                        key="bad_smooth_lp_hz",
                        disabled=(bad_smoothing_method != 'lowpass'),
                        help="Butterworth low-pass cutoff frequency.",
                    )
                with sm_col3:
                    bad_smooth_ma_ms = st.number_input(
                        "MA window (ms)",
                        min_value=10.0,
                        max_value=500.0,
                        value=100.0,
                        step=10.0,
                        key="bad_smooth_ma_ms",
                        disabled=(bad_smoothing_method != 'moving_average'),
                        help="Moving average window length in milliseconds.",
                    )

            # Run CTP-BAD
            if st.button("🎲 Run CTP-BAD Analysis", type="primary"):
                try:
                    with st.spinner(f"Running bootstrap ({bad_n_bootstrap} iterations)..."):
                        bad_results = ctp_bad_analysis(
                            epochs,
                            tmin=p300_start,
                            tmax=p300_end,
                            n_bootstrap=int(bad_n_bootstrap),
                            threshold=bad_threshold,
                            channels=selected_channels,
                            target_stim=target_stim,
                            baseline_stims=baseline_stims,
                            amplitude_method=bad_amplitude_method,
                            p2p_tmax_negative=bad_p2p_tmax_neg,
                            smoothing_method=bad_smoothing_method,
                            smoothing_lowpass_hz=bad_smooth_lp_hz,
                            smoothing_window_ms=bad_smooth_ma_ms,
                        )
                        
                        # Store in session state
                        st.session_state.bad_results = bad_results
                    
                    st.success("✅ CTP-BAD analysis complete!")
                    
                except Exception as e:
                    st.error(f"❌ Error in CTP-BAD analysis: {str(e)}")
                    st.code(traceback.format_exc())
            
            # Display results
            if 'bad_results' in st.session_state and st.session_state.bad_results is not None:
                bad_results = st.session_state.bad_results
                
                # Overall verdict
                st.markdown("---")
                st.subheader("🔍 Classification Result")
                
                col1, col2, col_pv, col3 = st.columns([1, 1, 1, 2])
                
                with col1:
                    st.metric(
                        "Classification",
                        bad_results['overall_classification'],
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Max Proportion",
                        f"{bad_results['max_proportion']*100:.1f}%",
                        delta=None
                    )

                with col_pv:
                    st.metric(
                        "p-value",
                        f"{bad_results.get('p_value', 1 - bad_results['max_proportion']):.4f}",
                    )
                
                with col3:
                    verdict_color = "red" if bad_results['overall_classification'] == "GUILTY" else "green"
                    target_lbl = bad_results.get('target_label', 'probe')
                    st.markdown(
                        f"**Target:** `{target_lbl}`  \n"
                        f"**Verdict:** :{verdict_color}[{bad_results['verdict']}]"
                    )
                
                # Analysis details
                with st.expander("📊 Analysis Details"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Probe Epochs", bad_results['n_probe_epochs'])
                    with col2:
                        st.metric("Irrelevant Epochs", bad_results['n_irrelevant_epochs'])
                    with col3:
                        st.metric("Bootstrap Iterations", bad_results['n_bootstrap'])
                    
                    method_labels = {
                        'mean': 'Mean amplitude',
                        'peak_to_peak': 'Peak-to-peak (Rosenfeld)',
                        'peak_valley': 'Peak-to-Peak (Peak-Valley)',
                        'baseline_to_peak': 'Baseline-to-peak',
                    }
                    amp_m = bad_results.get('amplitude_method', 'mean')
                    method_str = method_labels.get(amp_m, 'Mean amplitude')
                    extra = ""
                    if amp_m == 'peak_to_peak':
                        extra = (
                            f" | **Neg. trough end:** "
                            f"{bad_results.get('p2p_tmax_negative', 0.9):.2f}s"
                        )
                    sm = bad_results.get('smoothing_method')
                    if sm == 'lowpass':
                        extra += (
                            f" | **Smoothing:** LP "
                            f"{bad_results.get('smoothing_lowpass_hz', 12.0)} Hz"
                        )
                    elif sm == 'moving_average':
                        extra += (
                            f" | **Smoothing:** MA "
                            f"{bad_results.get('smoothing_window_ms', 100.0):.0f} ms"
                        )
                    st.info(
                        f"**Threshold:** {bad_results['threshold']*100:.0f}% | "
                        f"**Best Channel:** {bad_results['max_channel']} | "
                        f"**Amplitude measure:** {method_str}{extra}"
                    )
                
                # Per-channel results
                st.subheader("Per-Channel Results")
                st.dataframe(bad_results['channel_results'], width='stretch')
                
                # Visualization
                st.subheader("Bootstrap Proportions by Channel")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                channels = list(bad_results['channel_results']['Channel'])
                proportions = [p * 100 for p in bad_results['bootstrap_proportions']]
                
                # Color bars based on classification
                colors = ['#d62728' if p >= bad_threshold * 100 else '#2ca02c' 
                         for p in proportions]
                
                bars = ax.bar(channels, proportions, color=colors, alpha=0.7,
                             edgecolor='black', linewidth=1.5)
                
                # Add threshold line
                ax.axhline(bad_threshold * 100, color='black', linestyle='--',
                          linewidth=2, label=f'Guilty threshold ({bad_threshold*100:.0f}%)')
                
                # Add 50% reference line
                ax.axhline(50, color='gray', linestyle=':', linewidth=1,
                          label='Chance level (50%)')
                
                # Styling
                ax.set_xlabel('Channel', fontsize=12, fontweight='bold')
                ax.set_ylabel('Bootstrap Proportion (%)', fontsize=12, fontweight='bold')
                ax.set_title('CTP-BAD: Probe > Irrelevant (%)', 
                           fontsize=14, fontweight='bold')
                ax.set_ylim(0, 100)
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar, prop in zip(bars, proportions):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{prop:.1f}%',
                           ha='center', va='bottom', fontweight='bold')
                
                st.pyplot(fig)
                plt.close()
                
                # Interpretation guide
                with st.expander("📖 Result Interpretation"):
                    st.markdown("""
                    **Classification Guide:**
                    - 🔴 **Guilty**: Bootstrap proportion ≥ threshold (default 90%)
                      - Participant consistently shows larger P300 to probe
                      - Strong evidence of recognition
                    - 🟢 **Innocent**: Bootstrap proportion < threshold
                      - No consistent difference between probe and irrelevant
                      - Participant likely did not recognize probe
                    
                    **Confidence Levels:**
                    - **High**: Very clear result (>95% or <60%)
                    - **Moderate**: Clear result (90-95% or 60-75%)
                    - **Low**: Borderline result (75-90%)
                    
                    **Expected Patterns:**
                    - **Guilty participant**: 90-100% on multiple channels, strongest at Pz
                    - **Innocent participant**: 40-60% (around chance), no clear pattern
                    - **Borderline**: 75-90%, consider retesting or additional analysis
                    
                    **Notes:**
                    - Results < 50% are unusual (irrelevant > probe)
                    - All channels should show consistent pattern
                    - Pz typically most sensitive for P300
                    """)
    
    # ========================================================================
    # Page 6: Quick Pipeline
    # ========================================================================

    elif page == "⚡ Quick Pipeline":
        st.header("⚡ Quick CTP-BAD Pipeline")
        st.markdown(
            "Configure and run the full analysis chain in one click. "
            "All steps are configurable — defaults match the recommended "
            "protocol."
        )

        # --- Import config from Grid Search ---
        with st.expander(
            "Import pipeline config from Grid Search / Optuna",
            expanded=False,
        ):
            st.caption(
                "Upload a `best_pipeline_config.json` exported from "
                "the **Grid Search** tab, or click **Apply best config** "
                "there directly."
            )
            _cfg_upload = st.file_uploader(
                "Pipeline config JSON",
                type=['json'],
                key="pipe_cfg_upload",
            )
            if _cfg_upload is not None:
                _fid = (_cfg_upload.name, _cfg_upload.size)
                if st.session_state.get('_pipe_last_cfg_fid') != _fid:
                    try:
                        _imported = json.loads(
                            _cfg_upload.read().decode(),
                        )
                        _apply_config_to_widgets(_imported)
                        st.session_state['_pipe_last_cfg_fid'] = _fid
                        st.session_state['_pipe_cfg_imported'] = True
                        st.rerun()
                    except (json.JSONDecodeError, KeyError) as exc:
                        st.error(f"Invalid config file: {exc}")

            if st.session_state.pop('_pipe_cfg_imported', False):
                st.success(
                    "Config loaded and applied to all widgets below."
                )

        # --- Input mode ---
        input_mode = st.radio(
            "Input mode",
            [
                "Single file (loaded)",
                "Batch (upload multiple .fif)",
                "Batch with groups (innocent / guilty)",
            ],
            horizontal=True,
            key="pipeline_input_mode",
        )

        batch_files = None
        grouped_batch = None

        if input_mode == "Batch (upload multiple .fif)":
            batch_files = st.file_uploader(
                "Upload .fif files",
                type=['fif'],
                accept_multiple_files=True,
                key="pipeline_batch_upload",
            )
        elif input_mode == "Batch with groups (innocent / guilty)":
            st.caption(
                "Upload two separate sets of .fif files — one per "
                "ground-truth group. Classification metrics will be "
                "computed against these labels."
            )
            col_inn, col_gui = st.columns(2)
            with col_inn:
                innocent_files = st.file_uploader(
                    "Innocent .fif files",
                    type=['fif'],
                    accept_multiple_files=True,
                    key="pipeline_innocent_upload",
                )
            with col_gui:
                guilty_files = st.file_uploader(
                    "Guilty .fif files",
                    type=['fif'],
                    accept_multiple_files=True,
                    key="pipeline_guilty_upload",
                )
            if innocent_files or guilty_files:
                grouped_batch = {
                    'innocent': innocent_files or [],
                    'guilty': guilty_files or [],
                }
            st.markdown("**Grouped evaluation**")
            pipe_grouped_loocv = st.checkbox(
                "LOOCV metrics on max_proportion (leave-one-out cutoffs; "
                "F-β on training folds — same scheme as Grid Search)",
                value=False,
                key="pipe_grouped_loocv_eval",
            )
            pipe_grouped_loocv_fbeta = st.number_input(
                "F-β weight for LOOCV threshold selection",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                key="pipe_grouped_loocv_fbeta",
                disabled=not pipe_grouped_loocv,
            )
        else:
            if st.session_state.raw is None:
                st.warning(
                    "⚠️ No file loaded. Go to **Load Data** first, "
                    "or switch to **Batch** mode."
                )

        # --- Determine available channels for UI defaults ---
        _avail_chs = []
        if st.session_state.raw is not None:
            _avail_chs = list(st.session_state.raw.ch_names)
        elif batch_files or grouped_batch:
            _avail_chs = ['Fz', 'Cz', 'Pz']

        # ===============================================================
        # Pipeline configuration
        # ===============================================================
        st.markdown("---")
        st.subheader("Pipeline Configuration")

        # --- Step 1: Filtering ---
        with st.expander("**Step 1 — Filtering**", expanded=False):
            pipe_filter = st.radio(
                "Filter preset",
                ["Skip", "Aggressive (data rescue)", "Custom"],
                index=1,
                key="pipe_filter_preset",
                horizontal=True,
            )
            pipe_notch = [50]
            pipe_hp = 0.7
            pipe_lp = 30.0
            pipe_fmethod = 'iir'
            pipe_iir_order = 4

            if pipe_filter == "Aggressive (data rescue)":
                st.caption(
                    "IIR Butterworth bandpass with steep rolloff. "
                    "Adjust defaults below if needed."
                )
                fa1, fa2 = st.columns(2)
                with fa1:
                    pipe_notch = st.multiselect(
                        "Notch (Hz)", [50, 60], default=[50],
                        key="pipe_notch_agg",
                    )
                    pipe_hp = st.number_input(
                        "High-pass (Hz)", 0.1, 50.0, 0.7, 0.1,
                        key="pipe_hp_agg",
                    )
                with fa2:
                    pipe_lp = st.number_input(
                        "Low-pass (Hz)", 1.0, 125.0, 30.0, 1.0,
                        key="pipe_lp_agg",
                    )
                    pipe_iir_order = st.slider(
                        "IIR order", 2, 8, 4, key="pipe_iir_order_agg",
                    )

            elif pipe_filter == "Custom":
                fc1, fc2 = st.columns(2)
                with fc1:
                    pipe_notch = st.multiselect(
                        "Notch (Hz)", [50, 60], default=[50],
                        key="pipe_notch",
                    )
                    pipe_hp = st.number_input(
                        "High-pass (Hz)", 0.1, 50.0, 0.1, 0.1,
                        key="pipe_hp",
                    )
                with fc2:
                    pipe_lp = st.number_input(
                        "Low-pass (Hz)", 1.0, 125.0, 30.0, 1.0,
                        key="pipe_lp",
                    )
                    pipe_fmethod = st.selectbox(
                        "Method", ["fir", "iir"], index=1,
                        key="pipe_fmethod",
                    )
                    if pipe_fmethod == 'iir':
                        pipe_iir_order = st.slider(
                            "IIR order", 2, 8, 4, key="pipe_iir_order",
                        )

        # --- Step 2: S2 Target Epochs ---
        with st.expander("**Step 2 — S2 Target Epochs**", expanded=False):
            s2c1, s2c2, s2c3 = st.columns(3)
            with s2c1:
                pipe_s2_tmin = st.number_input(
                    "S2 tmin (s)", -2.0, 0.0, -0.2, 0.1,
                    key="pipe_s2_tmin",
                )
                pipe_s2_tmax = st.number_input(
                    "S2 tmax (s)", 0.1, 2.0, 0.8, 0.1,
                    key="pipe_s2_tmax",
                )
            with s2c2:
                pipe_s2_bl = st.checkbox(
                    "Baseline correction", True, key="pipe_s2_bl",
                )
                pipe_s2_detrend = st.selectbox(
                    "Detrend", ["None", "DC offset (0)", "Linear (1)"],
                    index=1, key="pipe_s2_detrend",
                )
            with s2c3:
                _s2_rej_opts = [
                    "None", "Static threshold", "Adaptive: IQR",
                    "Adaptive: Z-score",
                ]
                if _AUTOREJECT_AVAILABLE:
                    _s2_rej_opts.append("Autoreject (ML-based)")
                pipe_s2_rej = st.selectbox(
                    "Rejection method", _s2_rej_opts,
                    index=len(_s2_rej_opts) - 1,
                    key="pipe_s2_rej",
                )
                if pipe_s2_rej == "Static threshold":
                    pipe_s2_thresh = st.number_input(
                        "Threshold (µV)", 50.0, 1000.0, 100.0, 10.0,
                        key="pipe_s2_thresh",
                    )
                else:
                    pipe_s2_thresh = None

        # --- Step 3: Individual P300 Window ---
        with st.expander(
            "**Step 3 — Individual P300 Window**", expanded=False,
        ):
            pipe_use_iw = st.checkbox(
                "Enable individual P300 window from S2",
                value=True,
                key="pipe_use_iw",
            )
            iw1, iw2 = st.columns(2)
            with iw1:
                _default_peak = (
                    [c for c in ['Pz'] if c in _avail_chs]
                    or _avail_chs[:1]
                )
                pipe_peak_chs = st.multiselect(
                    "Peak channel(s)",
                    options=_avail_chs or ['Pz'],
                    default=_default_peak or ['Pz'],
                    key="pipe_peak_chs",
                    help="Averaged for peak detection (higher SNR).",
                )
                if not pipe_peak_chs:
                    pipe_peak_chs = _default_peak or ['Pz']
                pipe_search_min = st.number_input(
                    "Peak search start (s)", 0.1, 1.0, 0.25, 0.05,
                    key="pipe_search_min",
                )
                pipe_search_max = st.number_input(
                    "Peak search end (s)", 0.2, 1.5, 0.60, 0.05,
                    key="pipe_search_max",
                )
            with iw2:
                pipe_margin = st.number_input(
                    "Window margin ± (s)", 0.05, 0.50, 0.15, 0.01,
                    key="pipe_margin",
                )
                pipe_s2_erp_lp = st.checkbox(
                    "ERP smoothing for peak detection",
                    value=True,
                    key="pipe_s2_erp_lp_en",
                )
                pipe_s2_erp_lp_hz = st.number_input(
                    "LP cutoff (Hz)", 1.0, 30.0, 12.0, 0.5,
                    key="pipe_s2_erp_lp_hz",
                    disabled=not pipe_s2_erp_lp,
                )
            pipe_manual_tmin = 0.3
            pipe_manual_tmax = 0.6
            if not pipe_use_iw:
                mc1, mc2 = st.columns(2)
                with mc1:
                    pipe_manual_tmin = st.number_input(
                        "Manual window start (s)", 0.0, 1.5, 0.3, 0.05,
                        key="pipe_man_tmin",
                    )
                with mc2:
                    pipe_manual_tmax = st.number_input(
                        "Manual window end (s)", 0.0, 1.5, 0.6, 0.05,
                        key="pipe_man_tmax",
                    )

        # --- Step 4: S1 Probe/Irrelevant Epochs ---
        with st.expander(
            "**Step 4 — S1 Probe/Irrelevant Epochs**", expanded=False,
        ):
            s1c1, s1c2, s1c3 = st.columns(3)
            with s1c1:
                pipe_s1_tmin = st.number_input(
                    "S1 tmin (s)", -2.0, 0.0, -0.2, 0.1,
                    key="pipe_s1_tmin",
                )
                pipe_s1_tmax = st.number_input(
                    "S1 tmax (s)", 0.1, 2.0, 0.8, 0.1,
                    key="pipe_s1_tmax",
                )
            with s1c2:
                pipe_s1_bl = st.checkbox(
                    "Baseline correction", True, key="pipe_s1_bl",
                )
                pipe_s1_detrend = st.selectbox(
                    "Detrend", ["None", "DC offset (0)", "Linear (1)"],
                    index=1, key="pipe_s1_detrend",
                )
            with s1c3:
                _s1_rej_opts = [
                    "None", "Static threshold", "Adaptive: IQR",
                    "Adaptive: Z-score",
                ]
                if _AUTOREJECT_AVAILABLE:
                    _s1_rej_opts.append("Autoreject (ML-based)")
                pipe_s1_rej = st.selectbox(
                    "Rejection method", _s1_rej_opts,
                    index=len(_s1_rej_opts) - 1,
                    key="pipe_s1_rej",
                )
                pipe_s1_thresh = None
                pipe_s1_k = 3.0
                if pipe_s1_rej == "Static threshold":
                    pipe_s1_thresh = st.number_input(
                        "Threshold (µV)", 50.0, 1000.0, 100.0, 10.0,
                        key="pipe_s1_thresh",
                    )
                elif pipe_s1_rej in ("Adaptive: IQR", "Adaptive: Z-score"):
                    pipe_s1_k = st.slider(
                        "k multiplier", 1.0, 6.0, 3.0, 0.1,
                        key="pipe_s1_k",
                    )

            st.markdown("---")
            st.markdown("**S2-based trial rejection**")
            st.caption(
                "Drop S1 epochs whose paired S2 response was incorrect "
                "or exceeded the reaction-time limit."
            )
            s2r1, s2r2 = st.columns(2)
            with s2r1:
                pipe_s2_trial_rej = st.checkbox(
                    "Enable S2 trial rejection", value=True,
                    key="pipe_s2_trial_rej",
                )
            with s2r2:
                pipe_s2_max_rt_en = st.checkbox(
                    "Apply max S2 RT limit", value=True,
                    key="pipe_s2_max_rt_en",
                    disabled=not pipe_s2_trial_rej,
                )
                pipe_s2_max_rt = st.number_input(
                    "Max S2 RT (s)", 0.1, 10.0, 1.0, 0.1,
                    key="pipe_s2_max_rt",
                    disabled=not (pipe_s2_trial_rej and pipe_s2_max_rt_en),
                )

        # --- Step 5: CTP-BAD Bootstrap ---
        with st.expander(
            "**Step 5 — CTP-BAD Bootstrap**", expanded=False,
        ):
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                pipe_bad_chs_override = st.checkbox(
                    "Override analysis channels",
                    value=False,
                    key="pipe_bad_chs_override",
                    help="By default uses the same channels as Step 3.",
                )
                if pipe_bad_chs_override:
                    pipe_bad_chs = st.multiselect(
                        "CTP-BAD channels",
                        options=_avail_chs or ['Pz'],
                        default=_avail_chs or ['Pz'],
                        key="pipe_bad_chs",
                    )
                else:
                    pipe_bad_chs = pipe_peak_chs

                pipe_amp_method = st.selectbox(
                    "Amplitude method",
                    [
                        "Mean",
                        "Peak-to-peak (Rosenfeld)",
                        "Peak-to-Peak (Peak-Valley)",
                        "Baseline-to-peak",
                    ],
                    index=2,
                    key="pipe_amp_method",
                )
                if pipe_amp_method == "Peak-to-peak (Rosenfeld)":
                    pipe_p2p_neg = st.number_input(
                        "Neg. trough end (s)", 0.5, 1.5, 0.9, 0.05,
                        key="pipe_p2p_neg",
                    )
                else:
                    pipe_p2p_neg = 0.9

            with bc2:
                pipe_smooth_method = st.selectbox(
                    "Epoch smoothing",
                    ["None", "Low-pass (Butterworth)", "Moving average"],
                    index=1 if pipe_amp_method != "Mean" else 0,
                    key="pipe_smooth_method",
                )
                pipe_smooth_lp = st.number_input(
                    "LP cutoff (Hz)", 1.0, 30.0, 12.0, 0.5,
                    key="pipe_smooth_lp",
                    disabled=pipe_smooth_method != "Low-pass (Butterworth)",
                )
                pipe_smooth_ma = st.number_input(
                    "MA window (ms)", 10.0, 500.0, 100.0, 10.0,
                    key="pipe_smooth_ma",
                    disabled=pipe_smooth_method != "Moving average",
                )
                if pipe_amp_method == "Mean":
                    st.caption(
                        "Smoothing is ignored for Mean amplitude."
                    )

            with bc3:
                pipe_n_boot = st.number_input(
                    "Bootstrap iterations", 100, 100000, 10000, 100,
                    key="pipe_n_boot",
                )
                pipe_threshold = st.slider(
                    "Guilty threshold", 0.50, 0.99, 0.75, 0.01,
                    key="pipe_threshold",
                )

            # Target/baseline stimulus (uses event_id from loaded data)
            _ev_id = st.session_state.event_id or {}
            _stim_ids = extract_stim_ids(_ev_id)
            if _stim_ids:
                t_col, b_col = st.columns(2)
                with t_col:
                    _tgt_opts = ["probe (default)"] + _stim_ids
                    _tgt_choice = st.selectbox(
                        "Target stimulus", _tgt_opts, 0,
                        key="pipe_tgt_stim",
                    )
                    pipe_tgt_stim = (
                        None if _tgt_choice == "probe (default)"
                        else _tgt_choice
                    )
                with b_col:
                    _bl_opts = [
                        s for s in _stim_ids if s != pipe_tgt_stim
                    ]
                    pipe_bl_stims = st.multiselect(
                        "Baseline stimuli", _bl_opts, _bl_opts,
                        key="pipe_bl_stims",
                    ) or None
            else:
                pipe_tgt_stim = None
                pipe_bl_stims = None

        pipe_ar_jobs = st.number_input(
            "Autoreject parallel jobs", 1, 8, 1, key="pipe_ar_jobs",
        )

        # ===============================================================
        # Build config dict
        # ===============================================================
        def _build_cfg():
            _rej_map = {
                "None": "none",
                "Static threshold": "static",
                "Adaptive: IQR": "iqr",
                "Adaptive: Z-score": "zscore",
                "Autoreject (ML-based)": "autoreject",
            }
            return {
                'filter_preset': {
                    'Skip': 'skip',
                    'Aggressive (data rescue)': 'aggressive',
                    'Custom': 'custom',
                }[pipe_filter],
                'notch_freqs': pipe_notch or None,
                'hp_cutoff': pipe_hp,
                'lp_cutoff': pipe_lp,
                'filter_method': pipe_fmethod,
                'iir_order': pipe_iir_order,
                # S2
                's2_tmin': pipe_s2_tmin,
                's2_tmax': pipe_s2_tmax,
                's2_baseline': pipe_s2_bl,
                's2_detrend': pipe_s2_detrend,
                's2_rejection': _rej_map.get(pipe_s2_rej, 'autoreject'),
                's2_threshold_uv': pipe_s2_thresh if pipe_s2_rej == "Static threshold" else None,
                # Individual window
                'use_individual_window': pipe_use_iw,
                'peak_channels': pipe_peak_chs,
                'peak_search_tmin': pipe_search_min,
                'peak_search_tmax': pipe_search_max,
                'window_margin': pipe_margin,
                's2_erp_lowpass_hz': pipe_s2_erp_lp_hz if pipe_s2_erp_lp else None,
                'manual_tmin': pipe_manual_tmin if not pipe_use_iw else 0.3,
                'manual_tmax': pipe_manual_tmax if not pipe_use_iw else 0.6,
                # S1
                's1_tmin': pipe_s1_tmin,
                's1_tmax': pipe_s1_tmax,
                's1_baseline': pipe_s1_bl,
                's1_detrend': pipe_s1_detrend,
                's1_rejection': _rej_map.get(pipe_s1_rej, 'autoreject'),
                's1_threshold_uv': pipe_s1_thresh if pipe_s1_rej == "Static threshold" else None,
                's1_adaptive_k': pipe_s1_k if pipe_s1_rej in ("Adaptive: IQR", "Adaptive: Z-score") else 3.0,
                # S2 trial rejection
                's2_trial_rejection': pipe_s2_trial_rej,
                's2_max_rt': pipe_s2_max_rt if (pipe_s2_trial_rej and pipe_s2_max_rt_en) else None,
                # CTP-BAD
                'bad_channels': pipe_bad_chs,
                'amplitude_method': pipe_amp_method,
                'p2p_tmax_negative': pipe_p2p_neg,
                'smoothing_method': pipe_smooth_method,
                'smoothing_lp_hz': pipe_smooth_lp,
                'smoothing_ma_ms': pipe_smooth_ma,
                'n_bootstrap': int(pipe_n_boot),
                'guilty_threshold': pipe_threshold,
                'target_stim': pipe_tgt_stim,
                'baseline_stims': pipe_bl_stims,
                'ar_n_jobs': int(pipe_ar_jobs),
            }

        # ===============================================================
        # Single file execution
        # ===============================================================
        if input_mode == "Single file (loaded)":
            st.markdown("---")
            if st.session_state.raw is None:
                st.info("Load a .fif file first.")
            elif st.button("▶️ Run Pipeline", type="primary",
                           key="pipe_run_single"):
                cfg = _build_cfg()
                try:
                    progress = st.progress(0, text="Starting pipeline…")
                    status = st.status("Running pipeline…", expanded=True)

                    status.write("**Step 1** — Filtering…")
                    progress.progress(10, text="Filtering…")
                    # Run full pipeline
                    result = run_pipeline(
                        st.session_state.raw,
                        st.session_state.events,
                        st.session_state.event_id,
                        cfg,
                    )
                    progress.progress(100, text="Done!")
                    status.update(
                        label="Pipeline complete!", state="complete",
                    )

                    # Store in session state for manual pages
                    st.session_state.raw_filtered = result['raw_filtered']
                    st.session_state.epochs = result['epochs']
                    st.session_state.individual_p300_window = (
                        result['individual_p300_window']
                    )
                    st.session_state.bad_results = result['bad_results']
                    st.session_state.pipeline_results = result

                    for msg in result['log']:
                        status.write(f"• {msg}")

                except Exception as exc:
                    st.error(f"Pipeline error: {exc}")
                    st.code(traceback.format_exc())

            # --- Display results ---
            if st.session_state.pipeline_results is not None:
                _display_pipeline_results(
                    st.session_state.pipeline_results,
                )

        # ===============================================================
        # Batch execution
        # ===============================================================
        elif input_mode == "Batch (upload multiple .fif)":
            st.markdown("---")
            if not batch_files:
                st.info("Upload one or more .fif files above.")
            elif st.button("\u25b6\ufe0f Run Batch Pipeline", type="primary",
                           key="pipe_run_batch"):
                cfg = _build_cfg()
                batch_results = []
                n_files = len(batch_files)
                progress = st.progress(0, text="Starting batch\u2026")

                for i, fif_file in enumerate(batch_files):
                    fname = fif_file.name
                    progress.progress(
                        int((i / n_files) * 100),
                        text=f"Processing {fname} ({i+1}/{n_files})\u2026",
                    )
                    try:
                        temp_path = Path(f"_tmp_batch_{i}.fif")
                        temp_path.write_bytes(fif_file.read())
                        raw, events, event_id = load_fif_file(
                            str(temp_path),
                        )
                        temp_path.unlink(missing_ok=True)

                        result = run_pipeline(
                            raw, events, event_id, cfg,
                        )
                        result['filename'] = fname
                        batch_results.append(result)
                    except Exception as exc:
                        batch_results.append({
                            'filename': fname,
                            'error': str(exc),
                        })

                progress.progress(100, text="Batch complete!")
                st.session_state['batch_pipeline_results'] = batch_results

            # --- Display batch results ---
            batch_res = st.session_state.get('batch_pipeline_results')
            if batch_res:
                _display_batch_results(batch_res)

        # ===============================================================
        # Grouped batch execution (innocent / guilty)
        # ===============================================================
        elif input_mode == "Batch with groups (innocent / guilty)":
            st.markdown("---")
            if not grouped_batch:
                st.info(
                    "Upload .fif files for at least one group above."
                )
            elif st.button(
                "\u25b6\ufe0f Run Grouped Batch Pipeline",
                type="primary",
                key="pipe_run_grouped",
            ):
                cfg = _build_cfg()
                batch_results = []
                all_files = [
                    (f, 'innocent')
                    for f in grouped_batch['innocent']
                ] + [
                    (f, 'guilty')
                    for f in grouped_batch['guilty']
                ]
                n_files = len(all_files)
                progress = st.progress(
                    0, text="Starting grouped batch\u2026",
                )

                for i, (fif_file, ground_truth) in enumerate(all_files):
                    fname = fif_file.name
                    progress.progress(
                        int((i / n_files) * 100),
                        text=(
                            f"[{ground_truth}] {fname} "
                            f"({i+1}/{n_files})\u2026"
                        ),
                    )
                    try:
                        temp_path = Path(f"_tmp_gbatch_{i}.fif")
                        temp_path.write_bytes(fif_file.read())
                        raw, events, event_id = load_fif_file(
                            str(temp_path),
                        )
                        temp_path.unlink(missing_ok=True)

                        result = run_pipeline(
                            raw, events, event_id, cfg,
                        )
                        result['filename'] = fname
                        result['ground_truth'] = ground_truth
                        batch_results.append(result)
                    except Exception as exc:
                        batch_results.append({
                            'filename': fname,
                            'ground_truth': ground_truth,
                            'error': str(exc),
                        })

                progress.progress(100, text="Grouped batch complete!")
                st.session_state['grouped_batch_results'] = batch_results

            # --- Display grouped batch results ---
            grouped_res = st.session_state.get(
                'grouped_batch_results',
            )
            if grouped_res:
                _display_grouped_batch_results(
                    grouped_res,
                    loocv_evaluation=st.session_state.get(
                        'pipe_grouped_loocv_eval', False,
                    ),
                    fbeta_weight=float(
                        st.session_state.get(
                            'pipe_grouped_loocv_fbeta', 1.0,
                        ),
                    ),
                )

    # ========================================================================
    # Page 8: Grid Search
    # ========================================================================

    elif page == "\U0001f50e Grid Search":
        st.header("\U0001f50e BAD Parameter Grid Search")
        st.markdown(
            "Upload **innocent** and **guilty** .fif batches, define "
            "parameter ranges, and run a **grid search** or "
            "**Optuna TPE** optimisation to find the preprocessing / "
            "BAD configuration that maximises **ROC-AUC**.  \n"
            "*Tip: parameter widgets below are isolated — changing "
            "them won't trigger a full page reload.*"
        )

        # --- File uploaders ---
        st.subheader("Data")
        gs_c1, gs_c2 = st.columns(2)
        with gs_c1:
            gs_innocent = st.file_uploader(
                "Innocent .fif files",
                type=['fif'],
                accept_multiple_files=True,
                key="gs_innocent_upload",
            )
        with gs_c2:
            gs_guilty = st.file_uploader(
                "Guilty .fif files",
                type=['fif'],
                accept_multiple_files=True,
                key="gs_guilty_upload",
            )

        if not (gs_innocent or gs_guilty):
            st.info("Upload .fif files for both groups to continue.")
        else:
            st.caption(
                f"{len(gs_innocent or [])} innocent / "
                f"{len(gs_guilty or [])} guilty file(s) selected."
            )

        # --- Search strategy ---
        st.markdown("---")
        _strategies = ["Grid Search"]
        if _OPTUNA_AVAILABLE:
            _strategies.append("Optuna (TPE)")
        gs_strategy = st.radio(
            "Search strategy",
            _strategies,
            horizontal=True,
            key="gs_strategy",
        )
        if not _OPTUNA_AVAILABLE and len(_strategies) == 1:
            st.caption(
                "Install `optuna` (`pip install optuna`) to enable "
                "Bayesian optimisation."
            )

        # --- Parameter grid ---
        st.markdown("---")
        st.subheader("Parameter Space")
        st.caption(
            "Enable parameters to search and set **discrete** candidate "
            "values (Grid Search uses full factorial; Optuna-TPE samples "
            "from the same finite lists). "
            "Type comma-separated values in the text fields for extras. "
            "Non-searched parameters use their defaults. "
            "Ranking is by **ROC-AUC** (threshold is swept internally)."
        )

        # Helper: merge multiselect + custom text
        def _merged_floats(ms_vals, txt, enabled):
            if not enabled:
                return []
            custom = _parse_custom_floats(txt)
            return sorted(set(ms_vals + custom))

        # ---- Filtering ----
        with st.expander("**Filtering**", expanded=True):
            gs_preset_en = False
            st.caption(
                "IIR Butterworth bandpass + notch 50 Hz is always "
                "applied. Tune HP/LP/order below."
            )

            gs_hp_en = st.checkbox(
                "Search high-pass cutoff", value=True,
                key="gs_hp_en",
            )
            _hp_c1, _hp_c2 = st.columns([2, 1])
            with _hp_c1:
                gs_hp_vals = st.multiselect(
                    "High-pass (Hz) presets",
                    [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
                    default=[0.1, 0.3, 0.5, 0.7],
                    key="gs_hp_vals",
                    disabled=not gs_hp_en,
                )
            with _hp_c2:
                gs_hp_custom = st.text_input(
                    "Custom HP values",
                    placeholder="e.g. 0.05, 0.8",
                    key="gs_hp_custom",
                    disabled=not gs_hp_en,
                )
            gs_hp_all = _merged_floats(
                gs_hp_vals, gs_hp_custom, gs_hp_en,
            )

            gs_lp_en = st.checkbox(
                "Search low-pass cutoff", value=True,
                key="gs_lp_en",
            )
            _lp_c1, _lp_c2 = st.columns([2, 1])
            with _lp_c1:
                gs_lp_vals = st.multiselect(
                    "Low-pass (Hz) presets",
                    [12.0, 15.0, 20.0, 25.0, 30.0, 40.0],
                    default=[12.0, 15.0, 20.0, 30.0],
                    key="gs_lp_vals",
                    disabled=not gs_lp_en,
                )
            with _lp_c2:
                gs_lp_custom = st.text_input(
                    "Custom LP values",
                    placeholder="e.g. 35, 45",
                    key="gs_lp_custom",
                    disabled=not gs_lp_en,
                )
            gs_lp_all = _merged_floats(
                gs_lp_vals, gs_lp_custom, gs_lp_en,
            )

            gs_order_en = st.checkbox(
                "Search IIR order (zero-phase Butterworth)",
                value=True,
                key="gs_order_en",
            )
            gs_order_vals = st.multiselect(
                "IIR order presets",
                [2, 3, 4, 6, 8],
                default=[2, 3, 4],
                key="gs_order_vals",
                disabled=not gs_order_en,
            )
            gs_order_all = sorted(gs_order_vals) if gs_order_en else []

        # ---- Epoching ----
        with st.expander("**Epoching**", expanded=True):
            gs_s1tmin_en = st.checkbox(
                "Search S1 tmin", value=True, key="gs_s1tmin_en",
            )
            _tm_c1, _tm_c2 = st.columns([2, 1])
            with _tm_c1:
                gs_s1tmin_vals = st.multiselect(
                    "S1 tmin (s) presets",
                    [-0.5, -0.3, -0.2, -0.1],
                    default=[-0.2],
                    key="gs_s1tmin_vals",
                    disabled=not gs_s1tmin_en,
                )
            with _tm_c2:
                gs_s1tmin_custom = st.text_input(
                    "Custom tmin",
                    placeholder="e.g. -0.15",
                    key="gs_s1tmin_custom",
                    disabled=not gs_s1tmin_en,
                )
            gs_s1tmin_all = _merged_floats(
                gs_s1tmin_vals, gs_s1tmin_custom, gs_s1tmin_en,
            )

            gs_s1tmax_en = st.checkbox(
                "Search S1 tmax", value=True, key="gs_s1tmax_en",
            )
            _tx_c1, _tx_c2 = st.columns([2, 1])
            with _tx_c1:
                gs_s1tmax_vals = st.multiselect(
                    "S1 tmax (s) presets",
                    [0.6, 0.8, 1.0, 1.2],
                    default=[1.0],
                    key="gs_s1tmax_vals",
                    disabled=not gs_s1tmax_en,
                )
            with _tx_c2:
                gs_s1tmax_custom = st.text_input(
                    "Custom tmax",
                    placeholder="e.g. 0.9",
                    key="gs_s1tmax_custom",
                    disabled=not gs_s1tmax_en,
                )
            gs_s1tmax_all = _merged_floats(
                gs_s1tmax_vals, gs_s1tmax_custom, gs_s1tmax_en,
            )

            gs_s1det_en = st.checkbox(
                "Search S1 detrend", value=True, key="gs_s1det_en",
            )
            gs_s1det_vals = st.multiselect(
                "S1 detrend",
                ["DC offset (0)", "Linear (1)"],
                default=["DC offset (0)"],
                key="gs_s1det_vals",
                disabled=not gs_s1det_en,
            )

            gs_rej_en = st.checkbox(
                "Search S1 rejection method", value=True,
                key="gs_rej_en",
            )
            _gs_rej_options = ["iqr", "zscore"]
            if _AUTOREJECT_AVAILABLE:
                _gs_rej_options.append("autoreject")
            _gs_rej_default = (
                ["autoreject", "iqr"]
                if _AUTOREJECT_AVAILABLE
                else ["iqr"]
            )
            gs_rej_vals = st.multiselect(
                "S1 rejection",
                _gs_rej_options,
                default=_gs_rej_default,
                key="gs_rej_vals",
                disabled=not gs_rej_en,
            )

            _adaptive_selected = bool(
                gs_rej_en
                and {"iqr", "zscore"} & set(gs_rej_vals)
            )
            gs_k_en = st.checkbox(
                "Search adaptive k (IQR / Z-score sensitivity)",
                value=True,
                key="gs_k_en",
                disabled=not _adaptive_selected,
                help=(
                    "Used for IQR/Z-score epoch rejection and baseline QC. "
                    "Grid Search and Optuna vary *k* only when rejection "
                    "is IQR or Z-score."
                ),
            )
            _k_c1, _k_c2 = st.columns([2, 1])
            with _k_c1:
                gs_k_vals = st.multiselect(
                    "k presets",
                    [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                    default=[1.5, 2.0, 2.5, 3.0],
                    key="gs_k_vals",
                    disabled=not (gs_k_en and _adaptive_selected),
                )
            with _k_c2:
                gs_k_custom = st.text_input(
                    "Custom k",
                    placeholder="e.g. 2.8",
                    key="gs_k_custom",
                    disabled=not (gs_k_en and _adaptive_selected),
                )
            gs_k_all = _merged_floats(
                gs_k_vals, gs_k_custom,
                gs_k_en and _adaptive_selected,
            )

        # ---- P300 window ----
        with st.expander("**P300 Window**", expanded=True):
            gs_p300mode_en = st.checkbox(
                "Search P300 window mode (individual vs static)",
                value=True, key="gs_p300mode_en",
            )
            gs_p300mode_vals = st.multiselect(
                "P300 window mode",
                ["individual", "static"],
                default=["individual", "static"],
                key="gs_p300mode_vals",
                disabled=not gs_p300mode_en,
            )

            st.markdown("**Individual window parameters** "
                        "(used when mode = individual)")
            gs_margin_en = st.checkbox(
                "Search window margin", value=True,
                key="gs_margin_en",
            )
            _mg_c1, _mg_c2 = st.columns([2, 1])
            with _mg_c1:
                gs_margin_vals = st.multiselect(
                    "Margin \u00b1 (s) presets",
                    [0.05, 0.10, 0.12, 0.15, 0.20, 0.25],
                    default=[0.10, 0.15, 0.20],
                    key="gs_margin_vals",
                    disabled=not gs_margin_en,
                )
            with _mg_c2:
                gs_margin_custom = st.text_input(
                    "Custom margins",
                    placeholder="e.g. 0.12",
                    key="gs_margin_custom",
                    disabled=not gs_margin_en,
                )
            gs_margin_all = _merged_floats(
                gs_margin_vals, gs_margin_custom, gs_margin_en,
            )

            gs_srch_min_en = st.checkbox(
                "Search peak search start", value=True,
                key="gs_srch_min_en",
            )
            _smi_c1, _smi_c2 = st.columns([2, 1])
            with _smi_c1:
                gs_srch_min_vals = st.multiselect(
                    "Peak search start (s) presets",
                    [0.15, 0.20, 0.25, 0.30],
                    default=[0.25, 0.30],
                    key="gs_srch_min_vals",
                    disabled=not gs_srch_min_en,
                )
            with _smi_c2:
                gs_srch_min_custom = st.text_input(
                    "Custom start",
                    placeholder="e.g. 0.18",
                    key="gs_srch_min_custom",
                    disabled=not gs_srch_min_en,
                )
            gs_srch_min_all = _merged_floats(
                gs_srch_min_vals, gs_srch_min_custom, gs_srch_min_en,
            )

            gs_srch_max_en = st.checkbox(
                "Search peak search end", value=True,
                key="gs_srch_max_en",
            )
            _smx_c1, _smx_c2 = st.columns([2, 1])
            with _smx_c1:
                gs_srch_max_vals = st.multiselect(
                    "Peak search end (s) presets",
                    [0.45, 0.50, 0.55, 0.60, 0.70, 0.80],
                    default=[0.60, 0.70],
                    key="gs_srch_max_vals",
                    disabled=not gs_srch_max_en,
                )
            with _smx_c2:
                gs_srch_max_custom = st.text_input(
                    "Custom end",
                    placeholder="e.g. 0.65",
                    key="gs_srch_max_custom",
                    disabled=not gs_srch_max_en,
                )
            gs_srch_max_all = _merged_floats(
                gs_srch_max_vals, gs_srch_max_custom, gs_srch_max_en,
            )

            gs_erp_lp_en = st.checkbox(
                "Search S2 ERP smoothing LP (peak detection)",
                value=True, key="gs_erp_lp_en",
            )
            gs_erp_lp_none = st.checkbox(
                "Include 'no smoothing' (None)",
                value=True,
                key="gs_erp_lp_none",
                disabled=not gs_erp_lp_en,
            )
            _elp_c1, _elp_c2 = st.columns([2, 1])
            with _elp_c1:
                gs_erp_lp_vals = st.multiselect(
                    "S2 ERP LP cutoff (Hz) presets",
                    [6.0, 8.0, 10.0, 12.0, 15.0, 20.0],
                    default=[6.0, 8.0, 10.0, 12.0],
                    key="gs_erp_lp_vals",
                    disabled=not gs_erp_lp_en,
                )
            with _elp_c2:
                gs_erp_lp_custom = st.text_input(
                    "Custom S2 ERP LP",
                    placeholder="e.g. 14",
                    key="gs_erp_lp_custom",
                    disabled=not gs_erp_lp_en,
                )
            gs_erp_lp_all = _merged_floats(
                gs_erp_lp_vals, gs_erp_lp_custom, gs_erp_lp_en,
            )
            if gs_erp_lp_en and gs_erp_lp_none:
                gs_erp_lp_all = [None] + gs_erp_lp_all

            st.markdown("**Static window parameters** "
                        "(used when mode = static)")
            gs_man_tmin_en = st.checkbox(
                "Search static window start", value=True,
                key="gs_man_tmin_en",
            )
            _mt_c1, _mt_c2 = st.columns([2, 1])
            with _mt_c1:
                gs_man_tmin_vals = st.multiselect(
                    "Static window start (s) presets",
                    [0.25, 0.30, 0.35, 0.40],
                    default=[0.25, 0.30],
                    key="gs_man_tmin_vals",
                    disabled=not gs_man_tmin_en,
                )
            with _mt_c2:
                gs_man_tmin_custom = st.text_input(
                    "Custom start",
                    placeholder="e.g. 0.28",
                    key="gs_man_tmin_custom",
                    disabled=not gs_man_tmin_en,
                )
            gs_man_tmin_all = _merged_floats(
                gs_man_tmin_vals, gs_man_tmin_custom, gs_man_tmin_en,
            )

            gs_man_tmax_en = st.checkbox(
                "Search static window end", value=True,
                key="gs_man_tmax_en",
            )
            _mx_c1, _mx_c2 = st.columns([2, 1])
            with _mx_c1:
                gs_man_tmax_vals = st.multiselect(
                    "Static window end (s) presets",
                    [0.50, 0.55, 0.60, 0.65, 0.70, 0.80],
                    default=[0.60, 0.70, 0.80],
                    key="gs_man_tmax_vals",
                    disabled=not gs_man_tmax_en,
                )
            with _mx_c2:
                gs_man_tmax_custom = st.text_input(
                    "Custom end",
                    placeholder="e.g. 0.65",
                    key="gs_man_tmax_custom",
                    disabled=not gs_man_tmax_en,
                )
            gs_man_tmax_all = _merged_floats(
                gs_man_tmax_vals, gs_man_tmax_custom, gs_man_tmax_en,
            )

        # ---- CTP-BAD ----
        with st.expander("**CTP-BAD**", expanded=True):
            gs_amp_en = st.checkbox(
                "Search amplitude method", value=True,
                key="gs_amp_en",
            )
            gs_amp_vals = st.multiselect(
                "Amplitude method",
                [
                    "Mean",
                    "Peak-to-peak (Rosenfeld)",
                    "Peak-to-Peak (Peak-Valley)",
                    "Baseline-to-peak",
                ],
                default=[
                    "Mean",
                    "Peak-to-peak (Rosenfeld)",
                    "Peak-to-Peak (Peak-Valley)",
                    "Baseline-to-peak",
                ],
                key="gs_amp_vals",
                disabled=not gs_amp_en,
            )

            gs_smooth_en = st.checkbox(
                "Search smoothing method", value=True,
                key="gs_smooth_en",
            )
            gs_smooth_vals = st.multiselect(
                "Smoothing",
                ["None", "Low-pass (Butterworth)"],
                default=["None", "Low-pass (Butterworth)"],
                key="gs_smooth_vals",
                disabled=not gs_smooth_en,
            )

            gs_smooth_hz_en = st.checkbox(
                "Search epoch smoothing LP cutoff", value=True,
                key="gs_smooth_hz_en",
            )
            _sh_c1, _sh_c2 = st.columns([2, 1])
            with _sh_c1:
                gs_smooth_hz_vals = st.multiselect(
                    "Epoch smooth LP (Hz) presets",
                    [6.0, 8.0, 10.0, 12.0, 15.0],
                    default=[6.0, 8.0, 10.0, 12.0],
                    key="gs_smooth_hz_vals",
                    disabled=not gs_smooth_hz_en,
                )
            with _sh_c2:
                gs_smooth_hz_custom = st.text_input(
                    "Custom smooth LP",
                    placeholder="e.g. 9",
                    key="gs_smooth_hz_custom",
                    disabled=not gs_smooth_hz_en,
                )
            gs_smooth_hz_all = _merged_floats(
                gs_smooth_hz_vals, gs_smooth_hz_custom,
                gs_smooth_hz_en,
            )

        # --- Global settings ---
        st.markdown("---")
        _gs_c1, _gs_c2 = st.columns(2)
        with _gs_c1:
            gs_n_boot = st.number_input(
                "Bootstrap iterations (lower = faster search)",
                100, 100000, 1000, 100,
                key="gs_n_boot",
            )
            if gs_strategy == "Optuna (TPE)":
                gs_n_trials = st.number_input(
                    "Optuna trials", 10, 5000, 50, 10,
                    key="gs_n_trials",
                )
        with _gs_c2:
            _cpu_count = _os.cpu_count() or 4
            gs_max_workers = st.slider(
                "Parallel file workers",
                1, min(_cpu_count, 8), min(4, _cpu_count),
                key="gs_max_workers",
                help=(
                    "Number of threads for processing files in "
                    "parallel within each trial. MNE releases the "
                    "GIL for NumPy/SciPy, so threads give real "
                    "speedup. Set to 1 to disable parallelism."
                ),
            )

        gs_loocv = st.checkbox(
            "Use Leave-One-Out CV for accuracy metrics",
            value=True,
            key="gs_loocv",
            help=(
                "When enabled, the accuracy / sensitivity / specificity "
                "are computed via LOOCV: threshold is selected on N-1 "
                "subjects (using F-beta) and tested on the held-out one. "
                "AUC is always threshold-independent."
            ),
        )
        gs_s2_mirror_s1 = st.checkbox(
            "Mirror S2 epoch preprocessing to S1",
            value=True,
            key="gs_s2_mirror_s1",
            help=(
                "Copy **S1** epoch settings into **S2** target epochs used "
                "for the individual P300 window: ``s1_tmin/tmax``, "
                "baseline on/off, and detrend label. Recommended when "
                "Optuna/Grid Search varies S1 epoch hyperparameters so "
                "behavioural S2 ERP matches the same preprocessing."
            ),
        )
        gs_fbeta = st.number_input(
            "F-beta weight (β) for threshold selection",
            min_value=0.1, max_value=5.0, value=1.0, step=0.1,
            key="gs_fbeta",
            help=(
                "β controls the balance between precision and recall "
                "when choosing the optimal threshold.  \n"
                "**β = 1** → balanced F1.  \n"
                "**β > 1** (e.g. 2) → prioritises sensitivity "
                "(detecting guilty).  \n"
                "**β < 1** (e.g. 0.5) → prioritises specificity "
                "(protecting innocent)."
            ),
        )

        # ===============================================================
        # Assemble grid / Optuna param defs
        # ===============================================================
        grid_ranges = {}

        if gs_preset_en:
            grid_ranges['filter_preset'] = ['skip', 'aggressive']

        if gs_hp_en and gs_hp_all:
            grid_ranges['hp_cutoff'] = gs_hp_all

        if gs_lp_en and gs_lp_all:
            grid_ranges['lp_cutoff'] = gs_lp_all

        if gs_order_en and gs_order_all:
            grid_ranges['iir_order'] = gs_order_all

        if gs_s1tmin_en and gs_s1tmin_all:
            grid_ranges['s1_tmin'] = gs_s1tmin_all

        if gs_s1tmax_en and gs_s1tmax_all:
            grid_ranges['s1_tmax'] = gs_s1tmax_all

        if gs_s1det_en and gs_s1det_vals:
            grid_ranges['s1_detrend'] = gs_s1det_vals

        if gs_rej_en and gs_rej_vals:
            grid_ranges['s1_rejection'] = gs_rej_vals

        if gs_k_en and gs_k_all:
            grid_ranges['s1_adaptive_k'] = gs_k_all

        if gs_p300mode_en and gs_p300mode_vals:
            _mode_bools = []
            if "individual" in gs_p300mode_vals:
                _mode_bools.append(True)
            if "static" in gs_p300mode_vals:
                _mode_bools.append(False)
            if len(_mode_bools) > 1:
                grid_ranges['use_individual_window'] = _mode_bools

        if gs_margin_en and gs_margin_all:
            grid_ranges['window_margin'] = gs_margin_all

        if gs_srch_min_en and gs_srch_min_all:
            grid_ranges['peak_search_tmin'] = gs_srch_min_all

        if gs_srch_max_en and gs_srch_max_all:
            grid_ranges['peak_search_tmax'] = gs_srch_max_all

        if gs_erp_lp_en and gs_erp_lp_all:
            grid_ranges['s2_erp_lowpass_hz'] = gs_erp_lp_all

        if gs_man_tmin_en and gs_man_tmin_all:
            grid_ranges['manual_tmin'] = gs_man_tmin_all

        if gs_man_tmax_en and gs_man_tmax_all:
            grid_ranges['manual_tmax'] = gs_man_tmax_all

        if gs_amp_en and gs_amp_vals:
            grid_ranges['amplitude_method'] = gs_amp_vals

        if gs_smooth_en and gs_smooth_vals:
            grid_ranges['smoothing_method'] = gs_smooth_vals

        if gs_smooth_hz_en and gs_smooth_hz_all:
            grid_ranges['smoothing_lp_hz'] = gs_smooth_hz_all

        optuna_params = []
        optuna_adaptive_k_float_spec = None
        if gs_strategy == "Optuna (TPE)":
            if gs_preset_en:
                optuna_params.append(
                    ('filter_preset', 'cat', None, None,
                     ['skip', 'aggressive']),
                )
            if gs_hp_en and gs_hp_all:
                optuna_params.append(
                    ('hp_cutoff', 'cat', None, None,
                     sorted(set(gs_hp_all))),
                )
            if gs_lp_en:
                optuna_params.append(
                    ('lp_cutoff', 'int', 12, 30, 3),
                )
            if gs_order_en and gs_order_all:
                optuna_params.append(
                    ('iir_order', 'cat', None, None, list(gs_order_all)),
                )
            if gs_s1tmin_en and gs_s1tmin_all:
                optuna_params.append(
                    ('s1_tmin', 'cat', None, None,
                     sorted(set(gs_s1tmin_all))),
                )
            if gs_s1tmax_en and gs_s1tmax_all:
                optuna_params.append(
                    ('s1_tmax', 'cat', None, None,
                     sorted(set(gs_s1tmax_all))),
                )
            if gs_s1det_en and gs_s1det_vals:
                optuna_params.append(
                    ('s1_detrend', 'cat', None, None, list(gs_s1det_vals)),
                )
            if gs_rej_en and gs_rej_vals:
                optuna_params.append(
                    ('s1_rejection', 'cat', None, None, list(gs_rej_vals)),
                )
            if gs_k_en and gs_k_all:
                optuna_adaptive_k_float_spec = (1.5, 3.0, 0.5)

            if gs_p300mode_en and gs_p300mode_vals:
                _mode_bools_o = []
                if "individual" in gs_p300mode_vals:
                    _mode_bools_o.append(True)
                if "static" in gs_p300mode_vals:
                    _mode_bools_o.append(False)
                if len(_mode_bools_o) > 1:
                    optuna_params.append(
                        ('use_individual_window', 'cat', None, None,
                         _mode_bools_o),
                    )

            if gs_margin_en:
                optuna_params.append(
                    ('window_margin', 'float', 0.10, 0.20, 0.05),
                )
            if gs_srch_min_en:
                optuna_params.append(
                    ('peak_search_tmin', 'float', 0.25, 0.35, 0.05),
                )
            if gs_srch_max_en:
                optuna_params.append(
                    ('peak_search_tmax', 'float', 0.60, 0.75, 0.05),
                )
            if gs_erp_lp_en and gs_erp_lp_all:
                optuna_params.append(
                    ('s2_erp_lowpass_hz', 'cat', None, None,
                     list(gs_erp_lp_all)),
                )
            if gs_man_tmin_en:
                optuna_params.append(
                    ('manual_tmin', 'float', 0.25, 0.35, 0.05),
                )
            if gs_man_tmax_en:
                optuna_params.append(
                    ('manual_tmax', 'float', 0.60, 0.80, 0.05),
                )

            if gs_amp_en and gs_amp_vals:
                optuna_params.append(
                    ('amplitude_method', 'cat', None, None,
                     list(gs_amp_vals)),
                )
            if gs_smooth_en and gs_smooth_vals:
                optuna_params.append(
                    ('smoothing_method', 'cat', None, None,
                     list(gs_smooth_vals)),
                )
            if gs_smooth_hz_en:
                optuna_params.append(
                    ('smoothing_lp_hz', 'float', 6.0, 12.0, 2.0),
                )

        has_params = bool(grid_ranges) or (
            gs_strategy == "Optuna (TPE)" and bool(optuna_params)
        )
        if gs_strategy == "Grid Search":
            n_combos = (
                _grid_space_size(grid_ranges) if has_params else 0
            )
        else:
            n_combos = gs_n_trials if has_params else 0

        n_files = len(gs_innocent or []) + len(gs_guilty or [])
        n_runs = n_combos * n_files

        st.markdown("---")
        if gs_strategy == "Grid Search":
            st.markdown(
                f"**Grid size:** {n_combos} combinations "
                f"\u00d7 {n_files} files = **{n_runs} pipeline "
                f"runs**"
            )
            if n_combos > 500_000:
                st.error(
                    f"Grid has **{n_combos:,}** combinations — this "
                    f"would exhaust all RAM. Reduce candidate values "
                    f"or switch to Optuna."
                )
            elif n_combos > 200:
                st.warning(
                    "Large grid! Consider reducing candidate values "
                    "or switching to Optuna."
                )
        else:
            st.markdown(
                f"**Optuna:** {n_combos} trials "
                f"\u00d7 {n_files} files = **{n_runs} pipeline "
                f"runs**"
            )

        # ===============================================================
        # Optuna Resume
        # ===============================================================
        _resume_study = None
        if gs_strategy == "Optuna (TPE)":
            with st.expander("Resume from saved Optuna study", expanded=False):
                uploaded_study = st.file_uploader(
                    "Upload .pkl study file",
                    type=["pkl"],
                    key="gs_upload_study",
                )
                if uploaded_study is not None:
                    try:
                        _resume_data = pickle.loads(uploaded_study.read())
                        _resume_study = _resume_data['study']
                        _prev_params = _resume_data.get('param_defs', [])
                        _prev_n = len(_resume_study.trials)
                        st.success(
                            f"Loaded study with {_prev_n} completed "
                            f"trials (best AUC: "
                            f"{_resume_study.best_value:.4f})."
                        )
                        _prev_keys = sorted(
                            k for k, *_ in _prev_params
                        )
                        _curr_keys = sorted(
                            k for k, *_ in optuna_params
                        )
                        if _prev_keys != _curr_keys:
                            st.warning(
                                f"Parameter mismatch! Saved: "
                                f"{_prev_keys}, current: "
                                f"{_curr_keys}. The study will "
                                f"resume with current parameters "
                                f"but prior trials keep their "
                                f"original values."
                            )
                        st.info(
                            f"Click **Run Optuna Search** to add "
                            f"{gs_n_trials} new trials on top of "
                            f"the {_prev_n} existing ones."
                        )
                    except Exception as exc:
                        st.error(f"Failed to load study: {exc}")

        # ===============================================================
        # Run / Stop
        # ===============================================================
        _grid_too_large = (
            gs_strategy == "Grid Search" and n_combos > 500_000
        )
        can_run = (
            has_params
            and (gs_innocent or gs_guilty)
            and not _grid_too_large
        )

        run_col, stop_col = st.columns([2, 1])
        with run_col:
            run_clicked = st.button(
                (
                    "\u25b6\ufe0f Run Grid Search"
                    if gs_strategy == "Grid Search"
                    else "\u25b6\ufe0f Run Optuna Search"
                ),
                type="primary",
                key="gs_run",
                disabled=not can_run,
            )
        with stop_col:
            def _set_stop():
                st.session_state['_gs_stop'] = True

            st.button(
                "\u23f9\ufe0f Stop after current combo",
                key="gs_stop",
                on_click=_set_stop,
                help=(
                    "Sets a stop flag checked between iterations. "
                    "Partial results will be preserved."
                ),
            )

        if run_clicked:
            st.session_state['_gs_stop'] = False
            st.session_state.pop('_gs_partial', None)
            st.session_state.pop('grid_search_results', None)
            st.session_state.pop('_optuna_study', None)
            st.session_state.pop('_gs_optuna_show_hist', None)

            progress = st.progress(0, text="Loading files\u2026")
            status = st.empty()

            innocent_data = []
            for i, f in enumerate(gs_innocent or []):
                status.text(f"Loading innocent {f.name}\u2026")
                tmp = Path(f"_tmp_gs_i{i}.fif")
                tmp.write_bytes(f.read())
                raw, events, event_id = load_fif_file(str(tmp))
                tmp.unlink(missing_ok=True)
                innocent_data.append(
                    (raw, events, event_id, f.name),
                )

            guilty_data = []
            for i, f in enumerate(gs_guilty or []):
                status.text(f"Loading guilty {f.name}\u2026")
                tmp = Path(f"_tmp_gs_g{i}.fif")
                tmp.write_bytes(f.read())
                raw, events, event_id = load_fif_file(str(tmp))
                tmp.unlink(missing_ok=True)
                guilty_data.append(
                    (raw, events, event_id, f.name),
                )

            base_cfg = {
                'filter_preset': 'aggressive',
                'notch_freqs': [50],
                'hp_cutoff': 0.7,
                'lp_cutoff': 30.0,
                'filter_method': 'iir',
                'iir_order': 4,
                's2_tmin': -0.2,
                's2_tmax': 0.8,
                's2_baseline': True,
                's2_detrend': 'DC offset (0)',
                's2_rejection': (
                    'autoreject' if _AUTOREJECT_AVAILABLE
                    else 'iqr'
                ),
                's2_threshold_uv': None,
                'use_individual_window': True,
                'peak_channels': ['Pz'],
                'peak_search_tmin': 0.25,
                'peak_search_tmax': 0.60,
                'window_margin': 0.15,
                's2_erp_lowpass_hz': 12.0,
                'manual_tmin': 0.3,
                'manual_tmax': 0.6,
                's1_tmin': -0.2,
                's1_tmax': 0.8,
                's1_baseline': True,
                's1_detrend': 'DC offset (0)',
                's1_rejection': (
                    'autoreject' if _AUTOREJECT_AVAILABLE
                    else 'iqr'
                ),
                's1_threshold_uv': None,
                's1_adaptive_k': 3.0,
                's2_trial_rejection': True,
                's2_max_rt': 1.0,
                'bad_channels': ['Pz'],
                'amplitude_method': 'Peak-to-Peak (Peak-Valley)',
                'p2p_tmax_negative': 0.9,
                'smoothing_method': 'Low-pass (Butterworth)',
                'smoothing_lp_hz': 12.0,
                'smoothing_ma_ms': 100.0,
                'n_bootstrap': int(gs_n_boot),
                'guilty_threshold': 0.90,
                'target_stim': None,
                'baseline_stims': None,
                's2_match_s1_preprocessing': gs_s2_mirror_s1,
                'ar_n_jobs': 1,
            }

            if gs_strategy == "Grid Search":
                grid_space = _build_grid_space(grid_ranges)
                gs_results = _run_grid_search(
                    innocent_data, guilty_data,
                    base_cfg, grid_space,
                    progress, status,
                    use_loocv=gs_loocv,
                    fbeta_weight=gs_fbeta,
                    max_workers=gs_max_workers,
                )
            else:
                gs_results = _run_optuna_search(
                    innocent_data, guilty_data,
                    base_cfg, optuna_params,
                    gs_n_trials,
                    progress, status,
                    use_loocv=gs_loocv,
                    fbeta_weight=gs_fbeta,
                    study=_resume_study,
                    max_workers=gs_max_workers,
                    adaptive_k_float_spec=optuna_adaptive_k_float_spec,
                )
                st.session_state['_optuna_param_defs'] = optuna_params
                st.session_state['_optuna_base_cfg'] = base_cfg
            st.session_state['grid_search_results'] = gs_results
            st.session_state['grid_search_loocv'] = gs_loocv

        # --- Optuna study download ---
        _saved_study = st.session_state.get('_optuna_study')
        if _saved_study is not None:
            _study_payload = {
                'study': _saved_study,
                'param_defs': st.session_state.get(
                    '_optuna_param_defs', [],
                ),
                'base_cfg': st.session_state.get(
                    '_optuna_base_cfg', {},
                ),
            }
            _pkl_buf = pickle.dumps(_study_payload)
            _n_done = len(_saved_study.trials)
            st.download_button(
                f"Download Optuna study ({_n_done} trials)",
                data=_pkl_buf,
                file_name="optuna_study.pkl",
                mime="application/octet-stream",
                key="gs_download_study",
            )

            st.markdown("---")
            st.subheader("Optuna visualization")
            st.caption(
                "Built-in Optuna visualization (Plotly): objective vs trial "
                "from plot_optimization_history (objective = ROC-AUC here)."
            )
            if _OPTUNA_VIZ_AVAILABLE and plot_optimization_history is not None:
                _ov_c1, _ov_c2 = st.columns(2)
                with _ov_c1:
                    if st.button(
                        "Generate optimization history plot",
                        key="gs_btn_optuna_optimization_history",
                        help=(
                            "Interactive Plotly figure from "
                            "optuna.visualization.plot_optimization_history."
                        ),
                    ):
                        st.session_state['_gs_optuna_show_hist'] = True
                with _ov_c2:
                    if st.button(
                        "Hide optimization history plot",
                        key="gs_hide_optuna_hist",
                    ):
                        st.session_state['_gs_optuna_show_hist'] = False

                if st.session_state.get('_gs_optuna_show_hist'):
                    try:
                        _hist_fig = plot_optimization_history(_saved_study)
                        st.plotly_chart(
                            _hist_fig,
                            use_container_width=True,
                            key="gs_plotly_optuna_history",
                        )
                    except Exception as _hist_exc:
                        st.session_state['_gs_optuna_show_hist'] = False
                        st.error(
                            "Could not render optimization history: "
                            f"{_hist_exc}"
                        )
            else:
                st.info(
                    "Install **plotly** to enable Optuna charts "
                    "(e.g. `pip install plotly`)."
                )

        gs_res = st.session_state.get('grid_search_results')
        if gs_res:
            _display_grid_search_results(
                gs_res,
                use_loocv=st.session_state.get(
                    'grid_search_loocv', False,
                ),
            )

    # ========================================================================
    # Page 9: Export Results
    # ========================================================================
    
    elif page == "\U0001f4c9 Export Results":
        st.header("\U0001f4c9 Export Results")
        
        if st.session_state.raw is None:
            st.warning("⚠️ No data loaded")
            return
        
        st.markdown("""
        Export your analysis results and visualizations.
        """)
        
        # Export options
        st.subheader("Available Exports")
        
        # 1. Raw data info
        if st.button("Export Data Summary (CSV)"):
            raw = st.session_state.raw
            
            summary_data = {
                'Parameter': [
                    'Channels',
                    'Sampling Rate (Hz)',
                    'Duration (s)',
                    'Samples',
                    'Annotations'
                ],
                'Value': [
                    ', '.join(raw.ch_names),
                    raw.info['sfreq'],
                    raw.times[-1],
                    len(raw.times),
                    len(raw.annotations) if raw.annotations else 0
                ]
            }
            
            df_summary = pd.DataFrame(summary_data)
            csv = df_summary.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="eeg_summary.csv",
                mime="text/csv"
            )
        
        # 2. Signal quality
        if st.button("Export Signal Quality (CSV)"):
            raw = st.session_state.raw
            quality_df = check_signal_quality(raw)
            csv = quality_df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="signal_quality.csv",
                mime="text/csv"
            )
        
        # 3. P300 results
        if (st.session_state.probe_erp is not None and 
            st.session_state.irrelevant_erp is not None):
            
            if st.button("Export P300 Analysis (CSV)"):
                p300_df = analyze_p300(
                    st.session_state.probe_erp,
                    st.session_state.irrelevant_erp
                )
                csv = p300_df.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="p300_analysis.csv",
                    mime="text/csv"
                )
        
        # 3b. CTP-BAD results
        if 'bad_results' in st.session_state and st.session_state.bad_results is not None:
            if st.button("Export CTP-BAD Results (CSV)"):
                bad_results = st.session_state.bad_results
                
                # Create comprehensive results dataframe
                results_data = {
                    'Classification': [bad_results['overall_classification']],
                    'p_value': [bad_results.get('p_value', 1 - bad_results['max_proportion'])],
                    'Max_Proportion': [bad_results['max_proportion']],
                    'Max_Channel': [bad_results['max_channel']],
                    'Threshold': [bad_results['threshold']],
                    'N_Bootstrap': [bad_results['n_bootstrap']],
                    'N_Probe_Epochs': [bad_results['n_probe_epochs']],
                    'N_Irrelevant_Epochs': [bad_results['n_irrelevant_epochs']],
                    'Verdict': [bad_results['verdict']]
                }
                
                # Combine with per-channel results
                summary_df = pd.DataFrame(results_data)
                channel_df = bad_results['channel_results']
                
                # Export both
                csv_buffer = io.StringIO()
                csv_buffer.write("# CTP-BAD Bootstrap Analysis Results\n")
                csv_buffer.write(f"# Generated: {pd.Timestamp.now()}\n")
                csv_buffer.write("\n# Overall Classification\n")
                summary_df.to_csv(csv_buffer, index=False)
                csv_buffer.write("\n# Per-Channel Results\n")
                channel_df.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name="ctp_bad_results.csv",
                    mime="text/csv"
                )
        
        # 4. ERP plot
        if (st.session_state.probe_erp is not None and 
            st.session_state.irrelevant_erp is not None):
            
            if st.button("Export ERP Plot (PNG)"):
                fig = plot_erps(
                    st.session_state.probe_erp,
                    st.session_state.irrelevant_erp
                )
                
                # Save to bytes buffer
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                
                st.download_button(
                    label="Download PNG",
                    data=buf,
                    file_name="erp_analysis.png",
                    mime="image/png"
                )
                
                plt.close()
        
        # 5. Export epochs data
        if st.session_state.epochs is not None:
            if st.button("Export Epoch Data (NPY)"):
                epochs = st.session_state.epochs
                data = epochs.get_data()
                
                # Save to bytes buffer
                buf = io.BytesIO()
                np.save(buf, data)
                buf.seek(0)
                
                st.download_button(
                    label="Download NPY",
                    data=buf,
                    file_name="epochs_data.npy",
                    mime="application/octet-stream"
                )
        
        # Full report
        st.subheader("Generate Full Report")
        
        if st.button("Generate HTML Report"):
            st.info("🚧 HTML report generation coming soon!")


if __name__ == '__main__':
    main()
