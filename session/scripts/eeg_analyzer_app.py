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

"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import io
import traceback

try:
    import mne
    mne.set_log_level('WARNING')
except ImportError:
    st.error("MNE-Python not installed! Install with: pip install mne")
    st.stop()


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


def apply_filters(raw, notch_freqs=None, lowcut=None, highcut=None):
    """
    Apply filters to raw EEG data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    notch_freqs : list or None
        Frequencies for notch filter (Hz)
    lowcut : float or None
        High-pass filter cutoff (Hz)
    highcut : float or None
        Low-pass filter cutoff (Hz)
        
    Returns
    -------
    raw_filtered : mne.io.Raw
        Filtered EEG data
    """
    raw_filtered = raw.copy()
    
    # Apply notch filter
    if notch_freqs:
        raw_filtered.notch_filter(notch_freqs, verbose='ERROR')
    
    # Apply bandpass filter
    if lowcut is not None and highcut is not None:
        raw_filtered.filter(lowcut, highcut, verbose='ERROR')
    elif lowcut is not None:
        raw_filtered.filter(lowcut, None, verbose='ERROR')
    elif highcut is not None:
        raw_filtered.filter(None, highcut, verbose='ERROR')
    
    return raw_filtered


def create_epochs(raw, events, event_id, tmin, tmax, baseline, reject_threshold_uv):
    """
    Create epochs around stimulus events.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    events : np.ndarray
        Events array
    event_id : dict
        Event ID mapping
    tmin : float
        Start time (seconds)
    tmax : float
        End time (seconds)
    baseline : tuple or None
        Baseline period
    reject_threshold_uv : float or None
        Artifact rejection threshold (µV)
        
    Returns
    -------
    epochs : mne.Epochs
        Epoched data
    """
    # Filter S1 onset events
    s1_event_id = {
        k: v for k, v in event_id.items()
        if 'S1_onset' in k or 's1_onset' in k.lower()
    }
    
    if not s1_event_id:
        # Fallback to all events
        s1_event_id = event_id
    
    # Set up rejection
    if reject_threshold_uv is not None:
        reject = dict(eeg=reject_threshold_uv * 1e-6)  # Convert µV to V
    else:
        reject = None
    
    # Create epochs
    epochs = mne.Epochs(
        raw,
        events,
        event_id=s1_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        reject=reject,
        verbose='ERROR'
    )
    
    return epochs


def compute_erps(epochs):
    """
    Compute ERPs for probe and irrelevant conditions.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data
        
    Returns
    -------
    probe_erp : mne.Evoked or None
        Probe ERP
    irrelevant_erp : mne.Evoked or None
        Irrelevant ERP
    """
    available_events = list(epochs.event_id.keys())
    
    # Find probe and irrelevant events
    probe_events = [k for k in available_events 
                   if 'probe' in k.lower() and 's1_onset' in k.lower()]
    irrelevant_events = [k for k in available_events 
                        if 'irrelevant' in k.lower() and 's1_onset' in k.lower()]
    
    probe_erp = None
    irrelevant_erp = None
    
    if probe_events:
        probe_epochs = epochs[probe_events]
        if len(probe_epochs) > 0:
            probe_erp = probe_epochs.average()
    
    if irrelevant_events:
        irrelevant_epochs = epochs[irrelevant_events]
        if len(irrelevant_epochs) > 0:
            irrelevant_erp = irrelevant_epochs.average()
    
    return probe_erp, irrelevant_erp, probe_events, irrelevant_events


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


def ctp_bad_analysis(epochs, tmin=0.3, tmax=0.6, n_bootstrap=1000, threshold=0.90):
    """
    Perform CTP-BAD (Bootstrap Amplitude Difference) analysis.
    
    This method uses bootstrap resampling to determine if a participant
    shows recognition of the probe stimulus (guilty) or not (innocent).
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data containing probe and irrelevant trials
    tmin : float
        Analysis window start (seconds)
    tmax : float
        Analysis window end (seconds)
    n_bootstrap : int
        Number of bootstrap iterations (default: 1000)
    threshold : float
        Classification threshold (default: 0.90 = 90%)
        
    Returns
    -------
    dict
        Results containing:
        - channel_results: per-channel bootstrap proportions and classifications
        - overall_classification: final verdict (guilty/innocent)
        - bootstrap_proportions: proportion for each channel
    """
    # Get epoch data
    available_events = list(epochs.event_id.keys())
    probe_events = [k for k in available_events 
                   if 'probe' in k.lower() and 's1_onset' in k.lower()]
    irrelevant_events = [k for k in available_events 
                        if 'irrelevant' in k.lower() and 's1_onset' in k.lower()]
    
    if not probe_events or not irrelevant_events:
        raise ValueError("Could not find probe/irrelevant events in epochs")
    
    # Extract probe and irrelevant epochs
    probe_epochs = epochs[probe_events]
    irrelevant_epochs = epochs[irrelevant_events]
    
    # Get time mask for analysis window
    times = epochs.times
    time_mask = (times >= tmin) & (times <= tmax)
    
    # Get epoch data (n_epochs, n_channels, n_times)
    probe_data = probe_epochs.get_data() * 1e6  # Convert to µV
    irrelevant_data = irrelevant_epochs.get_data() * 1e6
    
    # Extract data in analysis window and compute mean amplitude per epoch
    # probe_amplitudes: (n_probe_epochs, n_channels)
    probe_amplitudes = probe_data[:, :, time_mask].mean(axis=2)
    irrelevant_amplitudes = irrelevant_data[:, :, time_mask].mean(axis=2)
    
    n_probe = probe_amplitudes.shape[0]
    n_irrelevant = irrelevant_amplitudes.shape[0]
    n_channels = probe_amplitudes.shape[1]
    
    # Bootstrap analysis per channel
    channel_results = []
    bootstrap_proportions = []
    
    for ch_idx, ch_name in enumerate(epochs.ch_names):
        # Get amplitudes for this channel
        probe_ch = probe_amplitudes[:, ch_idx]  # (n_probe,)
        irrelevant_ch = irrelevant_amplitudes[:, ch_idx]  # (n_irrelevant,)
        
        # Bootstrap resampling
        count_probe_greater = 0
        
        for _ in range(n_bootstrap):
            # Randomly sample with replacement
            probe_sample = np.random.choice(probe_ch, size=n_probe, replace=True)
            irrelevant_sample = np.random.choice(irrelevant_ch, size=n_irrelevant, replace=True)
            
            # Compute means
            probe_mean = probe_sample.mean()
            irrelevant_mean = irrelevant_sample.mean()
            
            # Check if probe > irrelevant
            if probe_mean > irrelevant_mean:
                count_probe_greater += 1
        
        # Calculate proportion
        proportion = count_probe_greater / n_bootstrap
        bootstrap_proportions.append(proportion)
        
        # Classify
        if proportion >= threshold:
            classification = "Guilty"
            status = "🔴"
            confidence = "High" if proportion >= 0.95 else "Moderate"
        else:
            classification = "Innocent"
            status = "🟢"
            if proportion <= 0.60:
                confidence = "High"
            elif proportion <= 0.75:
                confidence = "Moderate"
            else:
                confidence = "Low"
        
        channel_results.append({
            'Channel': ch_name,
            'Status': status,
            'Bootstrap %': f"{proportion * 100:.1f}%",
            'Classification': classification,
            'Confidence': confidence,
            'Probe > Irr': f"{count_probe_greater}/{n_bootstrap}"
        })
    
    # Overall classification (majority vote or max proportion)
    max_proportion = max(bootstrap_proportions)
    max_channel_idx = bootstrap_proportions.index(max_proportion)
    max_channel = epochs.ch_names[max_channel_idx]
    
    if max_proportion >= threshold:
        overall_classification = "GUILTY"
        overall_status = "🔴"
        verdict = f"Participant likely recognized the probe stimulus (max: {max_proportion*100:.1f}% at {max_channel})"
    else:
        overall_classification = "INNOCENT"
        overall_status = "🟢"
        verdict = f"No clear recognition of probe stimulus (max: {max_proportion*100:.1f}% at {max_channel})"
    
    return {
        'channel_results': pd.DataFrame(channel_results),
        'overall_classification': overall_classification,
        'overall_status': overall_status,
        'verdict': verdict,
        'bootstrap_proportions': bootstrap_proportions,
        'max_proportion': max_proportion,
        'max_channel': max_channel,
        'n_probe_epochs': n_probe,
        'n_irrelevant_epochs': n_irrelevant,
        'n_bootstrap': n_bootstrap,
        'threshold': threshold
    }


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


def plot_erps(probe_erp, irrelevant_erp, p300_window=(0.3, 0.6)):
    """Plot ERPs with P300 window."""
    fig, axes = plt.subplots(1, len(probe_erp.ch_names), 
                            figsize=(5*len(probe_erp.ch_names), 4))
    
    if len(probe_erp.ch_names) == 1:
        axes = [axes]
    
    times = probe_erp.times * 1000  # Convert to ms
    
    for idx, (ch_name, ax) in enumerate(zip(probe_erp.ch_names, axes)):
        # Get data in µV
        probe_data = probe_erp.data[idx] * 1e6
        irrelevant_data = irrelevant_erp.data[idx] * 1e6
        diff_data = probe_data - irrelevant_data
        
        # Plot ERPs
        ax.plot(times, probe_data, label='Probe', color='#d62728', 
                linewidth=2, alpha=0.8)
        ax.plot(times, irrelevant_data, label='Irrelevant', color='#1f77b4', 
                linewidth=2, alpha=0.8)
        ax.plot(times, diff_data, label='Difference', color='#2ca02c', 
                linewidth=2, linestyle='--', alpha=0.8)
        
        # Reference lines
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(0, color='black', linestyle='--', linewidth=1, 
                  label='Stimulus')
        
        # P300 window
        ax.axvspan(p300_window[0]*1000, p300_window[1]*1000, 
                  alpha=0.1, color='gray', label='P300 window')
        
        # Styling
        ax.set_xlabel('Time (ms)', fontsize=11)
        ax.set_ylabel('Amplitude (µV)', fontsize=11)
        ax.set_title(f'{ch_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Event-Related Potentials (ERPs)', 
                fontsize=14, fontweight='bold')
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
         "📈 Epoching", "🎯 ERP Analysis", "📉 Export Results"]
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
                        st.dataframe(df_events, use_container_width=True)
                    
                    # First few annotations
                    with st.expander("View First 10 Annotations"):
                        annotations_data = []
                        for i in range(min(10, len(raw.annotations))):
                            annotations_data.append({
                                'Time (s)': f"{raw.annotations.onset[i]:.3f}",
                                'Description': raw.annotations.description[i]
                            })
                        st.dataframe(pd.DataFrame(annotations_data), 
                                   use_container_width=True)
                
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
        st.dataframe(quality_df, use_container_width=True)
        
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
        
        st.markdown("""
        Apply filters to remove noise and artifacts from EEG data.
        """)
        
        # Filter settings
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Notch Filter")
            apply_notch = st.checkbox("Apply notch filter", value=True)
            
            if apply_notch:
                notch_freqs = st.multiselect(
                    "Frequencies (Hz)",
                    options=[50, 60],
                    default=[50, 60],
                    help="Remove powerline noise"
                )
            else:
                notch_freqs = None
        
        with col2:
            st.subheader("Bandpass Filter")
            apply_bandpass = st.checkbox("Apply bandpass filter", value=True)
            
            if apply_bandpass:
                col_low, col_high = st.columns(2)
                with col_low:
                    lowcut = st.number_input(
                        "High-pass (Hz)",
                        min_value=0.1,
                        max_value=50.0,
                        value=0.5,
                        step=0.1
                    )
                with col_high:
                    highcut = st.number_input(
                        "Low-pass (Hz)",
                        min_value=1.0,
                        max_value=125.0,
                        value=40.0,
                        step=1.0
                    )
            else:
                lowcut = None
                highcut = None
        
        # Apply filters
        if st.button("Apply Filters", type="primary"):
            with st.spinner("Applying filters..."):
                raw_filtered = apply_filters(
                    raw,
                    notch_freqs=notch_freqs,
                    lowcut=lowcut,
                    highcut=highcut
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
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tmin = st.number_input(
                "Start time (s)",
                min_value=-2.0,
                max_value=0.0,
                value=-0.2,
                step=0.1,
                help="Time before stimulus onset"
            )
        
        with col2:
            tmax = st.number_input(
                "End time (s)",
                min_value=0.0,
                max_value=2.0,
                value=0.8,
                step=0.1,
                help="Time after stimulus onset"
            )
        
        with col3:
            baseline_mode = st.selectbox(
                "Baseline correction",
                options=["Yes", "No"],
                index=0
            )
        
        if baseline_mode == "Yes":
            baseline = (tmin, 0)
        else:
            baseline = None
        
        # Artifact rejection
        st.subheader("Artifact Rejection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_rejection = st.checkbox(
                "Enable artifact rejection",
                value=True,
                help="Reject epochs with excessive amplitude"
            )
        
        with col2:
            if use_rejection:
                reject_threshold = st.slider(
                    "Threshold (µV)",
                    min_value=50.0,
                    max_value=500.0,
                    value=300.0,
                    step=10.0,
                    help="Higher = more epochs kept"
                )
            else:
                reject_threshold = None
        
        # Create epochs
        if st.button("Create Epochs", type="primary"):
            try:
                with st.spinner("Creating epochs..."):
                    epochs = create_epochs(
                        raw, events, event_id,
                        tmin=tmin,
                        tmax=tmax,
                        baseline=baseline,
                        reject_threshold_uv=reject_threshold
                    )
                    st.session_state.epochs = epochs
                
                st.success(f"✅ Created {len(epochs)} epochs!")
                
                # Show drop statistics
                n_events = len([e for e in event_id.keys() 
                               if 'S1_onset' in e or 's1_onset' in e.lower()])
                n_dropped = n_events - len(epochs)
                drop_pct = (n_dropped / n_events * 100) if n_events > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Epochs", len(epochs))
                with col2:
                    st.metric("Dropped", n_dropped)
                with col3:
                    st.metric("Drop Rate", f"{drop_pct:.1f}%")
                
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
                st.dataframe(df_counts, use_container_width=True)
            
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
        
        # Compute ERPs
        if st.button("Compute ERPs", type="primary"):
            with st.spinner("Computing ERPs..."):
                probe_erp, irrelevant_erp, probe_events, irrelevant_events = compute_erps(epochs)
                
                if probe_erp is not None and irrelevant_erp is not None:
                    st.session_state.probe_erp = probe_erp
                    st.session_state.irrelevant_erp = irrelevant_erp
                    
                    st.success("✅ ERPs computed successfully!")
                    
                    # Show trial counts
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Probe Trials", len(probe_erp.nave))
                    with col2:
                        st.metric("Irrelevant Trials", len(irrelevant_erp.nave))
                
                else:
                    st.error("❌ Could not find probe/irrelevant events")
                    st.info(f"Found events: {list(epochs.event_id.keys())}")
        
        # Display ERP analysis
        if (st.session_state.probe_erp is not None and 
            st.session_state.irrelevant_erp is not None):
            
            probe_erp = st.session_state.probe_erp
            irrelevant_erp = st.session_state.irrelevant_erp
            
            # Plot ERPs
            st.subheader("ERP Waveforms")
            
            col1, col2 = st.columns(2)
            
            with col1:
                p300_start = st.number_input(
                    "P300 window start (s)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.05
                )
            
            with col2:
                p300_end = st.number_input(
                    "P300 window end (s)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.6,
                    step=0.05
                )
            
            fig_erp = plot_erps(probe_erp, irrelevant_erp, 
                               p300_window=(p300_start, p300_end))
            st.pyplot(fig_erp)
            plt.close()
            
            # P300 Analysis
            st.subheader("P300 Amplitude Analysis")
            
            p300_df = analyze_p300(probe_erp, irrelevant_erp, 
                                  tmin=p300_start, tmax=p300_end)
            
            st.dataframe(p300_df, use_container_width=True)
            
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
                1. Extract mean amplitudes from each epoch in P300 window (300-600ms)
                2. Bootstrap resample (default: 1000 iterations):
                   - Randomly sample probe epochs with replacement
                   - Randomly sample irrelevant epochs with replacement
                   - Calculate: `diff = mean(probe) - mean(irrelevant)`
                   - Count when `diff > 0` (probe > irrelevant)
                3. Calculate proportion: `p = count(diff > 0) / n_iterations`
                4. Classify:
                   - **Guilty** if `p ≥ 90%` (probe consistently larger)
                   - **Innocent** if `p < 90%` (no consistent difference)
                
                **Interpretation:**
                - 95-100%: Very strong evidence of recognition
                - 90-95%: Strong evidence of recognition (guilty)
                - 75-90%: Moderate evidence (inconclusive)
                - 50-75%: Weak/no evidence
                - <50%: Irrelevant larger than probe (unusual)
                """)
            
            # Parameters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                bad_n_bootstrap = st.number_input(
                    "Bootstrap iterations",
                    min_value=100,
                    max_value=10000,
                    value=1000,
                    step=100,
                    help="More iterations = more stable results (slower)"
                )
            
            with col2:
                bad_threshold = st.slider(
                    "Guilty threshold",
                    min_value=0.50,
                    max_value=0.99,
                    value=0.90,
                    step=0.01,
                    help="Classification threshold (default: 0.90 = 90%)"
                )
            
            with col3:
                st.markdown("**Window:**")
                st.info(f"{p300_start:.2f}s - {p300_end:.2f}s")
            
            # Run CTP-BAD
            if st.button("🎲 Run CTP-BAD Analysis", type="primary"):
                try:
                    with st.spinner(f"Running bootstrap ({bad_n_bootstrap} iterations)..."):
                        bad_results = ctp_bad_analysis(
                            epochs,
                            tmin=p300_start,
                            tmax=p300_end,
                            n_bootstrap=int(bad_n_bootstrap),
                            threshold=bad_threshold
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
                
                col1, col2, col3 = st.columns([1, 1, 2])
                
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
                
                with col3:
                    verdict_color = "red" if bad_results['overall_classification'] == "GUILTY" else "green"
                    st.markdown(f"**Verdict:** :{verdict_color}[{bad_results['verdict']}]")
                
                # Analysis details
                with st.expander("📊 Analysis Details"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Probe Epochs", bad_results['n_probe_epochs'])
                    with col2:
                        st.metric("Irrelevant Epochs", bad_results['n_irrelevant_epochs'])
                    with col3:
                        st.metric("Bootstrap Iterations", bad_results['n_bootstrap'])
                    
                    st.info(f"**Threshold:** {bad_results['threshold']*100:.0f}% | "
                           f"**Best Channel:** {bad_results['max_channel']}")
                
                # Per-channel results
                st.subheader("Per-Channel Results")
                st.dataframe(bad_results['channel_results'], use_container_width=True)
                
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
    # Page 6: Export Results
    # ========================================================================
    
    elif page == "📉 Export Results":
        st.header("📉 Export Results")
        
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
