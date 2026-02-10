"""
Example MNE Analysis
====================

Demonstrates how to analyze EEG data recorded with BrainAccess Board
using MNE-Python.

Prerequisites:
- BrainAccess Board recording (.edf file) with LSL markers
- OR: Python-saved .fif file with annotations

Usage:
    python scripts/example_mne_analysis.py --eeg data/board/P001_S01_run001.edf

"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    import mne
except ImportError:
    print("ERROR: MNE-Python not installed!")
    print("Install with: pip install mne")
    sys.exit(1)


def load_eeg_with_markers(eeg_file: str):
    """
    Load EEG data with embedded markers.
    
    Parameters
    ----------
    eeg_file : str
        Path to .edf or .fif file
        
    Returns
    -------
    raw : mne.io.Raw
        Raw EEG object
    events : np.ndarray
        Events array [n_events, 3]
    event_id : dict
        Mapping of event names to IDs
    """
    print(f"\nLoading EEG: {eeg_file}")
    
    file_ext = Path(eeg_file).suffix.lower()
    
    if file_ext == '.edf':
        raw = mne.io.read_raw_edf(eeg_file, preload=True)
    elif file_ext == '.fif':
        raw = mne.io.read_raw_fif(eeg_file, preload=True)
    else:
        raise ValueError(f"Unsupported format: {file_ext}")
    
    print(f"  Channels: {raw.ch_names}")
    print(f"  Sampling rate: {raw.info['sfreq']} Hz")
    print(f"  Duration: {raw.times[-1]:.2f} s")
    print(f"  N samples: {len(raw.times)}")
    
    # Extract events from annotations
    print("\nExtracting events from annotations...")
    events, event_id = mne.events_from_annotations(raw)
    
    print(f"  Total events: {len(events)}")
    print(f"  Event types: {event_id}")
    
    return raw, events, event_id


def check_signal_quality(raw, duration=10.0):
    """
    Check EEG signal quality before epoching.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    duration : float
        Duration to check (seconds)
    """
    print("\n" + "="*70)
    print("Signal Quality Check")
    print("="*70)
    
    # Get data segment
    start_sample = 0
    end_sample = min(int(duration * raw.info['sfreq']), len(raw.times))
    data, times = raw[:, start_sample:end_sample]
    
    print(f"\nAnalyzing first {duration:.1f} seconds:")
    
    for i, ch_name in enumerate(raw.ch_names):
        ch_data = data[i, :] * 1e6  # Convert to µV
        
        mean_val = np.mean(ch_data)
        std_val = np.std(ch_data)
        min_val = np.min(ch_data)
        max_val = np.max(ch_data)
        p2p = max_val - min_val
        
        # Assess quality
        if std_val < 1.0 or p2p < 5.0:
            quality = "⚠ FLAT (no signal - check connection!)"
        elif std_val > 200.0 or p2p > 1000.0:
            quality = "✗ NOISY (too much artifact - check electrodes!)"
        elif std_val > 100.0 or p2p > 500.0:
            quality = "⚠ POOR (high noise - may drop epochs)"
        else:
            quality = "✓ GOOD"
        
        print(f"  {ch_name:6s}: mean={mean_val:7.2f} µV, std={std_val:6.2f} µV, "
              f"p2p={p2p:7.2f} µV  {quality}")
    
    print("\nInterpretation:")
    print("  - FLAT: Electrode not connected or bad contact")
    print("  - NOISY: Too much artifact (movement, EMG, power line)")
    print("  - POOR: Marginal quality, some epochs may be rejected")
    print("  - GOOD: Signal quality acceptable for analysis")
    print("="*70)


def create_epochs(raw, events, event_id, tmin=-0.2, tmax=0.8, reject_threshold_uv=300.0):
    """
    Create epochs around stimulus onsets.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    events : np.ndarray
        Events array
    event_id : dict
        Event ID mapping
    tmin : float
        Start time before event (seconds)
    tmax : float
        End time after event (seconds)
        
    Returns
    -------
    epochs : mne.Epochs
        Epoched data
    """
    print(f"\nCreating epochs: {tmin} to {tmax} s around events")
    
    # Filter event IDs to only include S1 onsets
    s1_event_id = {
        k: v for k, v in event_id.items()
        if 'S1_onset' in k or 's1_onset' in k.lower()
    }
    
    if not s1_event_id:
        print("WARNING: No S1_onset events found!")
        print(f"Available events: {list(event_id.keys())}")
        # Use all events as fallback
        s1_event_id = event_id
    
    print(f"  Using event types: {list(s1_event_id.keys())}")
    
    # Artifact rejection threshold
    if reject_threshold_uv is None:
        # No rejection - keep all epochs
        reject_dict = None
        print("  Artifact rejection: DISABLED (keeping all epochs)")
    else:
        # Note: MNE uses Volts (V), so we convert µV to V
        # reject_threshold_uv (µV) → reject_threshold (V)
        # Higher threshold = more epochs kept (less strict)
        # Typical values:
        #   - Strict: 100 µV
        #   - Moderate: 150 µV
        #   - Relaxed: 300 µV (default)
        #   - Very relaxed: 500 µV
        reject_threshold = reject_threshold_uv * 1e-6  # Convert µV to V
        reject_dict = dict(eeg=reject_threshold)
        print(f"  Artifact rejection threshold: {reject_threshold * 1e6:.0f} µV")
    
    epochs = mne.Epochs(
        raw,
        events,
        event_id=s1_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=(tmin, 0),
        preload=True,
        reject=reject_dict,
        verbose=False
    )
    
    print(f"  Created {len(epochs)} epochs")
    
    # Check if any epochs survived
    if len(epochs) == 0:
        print("\n⚠ WARNING: ALL EPOCHS WERE DROPPED!")
        print("\nPossible causes:")
        print("  1. EEG cap not worn during recording")
        print("  2. Poor electrode contact (high impedance)")
        print("  3. Reject threshold too strict (currently 150 µV)")
        print("  4. No S1_onset events in the data")
        print("\nTroubleshooting:")
        print("  - Check signal quality during recording")
        print("  - Inspect raw data: raw.plot()")
        print("  - Lower reject threshold: reject=dict(eeg=300e-6)")
        print("  - Check electrode impedances")
        
        # Show drop log
        print("\nDrop log (first 10 epochs):")
        for i, log in enumerate(epochs.drop_log[:10]):
            if log:
                print(f"  Epoch {i}: {log}")
        
        raise ValueError("No epochs available for analysis. Check EEG data quality!")
    else:
        drop_stats = epochs.drop_log_stats()
        if isinstance(drop_stats, dict) and 'percent' in drop_stats:
            print(f"  Dropped {drop_stats['percent']:.1f}% due to artifacts")
        
    return epochs


def compute_erps(epochs):
    """
    Compute ERPs for probe and irrelevant stimuli.
    
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
    print("\nComputing ERPs...")
    
    # Get available event types (may have metadata like "S1_onset_probe|trial=1")
    available_events = list(epochs.event_id.keys())
    
    # Find probe and irrelevant events using substring matching
    probe_events = [k for k in available_events if 'probe' in k.lower() and 's1_onset' in k.lower()]
    irrelevant_events = [k for k in available_events if 'irrelevant' in k.lower() and 's1_onset' in k.lower()]
    
    probe_erp = None
    irrelevant_erp = None
    
    # Compute probe ERP (average all probe events together)
    if probe_events:
        # Select all probe epochs and average
        probe_epochs = epochs[probe_events]
        probe_erp = probe_epochs.average()
        print(f"  Probe ERP: {len(probe_epochs)} trials (from {len(probe_events)} event types)")
    
    # Compute irrelevant ERP (average all irrelevant events together)
    if irrelevant_events:
        # Select all irrelevant epochs and average
        irrelevant_epochs = epochs[irrelevant_events]
        irrelevant_erp = irrelevant_epochs.average()
        print(f"  Irrelevant ERP: {len(irrelevant_epochs)} trials (from {len(irrelevant_events)} event types)")
    
    if probe_erp is None or irrelevant_erp is None:
        print("\nWARNING: Could not find probe/irrelevant events!")
        print(f"Probe events found: {len(probe_events)}")
        print(f"Irrelevant events found: {len(irrelevant_events)}")
        if probe_events:
            print(f"  Probe example: {probe_events[0]}")
        if irrelevant_events:
            print(f"  Irrelevant example: {irrelevant_events[0]}")
        print(f"\nAll available events: {len(available_events)}")
        if len(available_events) <= 10:
            for evt in available_events:
                print(f"  - {evt}")
    
    return probe_erp, irrelevant_erp


def plot_erps(probe_erp, irrelevant_erp, times):
    """Plot ERPs and P300 difference wave."""
    print("\nPlotting ERPs...")
    
    if probe_erp is None or irrelevant_erp is None:
        print("ERROR: Cannot plot - ERPs not computed")
        return
    
    # Create figure
    n_channels = len(probe_erp.ch_names)
    fig, axes = plt.subplots(1, n_channels, figsize=(5*n_channels, 4))
    
    if n_channels == 1:
        axes = [axes]
    
    for idx, (ch_name, ax) in enumerate(zip(probe_erp.ch_names, axes)):
        # Plot probe and irrelevant ERPs
        probe_data = probe_erp.data[idx] * 1e6  # Convert to µV
        irrelevant_data = irrelevant_erp.data[idx] * 1e6
        
        ax.plot(times, probe_data, label='Probe', color='red', linewidth=2)
        ax.plot(times, irrelevant_data, label='Irrelevant', color='blue', linewidth=2)
        
        # Plot difference wave
        diff_data = probe_data - irrelevant_data
        ax.plot(times, diff_data, label='Difference (P300)', color='green',
                linewidth=2, linestyle='--')
        
        # Styling
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(0, color='black', linestyle='--', linewidth=0.5, label='Stimulus onset')
        ax.axvspan(0.3, 0.6, alpha=0.1, color='gray', label='P300 window')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(f'ERP at {ch_name}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = "erp_analysis.png"
    plt.savefig(output_file, dpi=150)
    print(f"  Saved figure: {output_file}")
    
    plt.show()


def analyze_p300_amplitude(probe_erp, irrelevant_erp):
    """Analyze P300 amplitude in 300-600 ms window."""
    print("\nP300 Amplitude Analysis:")
    print("  Time window: 300-600 ms")
    
    if probe_erp is None or irrelevant_erp is None:
        return
    
    times = probe_erp.times
    time_mask = (times >= 0.3) & (times <= 0.6)
    
    for idx, ch_name in enumerate(probe_erp.ch_names):
        probe_data = probe_erp.data[idx] * 1e6  # µV
        irrelevant_data = irrelevant_erp.data[idx] * 1e6
        
        probe_mean = probe_data[time_mask].mean()
        irrelevant_mean = irrelevant_data[time_mask].mean()
        diff = probe_mean - irrelevant_mean
        
        print(f"\n  Channel {ch_name}:")
        print(f"    Probe:      {probe_mean:7.2f} µV")
        print(f"    Irrelevant: {irrelevant_mean:7.2f} µV")
        print(f"    Difference: {diff:7.2f} µV")
        
        if diff > 5.0:
            print(f"    ✓ P300 effect detected (diff > 5 µV)")
        else:
            print(f"    ⚠ Weak/no P300 effect")


def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(
        description='Analyze EEG data with MNE-Python'
    )
    parser.add_argument(
        '--eeg',
        type=str,
        required=True,
        help='Path to EEG file (.edf or .fif)'
    )
    parser.add_argument(
        '--reject',
        type=float,
        default=300.0,
        help='Artifact rejection threshold in µV (default: 300). '
             'Higher = more epochs kept. Try 500 for noisy data.'
    )
    parser.add_argument(
        '--no-reject',
        action='store_true',
        help='Disable artifact rejection (keep all epochs)'
    )
    
    args = parser.parse_args()
    
    if not Path(args.eeg).exists():
        print(f"ERROR: File not found: {args.eeg}")
        sys.exit(1)
    
    print("="*70)
    print("MNE-Python EEG Analysis")
    print("="*70)
    
    # Load data
    raw, events, event_id = load_eeg_with_markers(args.eeg)
    
    # Check signal quality
    check_signal_quality(raw, duration=10.0)
    
    # Create epochs
    epochs = create_epochs(
        raw, events, event_id,
        tmin=-0.2, tmax=0.8,
        reject_threshold_uv=args.reject if not args.no_reject else None
    )
    
    # Compute ERPs
    probe_erp, irrelevant_erp = compute_erps(epochs)
    
    # Analyze P300
    analyze_p300_amplitude(probe_erp, irrelevant_erp)
    
    # Plot
    if probe_erp and irrelevant_erp:
        plot_erps(probe_erp, irrelevant_erp, probe_erp.times)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == '__main__':
    main()
