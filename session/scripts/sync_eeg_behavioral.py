"""
Synchronize EEG and Behavioral Data
=====================================

Helper script to demonstrate how to synchronize continuous EEG recordings
with discrete behavioral events (stimulus onsets, responses).

Usage
-----
::

    python scripts/sync_eeg_behavioral.py \\
        --eeg data/eeg/P001_S01_eeg_20260108_211416.csv \\
        --behavioral data/behavioral/P001_S01_behavioral_20260108_211330.csv \\
        --output data/synchronized/P001_S01_epochs.npz

"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_eeg_data(eeg_file: str) -> tuple[pd.DataFrame, list[str], float]:
    """
    Load continuous EEG data from CSV.
    
    Parameters
    ----------
    eeg_file : str
        Path to EEG CSV file
    
    Returns
    -------
    tuple
        (eeg_df, channel_names, sampling_rate)
    """
    df = pd.read_csv(eeg_file)
    
    # Extract channel names (all columns except timestamp)
    channels = [col for col in df.columns if col != 'timestamp']
    
    # Estimate sampling rate from timestamps
    if len(df) > 1:
        time_diffs = np.diff(df['timestamp'].values[:100])
        avg_interval = np.mean(time_diffs)
        sampling_rate = 1.0 / avg_interval if avg_interval > 0 else 250.0
    else:
        sampling_rate = 250.0
    
    print(f"Loaded EEG: {len(df)} samples, {len(channels)} channels")
    print(f"  Channels: {channels}")
    print(f"  Sampling rate: {sampling_rate:.1f} Hz")
    print(f"  Duration: {(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]):.2f} s")
    
    return df, channels, sampling_rate


def load_behavioral_data(behavioral_file: str) -> pd.DataFrame:
    """
    Load behavioral event data from CSV.
    
    Parameters
    ----------
    behavioral_file : str
        Path to behavioral CSV file
    
    Returns
    -------
    pd.DataFrame
        Behavioral events dataframe
    """
    df = pd.read_csv(behavioral_file)
    
    print(f"\nLoaded behavioral: {len(df)} trials")
    print(f"  Conditions: {df['S1_type'].value_counts().to_dict()}")
    
    return df


def find_nearest_eeg_index(eeg_timestamps: np.ndarray, event_time: float) -> int:
    """
    Find index of EEG sample closest to behavioral event time.
    
    Parameters
    ----------
    eeg_timestamps : np.ndarray
        Array of EEG sample timestamps
    event_time : float
        Unix timestamp of behavioral event
    
    Returns
    -------
    int
        Index of nearest EEG sample
    """
    idx = np.searchsorted(eeg_timestamps, event_time)
    
    # Clamp to valid range
    idx = max(0, min(idx, len(eeg_timestamps) - 1))
    
    # Check if previous sample is closer
    if idx > 0:
        if abs(eeg_timestamps[idx - 1] - event_time) < abs(eeg_timestamps[idx] - event_time):
            idx -= 1
    
    return idx


def extract_epoch(eeg_df: pd.DataFrame, 
                  channels: list[str],
                  center_idx: int,
                  pre_samples: int,
                  post_samples: int) -> np.ndarray:
    """
    Extract epoch window around event.
    
    Parameters
    ----------
    eeg_df : pd.DataFrame
        EEG dataframe
    channels : list of str
        Channel names
    center_idx : int
        Index of event onset in EEG
    pre_samples : int
        Number of samples before event
    post_samples : int
        Number of samples after event
    
    Returns
    -------
    np.ndarray
        Epoch data [timepoints, channels] or None if out of bounds
    """
    start_idx = center_idx - pre_samples
    end_idx = center_idx + post_samples
    
    # Check bounds
    if start_idx < 0 or end_idx >= len(eeg_df):
        return None
    
    # Extract epoch
    epoch = eeg_df.iloc[start_idx:end_idx][channels].values
    
    return epoch


def synchronize_data(eeg_file: str, 
                     behavioral_file: str,
                     pre_ms: float = 200.0,
                     post_ms: float = 800.0) -> dict:
    """
    Synchronize EEG and behavioral data by extracting epochs around events.
    
    Parameters
    ----------
    eeg_file : str
        Path to EEG CSV
    behavioral_file : str
        Path to behavioral CSV
    pre_ms : float, optional
        Pre-event window in milliseconds (default: 200)
    post_ms : float, optional
        Post-event window in milliseconds (default: 800)
    
    Returns
    -------
    dict
        Dictionary with synchronized data:
        - 'epochs': np.ndarray [n_trials, n_timepoints, n_channels]
        - 'times': np.ndarray [n_timepoints] in seconds relative to event
        - 'channels': list of channel names
        - 'trial_info': pd.DataFrame with trial metadata
        - 'sampling_rate': float
    """
    # Load data
    eeg_df, channels, sfreq = load_eeg_data(eeg_file)
    behavioral_df = load_behavioral_data(behavioral_file)
    
    # Calculate epoch parameters
    pre_samples = int(pre_ms / 1000.0 * sfreq)
    post_samples = int(post_ms / 1000.0 * sfreq)
    n_timepoints = pre_samples + post_samples
    
    print(f"\nEpoch parameters:")
    print(f"  Window: -{pre_ms} ms to +{post_ms} ms")
    print(f"  Samples: {pre_samples} + {post_samples} = {n_timepoints}")
    
    # Extract epochs for each trial
    epochs_list = []
    trial_info_list = []
    eeg_timestamps = eeg_df['timestamp'].values
    
    print(f"\nExtracting epochs...")
    
    for idx, row in behavioral_df.iterrows():
        # Get S1 onset time (stimulus presentation)
        event_time = row['timestamp_unix']
        
        # Find nearest EEG sample
        eeg_idx = find_nearest_eeg_index(eeg_timestamps, event_time)
        
        # Time offset between event and EEG sample
        time_offset_ms = (eeg_timestamps[eeg_idx] - event_time) * 1000
        
        # Extract epoch
        epoch = extract_epoch(eeg_df, channels, eeg_idx, pre_samples, post_samples)
        
        if epoch is not None:
            epochs_list.append(epoch)
            
            # Store trial info
            trial_info = {
                'trial_index': row['trial_index'],
                'block': row['block'],
                'S1_type': row['S1_type'],
                'S1_object': row['S1_object'],
                'S2_type': row['S2_type'],
                'S2_correct': row['S2_correct'],
                'S1_RT': row['S1_RT'],
                'S2_RT': row['S2_RT'],
                'event_time': event_time,
                'eeg_sample_index': eeg_idx,
                'time_offset_ms': time_offset_ms
            }
            trial_info_list.append(trial_info)
        else:
            print(f"  Warning: Trial {row['trial_index']} out of bounds, skipped")
    
    print(f"Extracted {len(epochs_list)} / {len(behavioral_df)} epochs")
    
    # Convert to arrays
    epochs = np.array(epochs_list)  # [n_trials, n_timepoints, n_channels]
    trial_info = pd.DataFrame(trial_info_list)
    
    # Create time vector (in seconds relative to event)
    times = np.arange(-pre_samples, post_samples) / sfreq
    
    # Summary
    print(f"\nSynchronized data:")
    print(f"  Shape: {epochs.shape} [trials, timepoints, channels]")
    print(f"  Time range: {times[0]*1000:.1f} to {times[-1]*1000:.1f} ms")
    print(f"  Condition breakdown:")
    for cond, count in trial_info['S1_type'].value_counts().items():
        print(f"    {cond}: {count} trials")
    
    return {
        'epochs': epochs,
        'times': times,
        'channels': channels,
        'trial_info': trial_info,
        'sampling_rate': sfreq,
        'pre_ms': pre_ms,
        'post_ms': post_ms
    }


def save_synchronized_data(data: dict, output_file: str) -> None:
    """
    Save synchronized data to NPZ file.
    
    Parameters
    ----------
    data : dict
        Synchronized data dictionary
    output_file : str
        Path to output NPZ file
    """
    # Prepare data for saving
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save NPZ
    np.savez(
        output_file,
        epochs=data['epochs'],
        times=data['times'],
        channels=data['channels'],
        sampling_rate=data['sampling_rate'],
        pre_ms=data['pre_ms'],
        post_ms=data['post_ms']
    )
    
    # Save trial info as CSV
    trial_info_file = output_file.replace('.npz', '_trial_info.csv')
    data['trial_info'].to_csv(trial_info_file, index=False)
    
    print(f"\nSaved synchronized data:")
    print(f"  Epochs: {output_file}")
    print(f"  Trial info: {trial_info_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Synchronize EEG and behavioral data'
    )
    parser.add_argument(
        '--eeg',
        type=str,
        required=True,
        help='Path to EEG CSV file'
    )
    parser.add_argument(
        '--behavioral',
        type=str,
        required=True,
        help='Path to behavioral CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output NPZ file'
    )
    parser.add_argument(
        '--pre-ms',
        type=float,
        default=200.0,
        help='Pre-event window in milliseconds (default: 200)'
    )
    parser.add_argument(
        '--post-ms',
        type=float,
        default=800.0,
        help='Post-event window in milliseconds (default: 800)'
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.eeg).exists():
        print(f"Error: EEG file not found: {args.eeg}")
        sys.exit(1)
    
    if not Path(args.behavioral).exists():
        print(f"Error: Behavioral file not found: {args.behavioral}")
        sys.exit(1)
    
    # Synchronize data
    print("="*60)
    print("EEG-Behavioral Data Synchronization")
    print("="*60 + "\n")
    
    data = synchronize_data(
        args.eeg,
        args.behavioral,
        pre_ms=args.pre_ms,
        post_ms=args.post_ms
    )
    
    # Save
    save_synchronized_data(data, args.output)
    
    print("\nSynchronization complete!")


if __name__ == '__main__':
    main()
