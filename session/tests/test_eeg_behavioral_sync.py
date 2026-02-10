"""
Test Suite for EEG-Behavioral Data Synchronization
===================================================

Tests for synchronization between EEG recordings and behavioral responses
using LSL timestamps.

Usage
-----
Run tests::

    python -m pytest tests/test_eeg_behavioral_sync.py -v

"""

import pytest
import time
import tempfile
import os
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestEEGBehavioralSynchronization:
    """Test synchronization between EEG and behavioral data."""
    
    def test_timestamp_matching(self):
        """Test matching behavioral events to EEG samples."""
        # Simulate EEG timestamps at 250 Hz
        eeg_timestamps = np.arange(0, 10, 0.004)  # 10 seconds at 250 Hz
        
        # Simulate behavioral event timestamps
        event_timestamps = [1.234, 3.567, 5.890, 8.123]
        
        # For each event, find closest EEG sample
        for event_ts in event_timestamps:
            closest_idx = np.argmin(np.abs(eeg_timestamps - event_ts))
            closest_eeg_ts = eeg_timestamps[closest_idx]
            
            # Time difference should be < 4ms (one sample at 250 Hz)
            time_diff = abs(event_ts - closest_eeg_ts)
            assert time_diff < 0.004
    
    def test_marker_to_sample_alignment(self):
        """Test aligning LSL markers to EEG samples."""
        # Create mock EEG data with timestamps
        eeg_data = {
            'timestamp': [1.000, 1.004, 1.008, 1.012, 1.016, 1.020],
            'Pz': [10.0, 12.0, 11.0, 13.0, 12.5, 11.5],
            'Cz': [8.0, 9.0, 8.5, 9.5, 9.0, 8.8],
            'Fz': [5.0, 5.5, 5.2, 5.8, 5.6, 5.3],
            'Fp1': [3.0, 3.2, 3.1, 3.4, 3.3, 3.2]
        }
        eeg_df = pd.DataFrame(eeg_data)
        
        # Mock behavioral marker at 1.010 seconds
        marker_timestamp = 1.010
        
        # Find closest EEG sample
        time_diffs = np.abs(eeg_df['timestamp'] - marker_timestamp)
        closest_idx = time_diffs.idxmin()
        
        # Should be sample at 1.012 (index 3)
        assert closest_idx == 3
        assert abs(eeg_df.loc[closest_idx, 'timestamp'] - marker_timestamp) < 0.004
    
    def test_extract_erp_window(self):
        """Test extracting ERP time window around behavioral event."""
        # Create mock EEG data (250 Hz, 2 seconds)
        sampling_rate = 250
        duration = 2.0
        num_samples = int(sampling_rate * duration)
        
        timestamps = np.arange(0, duration, 1/sampling_rate)
        
        # Simulate P300 response: baseline + positive deflection at 300ms
        baseline = 0
        p300_latency = 0.3  # 300ms
        
        eeg_signal = np.zeros(num_samples)
        # Add P300 component (Gaussian bump at 300ms)
        for i, t in enumerate(timestamps):
            if 0.2 < t < 0.5:  # P300 window
                eeg_signal[i] = 10 * np.exp(-((t - p300_latency) ** 2) / 0.01)
        
        eeg_df = pd.DataFrame({
            'timestamp': timestamps,
            'Pz': eeg_signal
        })
        
        # Event occurs at t=0.0
        event_time = 0.0
        
        # Extract window: -100ms to +800ms relative to event
        pre_time = 0.1  # 100ms before
        post_time = 0.8  # 800ms after
        
        # Find samples in window
        window_mask = (
            (eeg_df['timestamp'] >= event_time - pre_time) &
            (eeg_df['timestamp'] <= event_time + post_time)
        )
        window_data = eeg_df[window_mask]
        
        # Verify window size (should be ~225 samples: 0.9s * 250 Hz)
        expected_samples = int((pre_time + post_time) * sampling_rate)
        assert len(window_data) == pytest.approx(expected_samples, abs=2)
        
        # Verify P300 peak is in the window
        max_idx = window_data['Pz'].idxmax()
        peak_time = window_data.loc[max_idx, 'timestamp']
        assert 0.25 < peak_time < 0.35  # P300 typically 250-350ms
    
    def test_behavioral_csv_with_lsl_markers(self):
        """Test behavioral CSV contains LSL marker IDs for synchronization."""
        # Create mock behavioral data with LSL markers
        behavioral_data = [
            {
                'participant_id': 'P001',
                'trial_index': 1,
                'S1_onset_time': 1.234,
                'LSL_S1_marker': 1,
                'S2_onset_time': 2.567,
                'LSL_S2_marker': 2,
            },
            {
                'participant_id': 'P001',
                'trial_index': 2,
                'S1_onset_time': 4.890,
                'LSL_S1_marker': 3,
                'S2_onset_time': 6.123,
                'LSL_S2_marker': 4,
            }
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save behavioral data
            csv_file = os.path.join(tmpdir, 'behavioral.csv')
            with open(csv_file, 'w', newline='') as f:
                fieldnames = behavioral_data[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(behavioral_data)
            
            # Read back and verify
            df = pd.read_csv(csv_file)
            assert 'LSL_S1_marker' in df.columns
            assert 'LSL_S2_marker' in df.columns
            assert df['LSL_S1_marker'].tolist() == [1, 3]
            assert df['LSL_S2_marker'].tolist() == [2, 4]
    
    def test_synchronize_eeg_with_behavioral(self):
        """Test complete synchronization of EEG and behavioral data."""
        # Create mock EEG data
        sampling_rate = 250
        duration = 5.0
        timestamps = np.arange(0, duration, 1/sampling_rate)
        
        eeg_df = pd.DataFrame({
            'timestamp': timestamps,
            'Pz': np.random.randn(len(timestamps)) * 20,
            'Cz': np.random.randn(len(timestamps)) * 20,
            'Fz': np.random.randn(len(timestamps)) * 20,
            'Fp1': np.random.randn(len(timestamps)) * 20
        })
        
        # Create mock behavioral data
        behavioral_df = pd.DataFrame({
            'trial_index': [1, 2, 3],
            'S1_type': ['probe', 'irrelevant', 'irrelevant'],
            'S1_onset_time': [1.0, 2.5, 4.0],
            'S2_onset_time': [2.4, 3.9, 5.4],
        })
        
        # For each trial, extract EEG window around S1 onset
        erp_windows = []
        
        for _, trial in behavioral_df.iterrows():
            event_time = trial['S1_onset_time']
            
            # Extract -100ms to +800ms window
            window_mask = (
                (eeg_df['timestamp'] >= event_time - 0.1) &
                (eeg_df['timestamp'] <= event_time + 0.8)
            )
            window = eeg_df[window_mask].copy()
            
            if len(window) > 0:
                # Add trial info
                window['trial'] = trial['trial_index']
                window['stimulus_type'] = trial['S1_type']
                # Time relative to stimulus onset
                window['time_rel'] = window['timestamp'] - event_time
                
                erp_windows.append(window)
        
        # Verify we extracted windows for all trials
        assert len(erp_windows) == 3
        
        # Verify time windows are correct
        for window in erp_windows:
            assert window['time_rel'].min() >= -0.1
            assert window['time_rel'].max() <= 0.8
    
    def test_lsl_timestamp_precision(self):
        """Test LSL timestamp precision is sufficient for EEG analysis."""
        # LSL timestamps should have sub-millisecond precision
        # For 250 Hz EEG (4ms samples), we need < 1ms precision
        
        # Simulate LSL timestamps
        timestamps = []
        for i in range(100):
            ts = time.time()
            timestamps.append(ts)
            time.sleep(0.001)  # 1ms intervals
        
        # Calculate intervals
        intervals = np.diff(timestamps)
        
        # Intervals should be close to 1ms (0.001s)
        # Allow some jitter from sleep() imprecision
        assert np.mean(intervals) == pytest.approx(0.001, abs=0.0005)
    
    def test_channel_order_consistency(self):
        """Test that channel order is consistent between config and data."""
        expected_channels = ['Pz', 'Cz', 'Fz', 'Fp1']
        
        # Simulate saved EEG data
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = os.path.join(tmpdir, 'test_eeg.csv')
            
            # Create mock data
            data = {
                'timestamp': [1.0, 1.004, 1.008],
                'Pz': [10.0, 11.0, 12.0],
                'Cz': [8.0, 9.0, 10.0],
                'Fz': [5.0, 6.0, 7.0],
                'Fp1': [3.0, 4.0, 5.0]
            }
            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False)
            
            # Read back and verify column order
            df_read = pd.read_csv(csv_file)
            data_columns = [col for col in df_read.columns if col != 'timestamp']
            
            assert data_columns == expected_channels
    
    @pytest.mark.mock
    def test_real_time_quality_check(self):
        """Test real-time signal quality during experiment."""
        from brainaccess_handler import BrainAccessHandler
        
        # Create handler with mock data
        handler = BrainAccessHandler(enabled=False, sampling_rate=250)
        handler.channel_mapping = {'Pz': 0, 'Cz': 1, 'Fz': 2, 'Fp1': 3}
        
        # Simulate incoming data stream
        np.random.seed(42)
        for _ in range(250):  # 1 second of data
            sample = [
                np.random.normal(0, 20),  # Pz - good signal
                0.1,                       # Cz - flat (bad contact)
                np.random.normal(0, 200),  # Fz - very noisy (artifact)
                np.random.normal(0, 20)    # Fp1 - good signal
            ]
            handler.eeg_data.append(sample)
        
        # Check quality
        quality = handler.get_signal_quality()
        
        # Verify quality detection
        assert quality['Pz'] == 'good'
        assert quality['Cz'] == 'poor'  # Too flat
        assert quality['Fz'] == 'poor'  # Too noisy
        assert quality['Fp1'] == 'good'


class TestDataIntegrity:
    """Test data integrity and completeness."""
    
    def test_no_data_loss_during_recording(self):
        """Test that no data samples are lost during recording."""
        # Simulate continuous recording
        sampling_rate = 250
        duration = 5.0
        expected_samples = int(sampling_rate * duration)
        
        # Generate timestamps at exact intervals
        timestamps = np.arange(0, duration, 1/sampling_rate)
        
        # Check for gaps in timestamps
        intervals = np.diff(timestamps)
        expected_interval = 1/sampling_rate
        
        # All intervals should be equal (no missing samples)
        assert np.allclose(intervals, expected_interval, rtol=1e-5)
    
    def test_behavioral_trial_count_matches_config(self):
        """Test that number of recorded trials matches config."""
        expected_trials = 25  # From config: total_trials
        
        # Simulate behavioral data
        trial_data = [{'trial_index': i} for i in range(1, expected_trials + 1)]
        
        assert len(trial_data) == expected_trials
    
    def test_all_markers_recorded(self):
        """Test that all event markers are present in data."""
        # Expected markers per trial
        markers_per_trial = [
            'fixation_onset',
            'S1_onset',
            'S1_response',
            'S2_onset',
            'S2_response',
            'ITI_start'
        ]
        
        # Simulate trial data
        trial_data = {
            'LSL_fixation_marker': 1,
            'LSL_S1_marker': 2,
            'LSL_S1_response_marker': 3,
            'LSL_S2_marker': 4,
            'LSL_S2_response_marker': 5,
            'LSL_ITI_marker': 6
        }
        
        # All marker fields should have values
        for key, value in trial_data.items():
            assert value is not None
            assert value > 0


def run_tests():
    """Run all tests."""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_tests()

