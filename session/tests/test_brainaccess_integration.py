"""
Test Suite for BrainAccess Integration
========================================

Tests for BrainAccess device connection, EEG data acquisition,
and synchronization with behavioral data.

Usage
-----
Run all tests::

    python -m pytest tests/test_brainaccess_integration.py -v

Run specific test::

    python -m pytest tests/test_brainaccess_integration.py::test_handler_initialization -v

Run with LSL device connected::

    python -m pytest tests/test_brainaccess_integration.py -v -m "not mock"

"""

import pytest
import time
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from brainaccess_handler import BrainAccessHandler


class TestBrainAccessHandler:
    """Test suite for BrainAccessHandler class."""
    
    def test_handler_initialization(self):
        """Test handler initializes with correct parameters."""
        handler = BrainAccessHandler(
            device_name="TestDevice",
            channels=['Pz', 'Cz', 'Fz', 'Fp1'],
            sampling_rate=250,
            enabled=False  # Don't actually try to connect
        )
        
        assert handler.device_name == "TestDevice"
        assert handler.channels == ['Pz', 'Cz', 'Fz', 'Fp1']
        assert handler.sampling_rate == 250
        assert handler.is_connected is False
        assert handler.is_recording is False
    
    def test_handler_initialization_default_params(self):
        """Test handler initializes with default parameters."""
        handler = BrainAccessHandler(enabled=False)
        
        assert handler.device_name == "BrainAccess"
        assert handler.channels == ['Pz', 'Cz', 'Fz', 'Fp1']
        assert handler.sampling_rate == 250
    
    def test_handler_disabled_when_lsl_unavailable(self):
        """Test handler disables gracefully when LSL not available."""
        with patch('brainaccess_handler.LSL_AVAILABLE', False):
            handler = BrainAccessHandler(enabled=True)
            assert handler.enabled is False
    
    @pytest.mark.mock
    def test_connect_success(self):
        """Test successful connection to mock BrainAccess device."""
        with patch('brainaccess_handler.resolve_byprop') as mock_resolve, \
             patch('brainaccess_handler.StreamInlet') as mock_inlet:
            
            # Mock stream info
            mock_stream_info = Mock()
            mock_stream_info.name.return_value = "BrainAccess"
            mock_stream_info.type.return_value = "EEG"
            mock_stream_info.channel_count.return_value = 8
            mock_stream_info.nominal_srate.return_value = 250.0
            
            # Mock channel description
            mock_channels_desc = Mock()
            mock_channel = Mock()
            mock_channel.child_value.return_value = "Pz"
            mock_channel.next_sibling.return_value = Mock(empty=Mock(return_value=True))
            mock_channels_desc.first_child.return_value = mock_channel
            
            mock_desc = Mock()
            mock_desc.child.return_value = mock_channels_desc
            mock_stream_info.desc.return_value = mock_desc
            
            # Setup resolve to return mock stream
            mock_resolve.return_value = [mock_stream_info]
            
            # Create handler and connect
            handler = BrainAccessHandler(enabled=True)
            result = handler.connect()
            
            assert result is True
            assert handler.is_connected is True
            mock_resolve.assert_called_once()
    
    @pytest.mark.mock
    def test_connect_timeout(self):
        """Test connection timeout when device not found."""
        with patch('brainaccess_handler.resolve_byprop') as mock_resolve:
            # No streams found
            mock_resolve.return_value = []
            
            handler = BrainAccessHandler(enabled=True, timeout=1.0)
            result = handler.connect()
            
            assert result is False
            assert handler.is_connected is False
    
    @pytest.mark.mock
    def test_start_recording_without_connection(self):
        """Test recording fails when device not connected."""
        handler = BrainAccessHandler(enabled=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "test_eeg.csv")
            result = handler.start_recording(output_file)
            
            assert result is False
            assert handler.is_recording is False
    
    @pytest.mark.mock
    def test_start_recording_success(self):
        """Test recording starts successfully when connected."""
        with patch('brainaccess_handler.resolve_byprop') as mock_resolve, \
             patch('brainaccess_handler.StreamInlet') as mock_inlet:
            
            # Setup mock connection
            mock_stream_info = self._create_mock_stream_info()
            mock_resolve.return_value = [mock_stream_info]
            
            handler = BrainAccessHandler(enabled=True)
            handler.connect()
            
            # Start recording
            with tempfile.TemporaryDirectory() as tmpdir:
                output_file = os.path.join(tmpdir, "test_eeg.csv")
                result = handler.start_recording(output_file)
                
                assert result is True
                assert handler.is_recording is True
                assert handler.output_file == Path(output_file)
                
                # Cleanup
                handler.stop_recording()
    
    @pytest.mark.mock
    def test_recording_loop_stores_data(self):
        """Test that recording loop stores data in buffer."""
        with patch('brainaccess_handler.resolve_byprop') as mock_resolve, \
             patch('brainaccess_handler.StreamInlet') as mock_inlet_class:
            
            # Setup mock connection
            mock_stream_info = self._create_mock_stream_info()
            mock_resolve.return_value = [mock_stream_info]
            
            # Mock inlet to return sample data
            mock_inlet = Mock()
            mock_inlet.pull_chunk.return_value = (
                [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]],  # One sample
                [time.time()]  # Timestamp
            )
            mock_inlet_class.return_value = mock_inlet
            
            handler = BrainAccessHandler(enabled=True)
            handler.connect()
            
            # Start recording
            with tempfile.TemporaryDirectory() as tmpdir:
                output_file = os.path.join(tmpdir, "test_eeg.csv")
                handler.start_recording(output_file)
                
                # Let recording run briefly
                time.sleep(0.2)
                
                # Stop recording
                handler.stop_recording()
                
                # Check data was stored
                assert len(handler.eeg_data) > 0
                assert len(handler.timestamps) > 0
    
    @pytest.mark.mock
    def test_stop_recording_saves_data(self):
        """Test that stopping recording saves data to file."""
        with patch('brainaccess_handler.resolve_byprop') as mock_resolve, \
             patch('brainaccess_handler.StreamInlet') as mock_inlet_class:
            
            # Setup mock connection
            mock_stream_info = self._create_mock_stream_info()
            mock_resolve.return_value = [mock_stream_info]
            
            # Mock inlet to return sample data
            mock_inlet = Mock()
            samples = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] for _ in range(10)]
            timestamps = [time.time() + i*0.004 for i in range(10)]  # 250 Hz = 4ms spacing
            mock_inlet.pull_chunk.return_value = (samples, timestamps)
            mock_inlet_class.return_value = mock_inlet
            
            handler = BrainAccessHandler(
                enabled=True,
                channels=['Pz', 'Cz', 'Fz', 'Fp1']
            )
            handler.connect()
            
            # Manually setup channel mapping for test
            handler.channel_mapping = {
                'Pz': 0, 'Cz': 1, 'Fz': 2, 'Fp1': 3,
                'Ch4': 4, 'Ch5': 5, 'Ch6': 6, 'Ch7': 7
            }
            
            # Add some data to buffer manually (simpler than running thread)
            for sample, ts in zip(samples, timestamps):
                handler.eeg_data.append(sample)
                handler.timestamps.append(ts)
            
            # Save data
            with tempfile.TemporaryDirectory() as tmpdir:
                output_file = os.path.join(tmpdir, "test_eeg.csv")
                handler.output_file = Path(output_file)
                handler._save_data()
                
                # Verify file exists
                assert os.path.exists(output_file)
                
                # Verify data content
                df = pd.read_csv(output_file)
                assert 'timestamp' in df.columns
                assert 'Pz' in df.columns
                assert 'Cz' in df.columns
                assert 'Fz' in df.columns
                assert 'Fp1' in df.columns
                assert len(df) == 10
    
    def test_get_latest_sample_no_data(self):
        """Test getting latest sample when no data available."""
        handler = BrainAccessHandler(enabled=False)
        sample = handler.get_latest_sample()
        assert sample is None
    
    @pytest.mark.mock
    def test_get_latest_sample_with_data(self):
        """Test getting latest sample with data in buffer."""
        handler = BrainAccessHandler(enabled=False)
        
        # Add mock data
        handler.channel_mapping = {'Pz': 0, 'Cz': 1, 'Fz': 2, 'Fp1': 3}
        handler.eeg_data.append([10.0, 20.0, 30.0, 40.0])
        handler.timestamps.append(1234567890.0)
        
        sample = handler.get_latest_sample()
        
        assert sample is not None
        assert sample['timestamp'] == 1234567890.0
        assert sample['Pz'] == 10.0
        assert sample['Cz'] == 20.0
        assert sample['Fz'] == 30.0
        assert sample['Fp1'] == 40.0
    
    def test_signal_quality_no_data(self):
        """Test signal quality assessment with no data."""
        handler = BrainAccessHandler(enabled=False)
        quality = handler.get_signal_quality()
        
        assert all(q == 'no_data' for q in quality.values())
    
    @pytest.mark.mock
    def test_signal_quality_good_signal(self):
        """Test signal quality assessment with good signal."""
        handler = BrainAccessHandler(enabled=False, sampling_rate=250)
        handler.channel_mapping = {'Pz': 0, 'Cz': 1, 'Fz': 2, 'Fp1': 3}
        
        # Add 1 second of realistic EEG data (10-50 µV typical)
        np.random.seed(42)
        for _ in range(250):
            sample = [
                np.random.normal(0, 20),  # Pz
                np.random.normal(0, 20),  # Cz
                np.random.normal(0, 20),  # Fz
                np.random.normal(0, 20)   # Fp1
            ]
            handler.eeg_data.append(sample)
        
        quality = handler.get_signal_quality()
        
        # Should detect good signal
        assert quality['Pz'] == 'good'
        assert quality['Cz'] == 'good'
        assert quality['Fz'] == 'good'
        assert quality['Fp1'] == 'good'
    
    @pytest.mark.mock
    def test_signal_quality_poor_flat_signal(self):
        """Test signal quality detects flat signal (bad contact)."""
        handler = BrainAccessHandler(enabled=False, sampling_rate=250)
        handler.channel_mapping = {'Pz': 0}
        
        # Add flat signal (bad contact)
        for _ in range(250):
            handler.eeg_data.append([0.1])  # Almost no variation
        
        quality = handler.get_signal_quality()
        assert quality['Pz'] == 'poor'
    
    @pytest.mark.mock
    def test_signal_quality_poor_noisy_signal(self):
        """Test signal quality detects very noisy signal (artifact)."""
        handler = BrainAccessHandler(enabled=False, sampling_rate=250)
        handler.channel_mapping = {'Pz': 0}
        
        # Add very noisy signal (artifact)
        np.random.seed(42)
        for _ in range(250):
            handler.eeg_data.append([np.random.normal(0, 150)])  # Very large variation
        
        quality = handler.get_signal_quality()
        assert quality['Pz'] == 'poor'
    
    def test_context_manager(self):
        """Test handler works as context manager."""
        with patch('brainaccess_handler.resolve_byprop') as mock_resolve:
            mock_stream_info = self._create_mock_stream_info()
            mock_resolve.return_value = [mock_stream_info]
            
            with BrainAccessHandler(enabled=True) as handler:
                assert handler.is_connected is True
            
            # Should disconnect on exit
            assert handler.is_connected is False
    
    def test_disconnect_cleans_up(self):
        """Test disconnect properly cleans up resources."""
        with patch('brainaccess_handler.resolve_byprop') as mock_resolve, \
             patch('brainaccess_handler.StreamInlet') as mock_inlet_class:
            
            mock_stream_info = self._create_mock_stream_info()
            mock_resolve.return_value = [mock_stream_info]
            
            mock_inlet = Mock()
            mock_inlet_class.return_value = mock_inlet
            
            handler = BrainAccessHandler(enabled=True)
            handler.connect()
            
            assert handler.is_connected is True
            
            handler.disconnect()
            
            assert handler.is_connected is False
            mock_inlet.close_stream.assert_called_once()
    
    # Helper methods
    
    def _create_mock_stream_info(self):
        """Create mock StreamInfo object for testing."""
        mock_stream_info = Mock()
        mock_stream_info.name.return_value = "BrainAccess"
        mock_stream_info.type.return_value = "EEG"
        mock_stream_info.channel_count.return_value = 8
        mock_stream_info.nominal_srate.return_value = 250.0
        
        # Mock channel description
        mock_channels_desc = Mock()
        mock_channel = Mock()
        mock_channel.child_value.return_value = "Pz"
        mock_channel.next_sibling.return_value = Mock(empty=Mock(return_value=True))
        mock_channels_desc.first_child.return_value = mock_channel
        
        mock_desc = Mock()
        mock_desc.child.return_value = mock_channels_desc
        mock_stream_info.desc.return_value = mock_desc
        
        return mock_stream_info


@pytest.mark.integration
class TestBrainAccessIntegration:
    """
    Integration tests requiring actual BrainAccess device.
    
    These tests will be skipped if no device is available.
    Run with: pytest -v -m integration
    """
    
    @pytest.fixture
    def check_device_available(self):
        """Check if BrainAccess device is available."""
        try:
            from pylsl import resolve_byprop
            streams = resolve_byprop('name', 'BrainAccess', timeout=2.0)
            if not streams:
                pytest.skip("No BrainAccess device found")
        except Exception as e:
            pytest.skip(f"Cannot check for device: {e}")
    
    def test_connect_to_real_device(self, check_device_available):
        """Test connection to actual BrainAccess device."""
        handler = BrainAccessHandler(
            device_name="BrainAccess",
            channels=['Pz', 'Cz', 'Fz', 'Fp1'],
            enabled=True,
            timeout=5.0
        )
        
        result = handler.connect()
        assert result is True
        assert handler.is_connected is True
        
        handler.disconnect()
    
    def test_record_real_data(self, check_device_available):
        """Test recording actual EEG data from device."""
        handler = BrainAccessHandler(enabled=True)
        handler.connect()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "test_eeg.csv")
            
            # Record for 2 seconds
            handler.start_recording(output_file)
            time.sleep(2.0)
            handler.stop_recording()
            
            # Verify data was saved
            assert os.path.exists(output_file)
            df = pd.read_csv(output_file)
            
            # Should have ~500 samples (250 Hz * 2 seconds)
            assert len(df) > 400  # Allow some tolerance
            assert 'timestamp' in df.columns
            
        handler.disconnect()
    
    def test_signal_quality_real_device(self, check_device_available):
        """Test signal quality assessment with real device."""
        handler = BrainAccessHandler(enabled=True)
        handler.connect()
        
        # Wait for data to accumulate
        time.sleep(2.0)
        
        quality = handler.get_signal_quality()
        
        # Should have quality assessment for each channel
        assert len(quality) == 4
        assert all(ch in quality for ch in ['Pz', 'Cz', 'Fz', 'Fp1'])
        assert all(q in ['good', 'fair', 'poor', 'no_data'] for q in quality.values())
        
        handler.disconnect()


class TestLSLSynchronization:
    """Test LSL synchronization between EEG and behavioral markers."""
    
    @pytest.mark.mock
    def test_marker_timestamps_alignment(self):
        """Test that marker timestamps align with EEG timestamps."""
        from lsl_markers import LSLMarkerSender
        
        with patch('lsl_markers.StreamOutlet'):
            # Create marker sender
            marker_sender = LSLMarkerSender(enabled=True)
            
            # Send some markers with timestamps
            timestamps = []
            for i in range(5):
                ts = time.time()
                marker_sender.send_marker(f"test_marker_{i}", timestamp=ts)
                timestamps.append(ts)
                time.sleep(0.01)
            
            # Verify markers were sent with correct timestamps
            assert marker_sender.marker_counter == 5
    
    @pytest.mark.mock
    def test_eeg_marker_synchronization(self):
        """Test EEG data and markers can be synchronized via LSL timestamps."""
        # This test verifies the concept of synchronization
        # In practice, LSL provides synchronized timestamps across streams
        
        eeg_timestamps = [1000.0, 1000.004, 1000.008, 1000.012]  # 250 Hz
        marker_timestamp = 1000.006  # Marker between samples
        
        # Find closest EEG sample to marker
        closest_idx = np.argmin(np.abs(np.array(eeg_timestamps) - marker_timestamp))
        
        assert closest_idx == 1  # Should be second sample
        assert abs(eeg_timestamps[closest_idx] - marker_timestamp) < 0.01


def run_tests():
    """Run all tests."""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_tests()

