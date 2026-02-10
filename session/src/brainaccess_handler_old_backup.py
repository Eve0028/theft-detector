"""
BrainAccess EEG Device Handler
================================

Module for managing BrainAccess MINI device connection and EEG data acquisition
using the official BrainAccess Python API.

Usage
-----
::

    from brainaccess_handler import BrainAccessHandler

    # Initialize handler
    handler = BrainAccessHandler(
        channels=['Pz', 'Cz', 'Fz', 'Fp1'],
        sampling_rate=250
    )

    # Connect and start recording
    handler.connect()
    handler.start_recording("output_file.csv")

    # ... run experiment ...

    # Stop recording
    handler.stop_recording()
    handler.disconnect()

References
----------
BrainAccess API: https://www.brainaccess.ai/documentation/python-api/3.6.0/

"""

import logging
import threading
import time
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from collections import deque
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    # EEGManager is defined in brainaccess.core.eeg_manager (not re-exported)
    from brainaccess.core.eeg_manager import EEGManager
    from brainaccess.utils import acquisition
    from brainaccess.core import scan as ba_scan
    from brainaccess.core import init as ba_init
    from brainaccess.core import close as ba_close
    from brainaccess.utils.exceptions import BrainAccessException
    BRAINACCESS_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    BRAINACCESS_AVAILABLE = False
    msg = "brainaccess not available. Install: pip install brainaccess"
    logger.warning(msg)
    # Log the underlying import/load failure for easier troubleshooting
    logger.exception("BrainAccess import failed", exc_info=exc)


class BrainAccessHandler:
    """
    Handler for BrainAccess MINI EEG device using official API.

    Parameters
    ----------
    channels : list of str, optional
        List of electrode channel names to record
    sampling_rate : int, optional
        Sampling rate in Hz (default: 250)
    buffer_size : int, optional
        Size of data buffer in seconds (default: 360 for 1 hour)
    enabled : bool, optional
        Whether device is enabled (default: True)
    """

    def __init__(
        self,
        channels: Optional[List[str]] = None,
        channel_mapping: Optional[Dict[str, int]] = None,
        sampling_rate: int = 250,
        buffer_size: int = 360,
        enabled: bool = True
    ):
        self.channels = channels or ['Pz', 'Cz', 'Fz', 'Fp1']
        self.custom_channel_mapping = channel_mapping
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.enabled = enabled and BRAINACCESS_AVAILABLE
        self.core_initialized = False

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # BrainAccess components
        self.eeg_manager: Optional[EEGManager] = None

        # Recording state
        self.is_connected = False
        self.is_recording = False
        self.recording_thread: Optional[threading.Thread] = None
        self.stop_recording_event = threading.Event()

        # Data storage
        max_samples = sampling_rate * buffer_size
        self.eeg_data: deque = deque(maxlen=max_samples)
        self.timestamps: deque = deque(maxlen=max_samples)
        self.output_file: Optional[Path] = None
        
        # Marker/annotation storage for MNE export
        # List of tuples: (timestamp, description)
        self.annotations: List[tuple] = []

        # Channel mapping (electrode name -> physical channel index)
        self.channel_mapping: Dict[str, int] = {}
        
        # Chunk index mapping (electrode name -> index in chunk_arrays)
        self.chunk_index_map: Dict[str, int] = {}

        # Chunk callback guard
        self._callback_lock = threading.Lock()

        if not self.enabled:
            if not BRAINACCESS_AVAILABLE:
                self.logger.warning(
                    "BrainAccess handler disabled: library not available"
                )
            else:
                self.logger.info(
                    "BrainAccess handler disabled in configuration"
                )

    def connect(self) -> bool:
        """
        Connect to BrainAccess device via Bluetooth.

        Returns
        -------
        bool
            True if connection successful, False otherwise
        """
        if not self.enabled:
            self.logger.warning("BrainAccess handler is disabled")
            return False

        if self.is_connected:
            self.logger.warning("Already connected to BrainAccess device")
            return True

        try:
            self.logger.info("Connecting to BrainAccess device...")

            # Initialize core library once
            if not self.core_initialized:
                try:
                    ba_init()
                    self.core_initialized = True
                    self.logger.info("BrainAccess core initialized")
                except BrainAccessException as init_exc:
                    if "already initialized" in str(init_exc).lower():
                        self.logger.info("BrainAccess core already initialized")
                        self.core_initialized = True
                    else:
                        raise

            # Create EEG manager
            self.eeg_manager = EEGManager()

            # Scan for devices using core scan helper
            self.logger.info("Scanning for BrainAccess devices...")
            devices = ba_scan()

            if not devices:
                self.logger.error(
                    "No BrainAccess devices found. "
                    "Ensure device is powered on and in range."
                )
                return False

            # Use first device found
            device = devices[0]
            self.logger.info(
                f"Found device: {device.name} "
                f"(MAC: {device.mac_address})"
            )

            # Connect to device by name (EEGManager expects device identifier string)
            self.eeg_manager.connect(device.name)
            self.logger.info("Connected to device")

            # Setup channels and stream callbacks
            self._setup_channels()
            self.eeg_manager.set_callback_chunk(self._on_chunk)

            # Apply config and start stream
            try:
                self.eeg_manager.load_config()
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Could not load config before stream: %s", exc)

            self.eeg_manager.start_stream()
            self.logger.info("EEG stream started")
            
            # Build chunk index mapping after stream starts
            self._build_chunk_index_map()

            self.is_connected = True
            self.logger.info("BrainAccess device connected successfully")

            return True

        except Exception as e:
            self.logger.error(
                f"Failed to connect to BrainAccess device: {e}",
                exc_info=True
            )
            return False

    def _setup_channels(self) -> None:
        """
        Configure EEG channels based on requested electrode positions.

        BrainAccess CAP physical channel mapping (via cable length):
        - REF: Fp1 position (35cm L, 30cm S-M) - Reference electrode
        - BIAS: Fp2 position (35cm L, 30cm S-M) - Bias electrode
        - Channel 0: F3 (30cm L, 25cm S-M)
        - Channel 1: F4 (30cm L, 25cm S-M)
        - Channel 2: C3 (20cm)
        - Channel 3: C4 (20cm)
        - Channel 4: P3 (15cm)
        - Channel 5: P4 (15cm)
        - Channel 6: O1 (10cm)
        - Channel 7: O2 (10cm)
        """
        if self.eeg_manager is None:
            return

        try:
            if self.custom_channel_mapping:
                electrode_to_channel = self.custom_channel_mapping
                self.logger.info("Using custom channel mapping from config")
            else:
                electrode_to_channel = {
                    'F3': 0,
                    'F4': 1,
                    'C3': 2,
                    'C4': 3,
                    'P3': 4,
                    'P4': 5,
                    'O1': 6,
                    'O2': 7,
                }
                self.logger.info(
                    "Using standard BrainAccess CAP mapping (F3,F4,C3,C4...)"
                )

            self.logger.info(f"Requested electrodes: {self.channels}")

            # Create channel mapping for requested electrodes
            self.channel_mapping.clear()
            for electrode in self.channels:
                if electrode in electrode_to_channel:
                    channel_idx = electrode_to_channel[electrode]
                    self.channel_mapping[electrode] = channel_idx
                    self.logger.info(f"  {electrode} -> Channel {channel_idx}")
                else:
                    self.logger.warning(
                        "Electrode %s not in channel mapping. Available: %s",
                        electrode,
                        list(electrode_to_channel.keys()),
                    )

            # Enable only the channels we need
            for idx in set(self.channel_mapping.values()):
                try:
                    self.eeg_manager.set_channel_enabled(idx, True)
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning(
                        "Could not enable channel %s: %s", idx, exc
                    )

            self.logger.info(f"Final channel mapping: {self.channel_mapping}")

            # Warn if we didn't map all requested channels
            missing = set(self.channels) - set(self.channel_mapping.keys())
            if missing:
                self.logger.error(
                    "Could not map electrodes: %s. Check electrode names and cable configuration.",
                    missing,
                )

        except Exception as e:
            self.logger.error(f"Error setting up channels: {e}")
    
    def _build_chunk_index_map(self) -> None:
        """
        Build mapping from electrode names to chunk array indices.
        Must be called after load_config() and start_stream().
        """
        if self.eeg_manager is None:
            return
        
        self.chunk_index_map.clear()
        for ch in self.channels:
            phys_idx = self.channel_mapping.get(ch, -1)
            if phys_idx >= 0:
                try:
                    chunk_idx = self.eeg_manager.get_channel_index(phys_idx)
                    self.chunk_index_map[ch] = chunk_idx
                    self.logger.debug(
                        "%s: physical=%s -> chunk_index=%s", ch, phys_idx, chunk_idx
                    )
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning(
                        "Channel %s (phys %s) not in stream: %s",
                        ch, phys_idx, exc
                    )

    def start_recording(self, output_file: str) -> bool:
        """
        Start recording EEG data to file.

        Parameters
        ----------
        output_file : str
            Path to output CSV file for EEG data

        Returns
        -------
        bool
            True if recording started successfully, False otherwise
        """
        if not self.is_connected:
            self.logger.error("Cannot start recording: device not connected")
            return False

        if self.is_recording:
            self.logger.warning("Recording already in progress")
            return True

        try:
            self.output_file = Path(output_file)
            self.output_file.parent.mkdir(parents=True, exist_ok=True)

            # Clear buffers
            self.eeg_data.clear()
            self.timestamps.clear()
            self.annotations.clear()

            # Start recording thread
            self.stop_recording_event.clear()
            self.recording_thread = threading.Thread(
                target=self._recording_loop,
                daemon=True
            )
            self.recording_thread.start()

            self.is_recording = True
            self.logger.info(f"Started EEG recording to: {self.output_file}")

            return True

        except Exception as e:
            self.logger.error(
                f"Failed to start recording: {e}",
                exc_info=True
            )
            return False

    def _recording_loop(self) -> None:
        """
        Main recording loop running in separate thread.
        Continuously reads data from device and stores in buffer.
        """
        self.logger.debug("Recording loop started (passive; data via callbacks)")
        try:
            while not self.stop_recording_event.is_set():
                time.sleep(0.1)
        except Exception as e:
            self.logger.error(f"Error in recording loop: {e}", exc_info=True)
        finally:
            self.logger.debug("Recording loop stopped")

    def stop_recording(self) -> bool:
        """
        Stop recording and save data to file.

        Returns
        -------
        bool
            True if data saved successfully, False otherwise
        """
        if not self.is_recording:
            self.logger.warning("No recording in progress")
            return False

        try:
            # Signal recording thread to stop
            self.stop_recording_event.set()

            # Wait for thread to finish
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=5.0)

            self.is_recording = False

            # Save data to file
            if len(self.eeg_data) > 0:
                # Auto-detect format from file extension
                if self.output_file and self.output_file.suffix.lower() == '.fif':
                    success = self.save_to_fif(str(self.output_file))
                else:
                    self._save_data()
                    success = True
                
                if success:
                    self.logger.info(
                        f"Recording stopped. Saved {len(self.eeg_data)} samples"
                    )
                return success
            else:
                self.logger.warning("No data recorded")
                return False

        except Exception as e:
            self.logger.error(
                f"Error stopping recording: {e}",
                exc_info=True
            )
            return False

    def _save_data(self) -> None:
        """Save buffered EEG data to CSV file."""
        if self.output_file is None:
            return

        try:
            # Convert deques to numpy arrays
            data_array = np.array(list(self.eeg_data))
            timestamps_array = np.array(list(self.timestamps))

            # Extract data for requested channels only (stored in order)
            available_channels = list(self.channels)
            channel_data = data_array[:, : len(available_channels)]

            # Create DataFrame
            df = pd.DataFrame(channel_data, columns=available_channels)
            df.insert(0, 'timestamp', timestamps_array)

            # Save to CSV
            df.to_csv(self.output_file, index=False)
            self.logger.info(f"Saved {len(df)} samples to {self.output_file}")

        except Exception as e:
            self.logger.error(f"Error saving EEG data: {e}", exc_info=True)

    def get_latest_sample(self) -> Optional[Dict[str, float]]:
        """
        Get the most recent EEG sample.

        Returns
        -------
        dict or None
            Dictionary with channel names as keys and voltages as values,
            plus 'timestamp' key. None if no data available.
        """
        if not self.eeg_data or not self.timestamps:
            return None

        try:
            latest_sample = self.eeg_data[-1]
            latest_timestamp = self.timestamps[-1]

            result = {'timestamp': latest_timestamp}

            for idx, ch in enumerate(self.channels):
                if idx < len(latest_sample):
                    result[ch] = latest_sample[idx]

            return result

        except Exception as e:
            self.logger.error(f"Error getting latest sample: {e}")
            return None

    def get_recent_data(self, seconds: float = 5.0) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Return recent EEG data window.

        Parameters
        ----------
        seconds : float, optional
            Window length in seconds (default: 5.0)

        Returns
        -------
        tuple
            (data array shape [samples, channels], channel order). Data is None if
            buffer is empty.
        """
        if not self.eeg_data:
            return None, self.channels

        try:
            n_samples = int(seconds * self.sampling_rate)
            if n_samples <= 0:
                n_samples = self.sampling_rate
            n_available = len(self.eeg_data)
            n = min(n_samples, n_available)
            data_array = np.array(list(self.eeg_data)[-n:])
            return data_array, list(self.channels)
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Error getting recent data: %s", exc, exc_info=True)
            return None, self.channels

    def _on_chunk(self, chunk_arrays, chunk_size: int) -> None:
        """
        Callback invoked by BrainAccess SDK when a data chunk arrives.
        Accumulates samples into deques.
        
        Note: chunk_arrays contains data ONLY for enabled channels,
        indexed by get_channel_index() - NOT physical channel numbers.
        """
        with self._callback_lock:
            # Base timestamp for this chunk
            base_timestamp = time.time()
            
            # Time interval between samples (e.g., 0.004 s for 250 Hz)
            sample_interval = 1.0 / self.sampling_rate
            
            # Accumulate samples with interpolated timestamps
            for i in range(chunk_size):
                # Interpolate timestamp for each sample in chunk
                timestamp = base_timestamp + (i * sample_interval)
                
                sample_values = []
                for ch in self.channels:
                    chunk_idx = self.chunk_index_map.get(ch, -1)
                    if 0 <= chunk_idx < len(chunk_arrays) and i < len(chunk_arrays[chunk_idx]):
                        sample_values.append(chunk_arrays[chunk_idx][i])
                    else:
                        sample_values.append(0.0)
                self.eeg_data.append(sample_values)
                self.timestamps.append(timestamp)

    def get_signal_quality(self) -> Dict[str, str]:
        """
        Assess signal quality for each channel based on recent data.

        Returns
        -------
        dict
            Dictionary mapping channel names to quality labels:
            'good', 'fair', 'poor', or 'no_data'
        """
        quality = {}

        if len(self.eeg_data) < self.sampling_rate:
            # Not enough data yet
            return {ch: 'no_data' for ch in self.channels}

        try:
            # Get last second of data
            recent_data = list(self.eeg_data)[-self.sampling_rate:]
            data_array = np.array(recent_data)

            for ch in self.channels:
                idx = self.channels.index(ch)
                if idx >= data_array.shape[1]:
                    quality[ch] = 'no_data'
                    continue

                channel_data = data_array[:, idx]

                # Calculate signal metrics
                std_dev = np.std(channel_data)
                peak_to_peak = np.ptp(channel_data)

                # Quality assessment based on typical EEG amplitudes
                if std_dev < 1.0 or peak_to_peak < 5.0:
                    quality[ch] = 'poor'  # Too flat, bad contact
                elif std_dev > 100.0 or peak_to_peak > 500.0:
                    quality[ch] = 'poor'  # Too noisy, artifact
                elif std_dev > 50.0 or peak_to_peak > 200.0:
                    quality[ch] = 'fair'  # Somewhat noisy
                else:
                    quality[ch] = 'good'  # Good signal

            return quality

        except Exception as e:
            self.logger.error(f"Error assessing signal quality: {e}")
            return {ch: 'no_data' for ch in self.channels}
    
    def add_annotation(self, description: str, timestamp: Optional[float] = None) -> None:
        """
        Add an event annotation/marker to the recording.
        
        Parameters
        ----------
        description : str
            Event description (e.g., "S1_onset_probe", "fixation_onset")
        timestamp : float, optional
            Unix timestamp of the event. If None, uses current time.
            
        Notes
        -----
        Annotations are saved when exporting to FIF format.
        For LSL markers, use the separate LSLMarkerSender.
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.annotations.append((timestamp, description))
        self.logger.debug(f"Added annotation: {description} @ {timestamp}")
    
    def save_to_fif(self, output_file: str) -> bool:
        """
        Save EEG data to MNE FIF format with annotations.
        
        Parameters
        ----------
        output_file : str
            Path to output .fif file
            
        Returns
        -------
        bool
            True if saved successfully, False otherwise
            
        Notes
        -----
        Requires MNE-Python to be installed.
        FIF format includes:
        - Continuous EEG data
        - Channel information (names, types, sampling rate)
        - Event annotations/markers
        """
        try:
            import mne
        except ImportError:
            self.logger.error(
                "MNE-Python not installed. Install with: pip install mne"
            )
            return False
        
        if len(self.eeg_data) == 0:
            self.logger.error("No EEG data to save")
            return False
        
        try:
            # Convert data to numpy array (channels x times)
            data_array = np.array(list(self.eeg_data)).T  # Transpose to (n_channels, n_times)
            timestamps_array = np.array(list(self.timestamps))
            
            # Create MNE Info object
            ch_types = ['eeg'] * len(self.channels)
            info = mne.create_info(
                ch_names=self.channels,
                sfreq=self.sampling_rate,
                ch_types=ch_types
            )
            
            # Create Raw object
            raw = mne.io.RawArray(data_array, info)
            
            # Add annotations if any
            if self.annotations:
                # Convert timestamps to seconds relative to first sample
                first_time = timestamps_array[0]
                onsets = [ann[0] - first_time for ann in self.annotations]
                durations = [0.0] * len(onsets)  # Point events
                descriptions = [ann[1] for ann in self.annotations]
                
                annotations = mne.Annotations(
                    onset=onsets,
                    duration=durations,
                    description=descriptions,
                    orig_time=first_time
                )
                raw.set_annotations(annotations)
                
                self.logger.info(f"Added {len(self.annotations)} annotations to FIF")
            
            # Save to file
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            raw.save(output_path, overwrite=True, verbose=False)
            self.logger.info(
                f"Saved {len(self.eeg_data)} samples to FIF: {output_path}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving to FIF: {e}", exc_info=True)
            return False

    def disconnect(self) -> None:
        """Disconnect from BrainAccess device."""
        if not self.is_connected:
            return

        try:
            # Stop recording if active
            if self.is_recording:
                self.stop_recording()

            # Stop stream and disconnect
            if self.eeg_manager:
                try:
                    if self.eeg_manager.is_streaming():
                        self.eeg_manager.stop_stream()
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("Error stopping stream: %s", exc)

                self.eeg_manager.disconnect()
                self.eeg_manager = None

            self.is_connected = False
            self.logger.info("Disconnected from BrainAccess device")

            # Close core library if we initialized it
            if self.core_initialized and BRAINACCESS_AVAILABLE:
                try:
                    ba_close()
                    self.logger.info("BrainAccess core closed")
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("Error closing BrainAccess core: %s", exc)

        except Exception as e:
            self.logger.error(f"Error disconnecting: {e}", exc_info=True)

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'is_connected') and self.is_connected:
            self.disconnect()
