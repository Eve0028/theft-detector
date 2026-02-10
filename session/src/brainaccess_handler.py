"""
BrainAccess EEG Device Handler (Optimized)
===========================================

Optimized handler using native BrainAccess SDK annotations.

Key features:
- Native SDK annotations (eeg_manager.annotate())
- Minimal logging overhead during recording
- Direct FIF export with embedded markers
- Optimized data collection callback

References
----------
BrainAccess API: https://www.brainaccess.ai/documentation/python-api/3.6.1/

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
    from brainaccess.core.eeg_manager import EEGManager
    from brainaccess.core import scan as ba_scan
    from brainaccess.core import init as ba_init
    from brainaccess.core import close as ba_close
    from brainaccess.utils.exceptions import BrainAccessException
    BRAINACCESS_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    BRAINACCESS_AVAILABLE = False
    logger.error("BrainAccess SDK not available: %s", exc)


class BrainAccessHandler:
    """
    Optimized handler for BrainAccess EEG device.
    
    Uses native SDK annotations for minimal overhead.
    """
    
    def __init__(
        self,
        channels: Optional[List[str]] = None,
        channel_mapping: Optional[Dict[str, int]] = None,
        sampling_rate: int = 250,
        buffer_size: int = 360,
        enabled: bool = True,
        verbose: bool = False
    ):
        self.channels = channels or ['P3', 'P4', 'C3', 'C4']
        self.custom_channel_mapping = channel_mapping
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.enabled = enabled and BRAINACCESS_AVAILABLE
        self.verbose = verbose
        self.core_initialized = False
        
        # Logger with configurable verbosity
        self.logger = logging.getLogger(__name__)
        if not verbose:
            self.logger.setLevel(logging.ERROR)
        
        # BrainAccess components
        self.eeg_manager: Optional[EEGManager] = None
        
        # Recording state
        self.is_connected = False
        self.is_recording = False
        self.recording_thread: Optional[threading.Thread] = None
        self.stop_recording_event = threading.Event()
        
        # Data storage (minimal overhead)
        max_samples = sampling_rate * buffer_size
        self.eeg_data: deque = deque(maxlen=max_samples)
        self.timestamps: deque = deque(maxlen=max_samples)
        self.output_file: Optional[Path] = None
        
        # Manual annotation buffer (fallback if SDK doesn't work)
        self.manual_annotations: List[Tuple[float, str]] = []
        
        # Channel mapping
        self.channel_mapping: Dict[str, int] = {}
        self.chunk_index_map: Dict[str, int] = {}
        
        # Lock for thread-safe operations (minimal usage)
        self._callback_lock = threading.Lock()
        
        if not self.enabled:
            self.logger.error("BrainAccess handler disabled")
    
    def connect(self) -> bool:
        """Connect to BrainAccess device and start streaming."""
        if not self.enabled:
            return False
        
        if self.is_connected:
            return True
        
        try:
            # Initialize core
            if not self.core_initialized:
                try:
                    ba_init()
                    self.core_initialized = True
                except BrainAccessException as e:
                    if "already initialized" not in str(e).lower():
                        raise
                    self.core_initialized = True
            
            # Create EEG manager
            self.eeg_manager = EEGManager()
            
            # Scan and connect
            devices = ba_scan()
            if not devices:
                self.logger.error("No BrainAccess devices found")
                return False
            
            device = devices[0]
            self.eeg_manager.connect(device.name)
            
            # Setup channels
            self._setup_channels()
            self.eeg_manager.set_callback_chunk(self._on_chunk)
            
            # Load config and start stream
            try:
                self.eeg_manager.load_config()
            except Exception:
                pass  # Non-critical
            
            self.eeg_manager.start_stream()
            self._build_chunk_index_map()
            
            self.is_connected = True
            if self.verbose:
                self.logger.info("BrainAccess device connected")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False
    
    def _setup_channels(self) -> None:
        """Configure EEG channels (minimal logging)."""
        if self.eeg_manager is None:
            return
        
        try:
            # Use custom or standard mapping
            if self.custom_channel_mapping:
                electrode_to_channel = self.custom_channel_mapping
                if self.verbose:
                    self.logger.info(f"Using custom channel mapping: {electrode_to_channel}")
            else:
                electrode_to_channel = {
                    'F3': 0, 'F4': 1, 'C3': 2, 'C4': 3,
                    'P3': 4, 'P4': 5, 'O1': 6, 'O2': 7
                }
                if self.verbose:
                    self.logger.info("Using default 8-channel mapping")
            
            # Map requested channels
            self.channel_mapping.clear()
            for electrode in self.channels:
                if electrode in electrode_to_channel:
                    self.channel_mapping[electrode] = electrode_to_channel[electrode]
            
            if self.verbose:
                self.logger.info(f"Channel mapping for {self.channels}: {self.channel_mapping}")
            
            # Enable channels
            physical_indices = set(self.channel_mapping.values())
            if self.verbose:
                self.logger.info(f"Enabling physical channels: {physical_indices}")
            
            for idx in physical_indices:
                try:
                    self.eeg_manager.set_channel_enabled(idx, True)
                    if self.verbose:
                        self.logger.info(f"  Channel {idx} enabled")
                except Exception as e:
                    if self.verbose:
                        self.logger.warning(f"  Failed to enable channel {idx}: {e}")
            
        except Exception as e:
            self.logger.error(f"Channel setup error: {e}")
    
    def _build_chunk_index_map(self) -> None:
        """Build chunk index mapping after stream starts."""
        if self.eeg_manager is None:
            return
        
        self.chunk_index_map.clear()
        if self.verbose:
            self.logger.info("Building chunk index map...")
        
        for ch in self.channels:
            phys_idx = self.channel_mapping.get(ch, -1)
            if phys_idx >= 0:
                try:
                    chunk_idx = self.eeg_manager.get_channel_index(phys_idx)
                    self.chunk_index_map[ch] = chunk_idx
                    if self.verbose:
                        self.logger.info(f"  {ch} (physical {phys_idx}) -> chunk index {chunk_idx}")
                except Exception as e:
                    if self.verbose:
                        self.logger.warning(f"  {ch} (physical {phys_idx}) failed: {e}")
        
        if self.verbose:
            self.logger.info(f"Final chunk_index_map: {self.chunk_index_map}")
    
    def annotate(self, marker: str) -> bool:
        """
        Send annotation using native BrainAccess SDK.
        
        This is the PREFERRED method for markers - uses SDK's internal
        annotation system with minimal overhead.
        
        Also stores in manual buffer as fallback.
        
        Parameters
        ----------
        marker : str
            Marker description (e.g., "S1_onset_probe", "fixation")
            
        Returns
        -------
        bool
            True if annotation sent successfully
        """
        if not self.is_connected:
            return False
        
        # Store in manual buffer (fallback)
        timestamp = time.time()
        self.manual_annotations.append((timestamp, marker))
        
        # Try SDK annotation
        if self.eeg_manager is not None:
            try:
                self.eeg_manager.annotate(marker)
                return True
            except Exception as e:
                if self.verbose:
                    self.logger.error(f"SDK annotation failed: {e}")
                # Continue - manual buffer will be used
        
        return True  # Success via manual buffer
    
    def start_recording(self, output_file: str) -> bool:
        """Start recording EEG data."""
        if not self.is_connected:
            self.logger.error("Device not connected")
            return False
        
        if self.is_recording:
            return True
        
        try:
            self.output_file = Path(output_file)
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Clear buffers
            self.eeg_data.clear()
            self.timestamps.clear()
            self.manual_annotations.clear()
            
            # Start recording thread (passive - data comes via callback)
            self.stop_recording_event.clear()
            self.recording_thread = threading.Thread(
                target=self._recording_loop,
                daemon=True
            )
            self.recording_thread.start()
            
            self.is_recording = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start recording: {e}")
            return False
    
    def _recording_loop(self) -> None:
        """Passive recording loop - data collected via callback."""
        try:
            while not self.stop_recording_event.is_set():
                time.sleep(0.1)
        except Exception as e:
            self.logger.error(f"Recording loop error: {e}")
    
    def _on_chunk(self, chunk_arrays, chunk_size: int) -> None:
        """
        Optimized chunk callback - minimal overhead.
        
        Data arrives here from BrainAccess SDK.
        ALWAYS collects data to circular buffer (for signal quality checks).
        """
        try:
            base_timestamp = time.time()
            sample_interval = 1.0 / self.sampling_rate
            
            # Accumulate samples (always, not only when recording)
            # Buffer is circular (deque with maxlen) so no overflow
            for i in range(chunk_size):
                timestamp = base_timestamp + (i * sample_interval)
                
                # Build sample vector
                sample_values = []
                for ch in self.channels:
                    chunk_idx = self.chunk_index_map.get(ch, -1)
                    if 0 <= chunk_idx < len(chunk_arrays) and i < len(chunk_arrays[chunk_idx]):
                        sample_values.append(chunk_arrays[chunk_idx][i])
                    else:
                        sample_values.append(0.0)
                
                self.eeg_data.append(sample_values)
                self.timestamps.append(timestamp)
        
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Chunk callback error: {e}")
    
    def stop_recording(self) -> bool:
        """Stop recording and save data."""
        if not self.is_recording:
            return False
        
        try:
            # Signal stop
            self.stop_recording_event.set()
            
            # Wait for thread
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=5.0)
            
            self.is_recording = False
            
            # Save data
            if len(self.eeg_data) > 0:
                # Auto-detect format
                if self.output_file and self.output_file.suffix.lower() == '.fif':
                    success = self._save_to_fif()
                else:
                    success = self._save_to_csv()
                
                return success
            else:
                self.logger.error("No data recorded")
                return False
                
        except Exception as e:
            self.logger.error(f"Stop recording error: {e}")
            return False
    
    def _save_to_csv(self) -> bool:
        """Save EEG data to CSV (legacy format)."""
        if self.output_file is None:
            return False
        
        try:
            data_array = np.array(list(self.eeg_data))
            timestamps_array = np.array(list(self.timestamps))
            
            df = pd.DataFrame(data_array, columns=self.channels)
            df.insert(0, 'timestamp', timestamps_array)
            
            df.to_csv(self.output_file, index=False)
            
            if self.verbose:
                self.logger.info(f"Saved {len(df)} samples to CSV")
            
            return True
            
        except Exception as e:
            self.logger.error(f"CSV save error: {e}")
            return False
    
    def _save_to_fif(self) -> bool:
        """
        Save EEG data to MNE FIF format with native BrainAccess annotations.
        
        Retrieves annotations from SDK using get_annotations().
        """
        try:
            import mne
        except ImportError:
            self.logger.error("MNE-Python not installed")
            return False
        
        if len(self.eeg_data) == 0:
            return False
        
        try:
            # Convert data
            data_array = np.array(list(self.eeg_data)).T  # (channels, times)
            timestamps_array = np.array(list(self.timestamps))
            
            # Create MNE Info
            ch_types = ['eeg'] * len(self.channels)
            info = mne.create_info(
                ch_names=self.channels,
                sfreq=self.sampling_rate,
                ch_types=ch_types
            )
            
            # Create Raw object
            raw = mne.io.RawArray(data_array, info)
            
            # Get annotations from BrainAccess SDK
            if self.eeg_manager:
                try:
                    sdk_result = self.eeg_manager.get_annotations()
                    
                    # SDK may return dict {"annotations": [...], "timestamps": [...]}
                    # or list of Annotation objects
                    sdk_annotations = None
                    sdk_timestamps = None
                    
                    if isinstance(sdk_result, dict):
                        # Dict format
                        sdk_annotations = sdk_result.get('annotations', [])
                        sdk_timestamps = sdk_result.get('timestamps', [])
                        if self.verbose:
                            self.logger.info(f"SDK returned dict: {len(sdk_annotations)} annotations")
                    elif isinstance(sdk_result, (list, tuple)):
                        # List format
                        sdk_annotations = sdk_result
                        if self.verbose:
                            self.logger.info(f"SDK returned list: {len(sdk_annotations)} annotations")
                    else:
                        if self.verbose:
                            self.logger.warning(f"SDK returned unexpected type: {type(sdk_result)}")
                    
                    if sdk_annotations and len(sdk_annotations) > 0:
                        # Convert SDK annotations to MNE format
                        first_time = timestamps_array[0]
                        
                        onsets = []
                        descriptions = []
                        
                        # Case 1: We have separate timestamps array
                        if sdk_timestamps and len(sdk_timestamps) == len(sdk_annotations):
                            for i, ann in enumerate(sdk_annotations):
                                try:
                                    timestamp = float(sdk_timestamps[i])
                                    desc = str(ann) if isinstance(ann, str) else str(ann.get('annotation', ann)) if isinstance(ann, dict) else str(ann)
                                    onsets.append(timestamp - first_time)
                                    descriptions.append(desc)
                                except Exception as e:
                                    if self.verbose:
                                        self.logger.warning(f"Failed to process annotation {i}: {e}")
                                    continue
                        
                        # Case 2: Annotations contain timestamps themselves
                        else:
                            for ann in sdk_annotations:
                                # SDK may return different structures:
                                # - Annotation object with .time and .annotation
                                # - Tuple (time, description)
                                # - Dict {"time": ..., "annotation": ...}
                                # - String (just description, no timestamp)
                                
                                try:
                                    # Try object attributes first
                                    if hasattr(ann, 'time') and hasattr(ann, 'annotation'):
                                        onsets.append(ann.time - first_time)
                                        descriptions.append(ann.annotation)
                                    # Try tuple
                                    elif isinstance(ann, (tuple, list)) and len(ann) >= 2:
                                        onsets.append(float(ann[0]) - first_time)
                                        descriptions.append(str(ann[1]))
                                    # Try dict
                                    elif isinstance(ann, dict):
                                        onsets.append(float(ann.get('time', 0)) - first_time)
                                        descriptions.append(str(ann.get('annotation', ann.get('description', 'marker'))))
                                    # String only - skip (no timestamp available)
                                    elif isinstance(ann, str):
                                        if self.verbose:
                                            self.logger.warning(f"Skipping string-only annotation: {ann}")
                                        continue
                                    else:
                                        if self.verbose:
                                            self.logger.warning(f"Unknown annotation type: {type(ann)}")
                                        continue
                                except Exception as conv_err:
                                    if self.verbose:
                                        self.logger.warning(f"Failed to convert annotation: {conv_err}")
                                    continue
                        
                        if onsets:  # Only if we have valid annotations
                            durations = [0.0] * len(onsets)
                            
                            # Use orig_time=None for relative onsets (avoids meas_date conflict)
                            mne_annotations = mne.Annotations(
                                onset=onsets,
                                duration=durations,
                                description=descriptions,
                                orig_time=None  # Relative to recording start
                            )
                            
                            raw.set_annotations(mne_annotations)
                            
                            if self.verbose:
                                self.logger.info(f"Embedded {len(onsets)} annotations from SDK")
                        else:
                            if self.verbose:
                                self.logger.warning("No valid annotations from SDK, trying manual buffer")
                            # Fall through to manual buffer
                
                except Exception as e:
                    self.logger.warning(f"Could not retrieve SDK annotations: {e}, using manual buffer")
            
            # Fallback: Use manual annotation buffer if SDK failed or returned nothing
            if not raw.annotations or len(raw.annotations) == 0:
                if self.manual_annotations:
                    if self.verbose:
                        self.logger.info(f"Using manual annotation buffer ({len(self.manual_annotations)} annotations)")
                    
                    first_time = timestamps_array[0]
                    onsets = [ann[0] - first_time for ann in self.manual_annotations]
                    descriptions = [ann[1] for ann in self.manual_annotations]
                    durations = [0.0] * len(onsets)
                    
                    # Use orig_time=None for relative onsets (avoids meas_date conflict)
                    mne_annotations = mne.Annotations(
                        onset=onsets,
                        duration=durations,
                        description=descriptions,
                        orig_time=None  # Relative to recording start
                    )
                    
                    raw.set_annotations(mne_annotations)
                    self.logger.info(f"Embedded {len(onsets)} annotations from manual buffer")
                else:
                    self.logger.warning("No annotations available (neither SDK nor manual buffer)")
            
            # Save FIF
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            raw.save(self.output_file, overwrite=True, verbose=False)
            
            if self.verbose:
                self.logger.info(f"Saved {len(self.eeg_data)} samples to FIF")
            
            return True
            
        except Exception as e:
            self.logger.error(f"FIF save error: {e}")
            return False
    
    def get_signal_quality(self) -> Dict[str, str]:
        """Assess signal quality for each channel."""
        if len(self.eeg_data) < self.sampling_rate:
            return {ch: 'no_data' for ch in self.channels}
        
        try:
            recent_data = list(self.eeg_data)[-self.sampling_rate:]
            data_array = np.array(recent_data)
            
            quality = {}
            for idx, ch in enumerate(self.channels):
                if idx >= data_array.shape[1]:
                    quality[ch] = 'no_data'
                    continue
                
                channel_data = data_array[:, idx]
                std_dev = np.std(channel_data)
                peak_to_peak = np.ptp(channel_data)
                
                # Quality thresholds
                if std_dev < 1.0 or peak_to_peak < 5.0:
                    quality[ch] = 'poor'
                elif std_dev > 100.0 or peak_to_peak > 500.0:
                    quality[ch] = 'poor'
                elif std_dev > 50.0 or peak_to_peak > 200.0:
                    quality[ch] = 'fair'
                else:
                    quality[ch] = 'good'
            
            return quality
            
        except Exception as e:
            self.logger.error(f"Quality check error: {e}")
            return {ch: 'no_data' for ch in self.channels}
    
    def get_latest_sample(self) -> Optional[Dict[str, float]]:
        """Get most recent EEG sample."""
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
        except Exception:
            return None
    
    def get_recent_data(self, seconds: float = 5.0) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Get recent EEG data window.
        
        Parameters
        ----------
        seconds : float
            Window length in seconds
            
        Returns
        -------
        tuple
            (data array [samples, channels], channel names)
        """
        if not self.eeg_data:
            return None, self.channels
        
        try:
            n_samples = int(seconds * self.sampling_rate)
            n_available = len(self.eeg_data)
            n = min(n_samples, n_available)
            
            data_array = np.array(list(self.eeg_data)[-n:])
            return data_array, list(self.channels)
        except Exception:
            return None, self.channels
    
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
                except Exception:
                    pass
                
                self.eeg_manager.disconnect()
                self.eeg_manager = None
            
            self.is_connected = False
            
            # Close core
            if self.core_initialized and BRAINACCESS_AVAILABLE:
                try:
                    ba_close()
                except Exception:
                    pass
            
        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")
    
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
