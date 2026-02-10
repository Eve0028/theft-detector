"""
Live EEG stream for SSVEP app using BrainAccess device.

Loads channels and channel_mapping from config; buffers recent samples
for real-time SSVEP analysis. Standalone (no dependency on session package).
"""

import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    from brainaccess.core.eeg_manager import EEGManager
    from brainaccess.core import scan as ba_scan
    from brainaccess.core import init as ba_init
    from brainaccess.core import close as ba_close
    from brainaccess.utils.exceptions import BrainAccessException
    BRAINACCESS_AVAILABLE = True
except Exception:
    BRAINACCESS_AVAILABLE = False

# Default physical channel mapping (BrainAccess CAP cable layout)
DEFAULT_CHANNEL_MAPPING = {
    "F3": 0, "F4": 1, "C3": 2, "C4": 3,
    "P3": 4, "P4": 5, "O1": 6, "O2": 7,
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load SSVEP config YAML. Returns full config dict."""
    path: Path = (
        Path(config_path) if config_path is not None
        else Path(__file__).parent / "config.yaml"
    )
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class EEGStream:
    """
    Live EEG buffer from BrainAccess. Config-driven channels and mapping.
    """

    def __init__(
        self,
        channels: List[str],
        channel_mapping: Optional[Dict[str, int]] = None,
        sampling_rate: int = 250,
        buffer_seconds: int = 60,
        raw_to_uv_scale: float = 1.0,
    ) -> None:
        self.channels = list(channels)
        self.sampling_rate = sampling_rate
        self.raw_to_uv_scale = raw_to_uv_scale
        self._mapping = channel_mapping or DEFAULT_CHANNEL_MAPPING
        self.channel_mapping: Dict[str, int] = {}
        self.chunk_index_map: Dict[str, int] = {}
        max_samples = sampling_rate * buffer_seconds
        self._data: deque = deque(maxlen=max_samples)
        self._timestamps: deque = deque(maxlen=max_samples)
        self._eeg_manager: Optional[EEGManager] = None
        self._core_initialized = False
        self.is_connected = False

        for ch in self.channels:
            if ch in self._mapping:
                self.channel_mapping[ch] = self._mapping[ch]

    def connect(self) -> bool:
        """Connect to first BrainAccess device and start streaming."""
        if not BRAINACCESS_AVAILABLE:
            return False
        if self.is_connected:
            return True
        try:
            if not self._core_initialized:
                try:
                    ba_init()
                    self._core_initialized = True
                except BrainAccessException as e:
                    if "already initialized" not in str(e).lower():
                        raise
                    self._core_initialized = True
            self._eeg_manager = EEGManager()
            devices = ba_scan()
            if not devices:
                return False
            self._eeg_manager.connect(devices[0].name)
            self._setup_channels()
            self._eeg_manager.set_callback_chunk(self._on_chunk)
            try:
                self._eeg_manager.load_config()
            except Exception:
                pass
            self._eeg_manager.start_stream()
            self._build_chunk_index_map()
            self.is_connected = True
            return True
        except Exception:
            return False

    def _setup_channels(self) -> None:
        if self._eeg_manager is None:
            return
        for idx in set(self.channel_mapping.values()):
            try:
                self._eeg_manager.set_channel_enabled(idx, True)
            except Exception:
                pass

    def _build_chunk_index_map(self) -> None:
        if self._eeg_manager is None:
            return
        self.chunk_index_map.clear()
        for ch in self.channels:
            phys = self.channel_mapping.get(ch, -1)
            if phys >= 0:
                try:
                    self.chunk_index_map[ch] = self._eeg_manager.get_channel_index(phys)
                except Exception:
                    pass

    def _on_chunk(self, chunk_arrays, chunk_size: int) -> None:
        base_t = time.time()
        dt = 1.0 / self.sampling_rate
        for i in range(chunk_size):
            t = base_t + i * dt
            row = []
            for ch in self.channels:
                ci = self.chunk_index_map.get(ch, -1)
                if 0 <= ci < len(chunk_arrays) and i < len(chunk_arrays[ci]):
                    row.append(chunk_arrays[ci][i])
                else:
                    row.append(0.0)
            self._data.append(row)
            self._timestamps.append(t)

    def get_recent(self, seconds: float) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Return (samples, channels) for last `seconds` of data.
        Shape: (n_samples, n_channels), channel order = self.channels.
        If raw_to_uv_scale != 1.0, values are converted to microvolts (µV).
        """
        if not self._data:
            return None, self.channels
        n = min(int(seconds * self.sampling_rate), len(self._data))
        if n <= 0:
            return None, self.channels
        arr = np.array(list(self._data)[-n:], dtype=float)
        if self.raw_to_uv_scale != 1.0:
            arr = arr * self.raw_to_uv_scale
        return arr, list(self.channels)

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        try:
            if self._eeg_manager:
                try:
                    if self._eeg_manager.is_streaming():
                        self._eeg_manager.stop_stream()
                except Exception:
                    pass
                self._eeg_manager.disconnect()
                self._eeg_manager = None
            self.is_connected = False
            if self._core_initialized and BRAINACCESS_AVAILABLE:
                try:
                    ba_close()
                except Exception:
                    pass
        except Exception:
            pass

    def __enter__(self) -> "EEGStream":
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.disconnect()
