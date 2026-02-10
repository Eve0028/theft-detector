"""
LSL Marker Integration for EEG Synchronization
==============================================

This module provides LSL (Lab Streaming Layer) marker streaming functionality
for synchronizing behavioral events with EEG recordings.

Supports multiple EEG devices:
- BrainAccess Standard Kit
- Muse S
- Any LSL-compatible device

Usage
-----
::

    from lsl_markers import LSLMarkerSender
    
    # Initialize marker sender
    sender = LSLMarkerSender(
        stream_name="ExperimentMarkers",
        stream_type="Markers",
        device_type="brainaccess"
    )
    
    # Send marker
    sender.send_marker("S1_onset_probe", metadata={"trial": 1})
    
    # Close stream
    sender.close()

"""

import time
from typing import Optional, Dict, Any
import logging

# Try BrainAccess Board first (native for BrainAccess devices)
try:
    import brainaccess_board as bb
    BRAINACCESS_BOARD_AVAILABLE = True
except ImportError:
    BRAINACCESS_BOARD_AVAILABLE = False

# Fallback to generic pylsl
try:
    from pylsl import StreamInfo, StreamOutlet
    PYLSL_AVAILABLE = True
except ImportError:
    PYLSL_AVAILABLE = False

LSL_AVAILABLE = BRAINACCESS_BOARD_AVAILABLE or PYLSL_AVAILABLE

if not LSL_AVAILABLE:
    logging.warning("Neither brainaccess_board nor pylsl available. LSL markers will not be sent.")


class LSLMarkerSender:
    """
    Send event markers via LSL for EEG synchronization.
    
    Parameters
    ----------
    stream_name : str
        Name of the LSL marker stream
    stream_type : str, optional
        Type of LSL stream (default: "Markers")
    stream_id : str, optional
        Unique identifier for the stream
    device_type : str, optional
        EEG device type: "brainaccess", "muse_s", or "generic"
    enabled : bool, optional
        Whether to actually send markers (default: True)
    """
    
    def __init__(
        self,
        stream_name: str = "P300_CIT_Markers",
        stream_type: str = "Markers",
        stream_id: str = "p300cit001",
        device_type: str = "brainaccess",
        enabled: bool = True
    ):
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.stream_id = stream_id
        self.device_type = device_type.lower()
        self.enabled = enabled and LSL_AVAILABLE
        self.outlet: Optional[StreamOutlet] = None
        self.ba_stimulation = None  # BrainAccess Board stimulation object
        self.marker_counter = 0
        self.use_brainaccess_board = False
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Decide which LSL backend to use
        if self.device_type == "brainaccess" and BRAINACCESS_BOARD_AVAILABLE:
            self.use_brainaccess_board = True
            self.logger.info("Using brainaccess_board for LSL markers (native)")
        elif PYLSL_AVAILABLE:
            self.use_brainaccess_board = False
            self.logger.info("Using pylsl for LSL markers (generic)")
        else:
            self.enabled = False
            self.logger.error("No LSL backend available!")
        
        if self.enabled:
            self._initialize_stream()
        else:
            if not LSL_AVAILABLE:
                self.logger.warning("LSL not available - markers will not be sent")
            else:
                self.logger.info("LSL markers disabled in configuration")
    
    def _initialize_stream(self) -> None:
        """Initialize LSL outlet stream."""
        try:
            if self.use_brainaccess_board:
                # Use BrainAccess Board native stimulation
                self.ba_stimulation = bb.stimulation_connect(
                    name=self.stream_name
                )
                self.logger.info(
                    f"BrainAccess Board LSL stream initialized: {self.stream_name}"
                )
                self.logger.info(
                    "NOTE: Make sure BrainAccess Board is running and "
                    "the marker stream is connected in Board Configuration tab!"
                )
                
            else:
                # Use generic pylsl
                info = StreamInfo(
                    name=self.stream_name,
                    type=self.stream_type,
                    channel_count=1,
                    nominal_srate=0,  # Irregular sampling rate for markers
                    channel_format='string',
                    source_id=self.stream_id
                )
                
                # Add metadata for device-specific configuration
                channels = info.desc().append_child("channels")
                channels.append_child("channel") \
                    .append_child_value("label", "Markers") \
                    .append_child_value("type", "Marker")
                
                info.desc().append_child_value("device_type", self.device_type)
                info.desc().append_child_value("experiment", "P300_CIT")
                
                # Create outlet
                self.outlet = StreamOutlet(info)
                self.logger.info(
                    f"pylsl stream initialized: {self.stream_name} "
                    f"(type: {self.stream_type}, device: {self.device_type})"
                )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LSL stream: {e}", exc_info=True)
            self.enabled = False
    
    def send_marker(
        self,
        marker: str,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Send an event marker via LSL.
        
        Parameters
        ----------
        marker : str
            Marker label/string to send
        timestamp : float, optional
            Custom timestamp (default: current time)
        metadata : dict, optional
            Additional metadata to append to marker
            
        Returns
        -------
        int
            Marker counter ID (0 if sending failed)
        """
        if not self.enabled:
            return 0
        
        if self.use_brainaccess_board and self.ba_stimulation is None:
            return 0
        
        if not self.use_brainaccess_board and self.outlet is None:
            return 0
        
        try:
            # Increment counter
            self.marker_counter += 1
            
            # Format marker with metadata
            marker_str = marker
            if metadata:
                metadata_str = ",".join([f"{k}={v}" for k, v in metadata.items()])
                marker_str = f"{marker}|{metadata_str}"
            
            # Send marker using appropriate backend
            if self.use_brainaccess_board:
                # BrainAccess Board: send as string
                self.ba_stimulation.annotate(marker_str)
                self.logger.debug(
                    f"Sent marker #{self.marker_counter} [BA Board]: {marker_str}"
                )
            else:
                # pylsl: send with optional timestamp
                if timestamp is None:
                    self.outlet.push_sample([marker_str])
                else:
                    self.outlet.push_sample([marker_str], timestamp)
                self.logger.debug(
                    f"Sent marker #{self.marker_counter} [pylsl]: {marker_str}"
                )
            
            return self.marker_counter
            
        except Exception as e:
            self.logger.error(f"Failed to send marker '{marker}': {e}", exc_info=True)
            return 0
    
    def send_trial_start(self, trial_num: int, block_num: int) -> int:
        """Send trial start marker."""
        return self.send_marker(
            "trial_start",
            metadata={"trial": trial_num, "block": block_num}
        )
    
    def send_fixation_onset(self, trial_num: int) -> int:
        """Send fixation onset marker."""
        return self.send_marker(
            "fixation_onset",
            metadata={"trial": trial_num}
        )
    
    def send_s1_onset(
        self,
        trial_num: int,
        stimulus_type: str,
        stimulus_id: str
    ) -> int:
        """
        Send S1 (image) onset marker.
        
        Parameters
        ----------
        trial_num : int
            Trial number
        stimulus_type : str
            "probe" or "irrelevant"
        stimulus_id : str
            Stimulus identifier (e.g., "pendrive", "mouse")
        """
        return self.send_marker(
            f"S1_onset_{stimulus_type}",
            metadata={
                "trial": trial_num,
                "stim_id": stimulus_id
            }
        )
    
    def send_s1_response(
        self,
        trial_num: int,
        key: str,
        rt: float
    ) -> int:
        """Send S1 response marker."""
        return self.send_marker(
            "S1_response",
            metadata={
                "trial": trial_num,
                "key": key,
                "rt": round(rt, 4)
            }
        )
    
    def send_s2_onset(
        self,
        trial_num: int,
        stimulus_type: str
    ) -> int:
        """
        Send S2 (digit string) onset marker.
        
        Parameters
        ----------
        trial_num : int
            Trial number
        stimulus_type : str
            "target" or "nontarget"
        """
        return self.send_marker(
            f"S2_onset_{stimulus_type}",
            metadata={"trial": trial_num}
        )
    
    def send_s2_response(
        self,
        trial_num: int,
        key: str,
        rt: float,
        correct: bool
    ) -> int:
        """Send S2 response marker."""
        return self.send_marker(
            "S2_response",
            metadata={
                "trial": trial_num,
                "key": key,
                "rt": round(rt, 4),
                "correct": int(correct)
            }
        )
    
    def send_iti_start(self, trial_num: int) -> int:
        """Send ITI start marker."""
        return self.send_marker(
            "ITI_start",
            metadata={"trial": trial_num}
        )
    
    def send_block_start(self, block_num: int) -> int:
        """Send block start marker."""
        return self.send_marker(
            "block_start",
            metadata={"block": block_num}
        )
    
    def send_block_end(self, block_num: int) -> int:
        """Send block end marker."""
        return self.send_marker(
            "block_end",
            metadata={"block": block_num}
        )
    
    def close(self) -> None:
        """Close LSL stream."""
        if self.use_brainaccess_board and self.ba_stimulation is not None:
            self.logger.info(
                f"Closing BrainAccess Board LSL stream. "
                f"Total markers sent: {self.marker_counter}"
            )
            # BrainAccess Board stimulation object doesn't need explicit close
            self.ba_stimulation = None
        
        if not self.use_brainaccess_board and self.outlet is not None:
            self.logger.info(
                f"Closing pylsl stream. Total markers sent: {self.marker_counter}"
            )
            self.outlet = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

