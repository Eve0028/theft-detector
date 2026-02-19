"""
Utility Functions
=================

Helper functions for the P300-based CIT experiment.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime


def setup_logging(
    log_dir: str,
    participant_id: str,
    session_id: int,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configure logging for the experiment.
    
    Parameters
    ----------
    log_dir : str
        Directory to save log files
    participant_id : str
        Participant identifier
    session_id : int
        Session number
    level : int, optional
        Logging level (default: INFO)
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
        log_dir,
        f"{participant_id}_S{session_id:02d}_{timestamp}.log"
    )
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('P300_CIT')
    logger.info(f"Logging initialized: {log_file}")
    
    return logger


def get_output_filename(
    output_dir: str,
    participant_id: str,
    session_id: int,
    suffix: str = "behavioral",
    extension: str = "csv",
    condition: Optional[str] = None
) -> str:
    """
    Generate output filename with timestamp.

    Parameters
    ----------
    output_dir : str
        Output directory
    participant_id : str
        Participant identifier
    session_id : int
        Session number
    suffix : str, optional
        Filename suffix (default: "behavioral")
    extension : str, optional
        File extension without dot (default: "csv")
    condition : str, optional
        Participant condition label, e.g. "thief" or "control"

    Returns
    -------
    str
        Full path to output file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    condition_part = f"_{condition}" if condition else ""

    # MNE convention: *_raw.fif → suffix at end before extension
    if suffix:
        filename = (
            f"{participant_id}_S{session_id:02d}{condition_part}"
            f"_{timestamp}_{suffix}.{extension}"
        )
    else:
        filename = (
            f"{participant_id}_S{session_id:02d}{condition_part}"
            f"_{timestamp}.{extension}"
        )

    return os.path.join(output_dir, filename)


def load_image_metadata(metadata_file: str) -> Dict[str, Dict]:
    """
    Load image metadata from CSV file.
    
    Parameters
    ----------
    metadata_file : str
        Path to metadata CSV file
        
    Returns
    -------
    dict
        Dictionary mapping filenames to metadata
    """
    import pandas as pd
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    df = pd.read_csv(metadata_file)
    metadata = {}
    
    for _, row in df.iterrows():
        metadata[row['filename']] = {
            'id': row['id'],
            'type': row['type'],
            'object_name': row['object_name'],
            'view': row['view'],
            'width_px': row['width_px'],
            'height_px': row['height_px'],
            'mean_brightness': row['mean_brightness'],
            'contrast_rms': row['contrast_rms']
        }
    
    return metadata


def find_image_files(
    image_dir: str,
    use_normalized: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Find probe and irrelevant image files.
    
    Parameters
    ----------
    image_dir : str
        Base image directory
    use_normalized : bool, optional
        Whether to use normalized images (default: True)
        
    Returns
    -------
    tuple
        (probe_images, irrelevant_images) - lists of file paths
    """
    # Determine source directory
    if use_normalized:
        source_dir = os.path.join(image_dir, "normalized")
        if not os.path.exists(source_dir):
            logging.warning(
                f"Normalized image directory not found: {source_dir}. "
                "Using raw images."
            )
            source_dir = image_dir
    else:
        source_dir = image_dir
    
    # Find image files
    probe_images = []
    irrelevant_images = []
    
    for filename in os.listdir(source_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        filepath = os.path.join(source_dir, filename)
        
        if filename.startswith('probe_'):
            probe_images.append(filepath)
        elif filename.startswith('irr_'):
            irrelevant_images.append(filepath)
    
    # Sort for consistency
    probe_images.sort()
    irrelevant_images.sort()
    
    return probe_images, irrelevant_images


def validate_responses(
    s1_key: str,
    s2_key: str,
    s2_type: str,
    config_keys: Dict[str, str]
) -> Tuple[bool, bool]:
    """
    Validate participant responses.
    
    Parameters
    ----------
    s1_key : str
        Response key for S1
    s2_key : str
        Response key for S2
    s2_type : str
        S2 stimulus type ("target" or "nontarget")
    config_keys : dict
        Key configuration from config file
        
    Returns
    -------
    tuple
        (s1_correct, s2_correct) - boolean flags
    """
    # S1: always expect configured S1 response key
    s1_correct = (s1_key == config_keys['s1_response'])
    
    # S2: check if response matches stimulus type
    if s2_type == 'target':
        s2_correct = (s2_key == config_keys['s2_target'])
    else:
        s2_correct = (s2_key == config_keys['s2_nontarget'])
    
    return s1_correct, s2_correct


def format_break_time(seconds: float) -> str:
    """
    Format break time as MM:SS.
    
    Parameters
    ----------
    seconds : float
        Time in seconds
        
    Returns
    -------
    str
        Formatted time string
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def get_timestamp() -> Tuple[float, str]:
    """
    Get current timestamp in multiple formats.
    
    Returns
    -------
    tuple
        (unix_timestamp, iso_string)
    """
    now = time.time()
    iso_str = datetime.fromtimestamp(now).isoformat()
    return now, iso_str
