"""
Signal Quality Check Script
============================

Quick script to test BrainAccess device connection and assess
signal quality before running the experiment.

Usage
-----
::

    python scripts/check_signal_quality.py

"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from brainaccess_handler import BrainAccessHandler
import logging
from mne.filter import filter_data, notch_filter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_signal_quality(duration: float = 10.0):
    """
    Test device connection and signal quality.
    
    Parameters
    ----------
    duration : float
        Duration to record in seconds (default: 10)
    """
    print("\n" + "="*60)
    print("BrainAccess Signal Quality Check")
    print("="*60 + "\n")
    
    # Create handler
    print("Connecting to BrainAccess device...")
    handler = BrainAccessHandler(
        channels=['F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2'],  # 8-channel CAP
        sampling_rate=250,
        buffer_size=360,
        enabled=True,
        verbose=True  # Show detailed logs for diagnostics
    )
    
    # Connect
    if not handler.connect():
        print("\n❌ FAILED: Could not connect to device")
        print("\nTroubleshooting:")
        print("  1. Is BrainAccess Board running?")
        print("  2. Is device powered on (red LED)?")
        print("  3. Is device connected in Board (blue LED)?")
        return False
    
    print("✓ Connected successfully\n")
    
    # Wait for data accumulation
    print(f"Recording {duration} seconds of data...")
    print("(Please remain still with eyes open)\n")
    
    for i in range(int(duration)):
        time.sleep(1)
        print(f"  {i+1}/{int(duration)} seconds...", end='\r')
    
    print(f"\n\n{'='*60}")
    print("Signal Quality Assessment")
    print("="*60 + "\n")
    
    # Assess quality
    quality = handler.get_signal_quality()
    
    all_good = True
    for channel, qual in quality.items():
        status_icon = "✓" if qual == "good" else "⚠" if qual == "fair" else "❌"
        print(f"  {channel:6s}: {status_icon} {qual.upper()}")
        
        if qual == "poor":
            all_good = False
            print(f"           → Check electrode contact!")
        elif qual == "fair":
            print(f"           → Signal usable but could be better")
    
    print("\n" + "="*60)
    
    if all_good:
        print("\n✓ All channels show GOOD signal quality")
        print("  Ready to start experiment!")
    else:
        print("\n⚠ Some channels show POOR signal quality")
        print("\nRecommendations:")
        print("  1. Check electrode contacts (especially reference at Fp1)")
        print("  2. Reposition electrodes with poor contact")
        print("  3. Apply small amount of saline/gel if using dry electrodes")
        print("  4. Wait 2-3 minutes for signals to stabilize")
        print("  5. Run this check again")
    
    # Get sample data
    print("\n" + "="*60)
    print("Sample EEG Values (µV) [filtered 1-40 Hz + notch 50/60]")
    print("="*60 + "\n")
    
    window_seconds = 8.0  # longer window to stabilize filters
    window_data, ch_order = handler.get_recent_data(seconds=window_seconds)
    if window_data is None:
        print("  No data in buffer yet.")
    else:
        try:
            data = window_data.T  # shape (n_channels, n_times)
            # Notch 50/60 Hz then band-pass 1-40 Hz
            # Notch + band-pass with IIR to avoid long FIR requirements on short window
            # Notch tylko na 50 Hz (IIR), by uniknąć wielu pasm w IIR
            data = notch_filter(
                data,
                Fs=handler.sampling_rate,
                freqs=50,
                method="iir",
                verbose=False
            )
            data = filter_data(
                data,
                sfreq=handler.sampling_rate,
                l_freq=1.0,
                h_freq=40.0,
                method='iir',
                verbose=False
            )
            if data.shape[1] > 0:
                last_sample = data[:, -1]
                for ch, val in zip(ch_order, last_sample):
                    print(f"  {ch:6s}: {val:8.2f} µV")
            else:
                print("  No samples after filtering.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Filtering failed: %s", exc, exc_info=True)
            sample = handler.get_latest_sample()
            if sample:
                for channel in handler.channels:
                    if channel in sample:
                        print(f"  {channel:6s}: {sample[channel]:8.2f} µV")
            else:
                print("  No sample received yet.")
    
    print("\n" + "="*60 + "\n")
    
    # Disconnect
    handler.disconnect()
    print("Disconnected from device\n")
    
    return all_good


def main():
    """Main entry point."""
    try:
        result = check_signal_quality(duration=10.0)
        
        if result:
            print("Signal quality check PASSED ✓")
            sys.exit(0)
        else:
            print("Signal quality check needs attention ⚠")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nCheck interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during check: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

