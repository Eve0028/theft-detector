"""
Test LSL Marker Sending
=========================

Quick script to test if LSL markers are being sent correctly.

Usage
-----
::

    python scripts/test_lsl_markers.py

"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lsl_markers import LSLMarkerSender
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_markers():
    """Test LSL marker sending."""
    print("\n" + "="*70)
    print("LSL Marker Test")
    print("="*70 + "\n")
    
    print("IMPORTANT:")
    print("  1. Make sure BrainAccess Board is RUNNING")
    print("  2. Go to Configuration tab → Select Source: Lab Streaming Layer (LSL)")
    print("  3. Select marker stream: P300_CIT_Markers")
    print("  4. Press Connect")
    print("\nPress Enter when ready...")
    input()
    
    # Initialize marker sender
    print("\nInitializing LSL marker sender...")
    sender = LSLMarkerSender(
        stream_name="P300_CIT_Markers",
        stream_type="Markers",
        device_type="brainaccess",
        enabled=True
    )
    
    if not sender.enabled:
        print("\n❌ FAILED: LSL sender is not enabled!")
        print("\nPossible causes:")
        print("  - brainaccess_board or pylsl not installed")
        print("  - Import error (check logs above)")
        return False
    
    print("✓ LSL sender initialized\n")
    
    # Check if using BrainAccess Board
    if sender.use_brainaccess_board:
        print("Backend: BrainAccess Board (native)")
        if sender.ba_stimulation and hasattr(sender.ba_stimulation, 'have_consumers'):
            if sender.ba_stimulation.have_consumers():
                print("✓ BrainAccess Board is connected and receiving markers\n")
            else:
                print("⚠ WARNING: No consumers connected!")
                print("  Make sure marker stream is connected in BrainAccess Board\n")
    else:
        print("Backend: pylsl (generic)")
        print("  Note: For BrainAccess, install brainaccess_board for better integration\n")
    
    # Send test markers
    print("Sending 10 test markers (1 per second)...")
    print("-" * 70)
    
    for i in range(1, 11):
        marker_id = sender.send_marker(
            f"test_marker_{i}",
            metadata={"index": i, "timestamp": time.time()}
        )
        
        if marker_id > 0:
            print(f"  [{i}/10] Sent marker #{marker_id}: test_marker_{i}")
        else:
            print(f"  [{i}/10] ❌ FAILED to send marker")
        
        time.sleep(1)
    
    print("-" * 70)
    
    # Close sender
    sender.close()
    
    print("\n" + "="*70)
    print("Test Complete")
    print("="*70)
    
    if marker_id > 0:
        print("\n✓ SUCCESS: Markers are being sent!")
        print("\nNext steps:")
        print("  1. Check BrainAccess Board viewer - you should see markers appearing")
        print("  2. Set send_markers: true in config/experiment_config.yaml")
        print("  3. Run the experiment")
        return True
    else:
        print("\n❌ FAILED: Markers were not sent!")
        print("\nTroubleshooting:")
        print("  1. Check logs above for errors")
        print("  2. Verify brainaccess_board is installed: pip install brainaccess-board")
        print("  3. Make sure BrainAccess Board is running and marker stream is connected")
        print("  4. See docs/LSL_MARKERS_GUIDE.md for detailed setup")
        return False


def main():
    """Main entry point."""
    try:
        result = test_markers()
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
