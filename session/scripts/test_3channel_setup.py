"""
Test script for 3-channel custom setup (Fz, Cz, Pz).

Tests:
1. Custom channel mapping load
2. Connection and channel setup
3. Data collection
4. Signal quality check
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from brainaccess_handler import BrainAccessHandler

def test_3channel_setup():
    """Test custom 3-channel configuration."""
    
    print("\n" + "="*60)
    print("BrainAccess 3-Channel Custom Setup Test")
    print("="*60)
    
    # Custom mapping (same as in config)
    custom_mapping = {
        "Fz": 0,  # F3 cable at Fz position
        "Cz": 2,  # C3 cable at Cz position
        "Pz": 4   # P3 cable at Pz position
    }
    
    channels = ["Fz", "Cz", "Pz"]
    
    print(f"\nChannels: {channels}")
    print(f"Custom mapping: {custom_mapping}")
    
    # Create handler with verbose mode
    print("\n" + "-"*60)
    print("Creating BrainAccessHandler...")
    print("-"*60)
    
    handler = BrainAccessHandler(
        channels=channels,
        channel_mapping=custom_mapping,
        sampling_rate=250,
        buffer_size=60,
        enabled=True,
        verbose=True  # Show all diagnostics
    )
    
    # Connect
    print("\n" + "-"*60)
    print("Connecting to device...")
    print("-"*60)
    
    if not handler.connect():
        print("\n❌ CONNECTION FAILED!")
        print("Make sure:")
        print("  1. BrainAccess device is powered on")
        print("  2. Device is in pairing mode")
        print("  3. Bluetooth is enabled")
        return False
    
    print("\n✓ Connected successfully!")
    
    # Wait for data collection
    print("\n" + "-"*60)
    print("Collecting data...")
    print("-"*60)
    print("Waiting 5 seconds for buffer to fill...")
    
    for i in range(5, 0, -1):
        print(f"  {i}...", end="\r")
        time.sleep(1)
    
    print("\n\nChecking buffer...")
    buffer_size = len(handler.eeg_data)
    expected_samples = 250 * 5  # 5 seconds @ 250 Hz
    
    print(f"  Buffer size: {buffer_size} samples")
    print(f"  Expected: ~{expected_samples} samples")
    
    if buffer_size < 250:
        print("\n⚠ WARNING: Not enough data collected!")
        print(f"  Only {buffer_size} samples (need at least 250)")
        print("\nPossible causes:")
        print("  1. Callback not being called")
        print("  2. Channels not enabled properly")
        print("  3. No data from device")
    else:
        print(f"\n✓ Buffer OK ({buffer_size} samples)")
    
    # Check signal quality
    print("\n" + "-"*60)
    print("Signal Quality Check")
    print("-"*60)
    
    quality = handler.get_signal_quality()
    
    for ch, q in quality.items():
        symbol = "✓" if q == "good" else "⚠" if q == "fair" else "✗"
        print(f"  {ch:6s}: {symbol} {q.upper()}")
    
    # Get sample values
    print("\n" + "-"*60)
    print("Sample EEG Values (last sample)")
    print("-"*60)
    
    sample = handler.get_latest_sample()
    if sample:
        for ch in channels:
            if ch in sample:
                val = sample[ch]
                if val == 0.0:
                    print(f"  {ch:6s}: {val:8.2f} µV  ⚠ ZERO - check mapping!")
                else:
                    print(f"  {ch:6s}: {val:8.2f} µV")
    else:
        print("  No sample available")
    
    # Show chunk index map
    print("\n" + "-"*60)
    print("Internal Mappings")
    print("-"*60)
    print(f"  channel_mapping: {handler.channel_mapping}")
    print(f"  chunk_index_map: {handler.chunk_index_map}")
    
    # Disconnect
    print("\n" + "-"*60)
    print("Disconnecting...")
    print("-"*60)
    
    handler.disconnect()
    print("✓ Disconnected")
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    all_good = all(q in ['good', 'fair'] for q in quality.values())
    no_zeros = sample and all(sample.get(ch, 0.0) != 0.0 for ch in channels)
    
    if all_good and no_zeros and buffer_size >= 250:
        print("✓ ALL TESTS PASSED!")
        print("\nYour setup is ready for experiments.")
        return True
    else:
        print("⚠ SOME ISSUES DETECTED")
        if not all_good:
            print("  - Poor signal quality on some channels")
        if not no_zeros:
            print("  - Some channels showing 0.0 (mapping issue?)")
        if buffer_size < 250:
            print("  - Insufficient data collection")
        print("\nCheck logs above for details.")
        return False


if __name__ == "__main__":
    try:
        success = test_3channel_setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
