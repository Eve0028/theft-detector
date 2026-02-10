#!/usr/bin/env python3
"""
Inspect FIF file structure and annotations.
"""
import sys
import mne

if len(sys.argv) < 2:
    print("Usage: python inspect_fif.py <fif_file>")
    sys.exit(1)

fif_file = sys.argv[1]

# Load FIF
raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)

print("="*70)
print("FIF FILE STRUCTURE")
print("="*70)

# Basic info
print(f"\nFile: {fif_file}")
print(f"Channels: {raw.info['ch_names']}")
print(f"Number of channels: {raw.info['nchan']}")
print(f"Sampling rate: {raw.info['sfreq']} Hz")
print(f"Duration: {raw.times[-1]:.2f} s")
print(f"Number of samples: {len(raw.times)}")

# Data shape
print(f"\nData shape: {raw.get_data().shape}")
print(f"  (channels × samples)")

# Channel info
print("\nChannel details:")
for ch in raw.info['chs']:
    print(f"  {ch['ch_name']}: type={ch['kind']}, unit={ch['unit']}, loc={ch['loc'][:3]}")

# Annotations
print(f"\n{'='*70}")
print("ANNOTATIONS (MARKERS)")
print("="*70)
print(f"\nTotal annotations: {len(raw.annotations)}")

if len(raw.annotations) > 0:
    print("\nFirst 20 annotations:")
    for i, (onset, duration, desc) in enumerate(zip(
        raw.annotations.onset[:20],
        raw.annotations.duration[:20],
        raw.annotations.description[:20]
    )):
        print(f"  {i+1:3d}. {onset:8.3f}s - {desc}")
    
    # Summary by type
    print("\nAnnotation types (counts):")
    unique_types = {}
    for desc in raw.annotations.description:
        # Extract base type (before |)
        base = desc.split('|')[0] if '|' in desc else desc
        unique_types[base] = unique_types.get(base, 0) + 1
    
    for type_name, count in sorted(unique_types.items()):
        print(f"  {type_name:30s}: {count:3d}")
else:
    print("  No annotations found!")

print("\n" + "="*70)
print("DATA STATISTICS")
print("="*70)

data = raw.get_data()
for i, ch_name in enumerate(raw.info['ch_names']):
    ch_data = data[i, :] * 1e6  # Convert to µV
    print(f"\n{ch_name}:")
    print(f"  Mean:  {ch_data.mean():8.2f} µV")
    print(f"  Std:   {ch_data.std():8.2f} µV")
    print(f"  Min:   {ch_data.min():8.2f} µV")
    print(f"  Max:   {ch_data.max():8.2f} µV")
    print(f"  P2P:   {ch_data.max() - ch_data.min():8.2f} µV")

print("\n" + "="*70)
