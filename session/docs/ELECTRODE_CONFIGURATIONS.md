# BrainAccess Electrode Configurations

Guide for configuring electrode positions based on physical cable setup.

## Physical Cable Mapping

BrainAccess CAP has fixed cable lengths that determine which electrode position connects to which device channel:

| Channel | Position | Cable Length (S-M / L) | Notes |
|---------|----------|------------------------|-------|
| REF     | Fp1      | 30cm / 35cm           | Reference electrode |
| BIAS    | Fp2      | 30cm / 35cm           | Bias/ground electrode |
| 0       | F3       | 25cm / 30cm           | Left frontal |
| 1       | F4       | 25cm / 30cm           | Right frontal |
| 2       | C3       | 20cm                  | Left central |
| 3       | C4       | 20cm                  | Right central |
| 4       | P3       | 15cm                  | Left parietal |
| 5       | P4       | 15cm                  | Right parietal |
| 6       | O1       | 10cm                  | Left occipital |
| 7       | O2       | 10cm                  | Right occipital |

**Important:** Fp1 (REF) and Fp2 (BIAS) are not recording channels - they provide reference and bias signals.

---

## Configuration Options

### Option 1: Standard Bilateral Setup (Recommended)

Use symmetric left-right electrode pairs from available positions.

**Best for P300 CIT:**

```yaml
brainaccess:
  channels:
    - "P3"    # Left parietal (primary P300)
    - "P4"    # Right parietal (primary P300)
    - "C3"    # Left central
    - "C4"    # Right central
  
  channel_mapping: null  # Use default
```

**Rationale:**
- P3/P4 are optimal for P300 detection
- C3/C4 provide additional central activity
- Symmetric coverage reduces lateralization artifacts

---

### Option 2: Parietal-Occipital Focus

Focus on posterior regions where P300 is strongest.

```yaml
brainaccess:
  channels:
    - "P3"    # Left parietal
    - "P4"    # Right parietal
    - "O1"    # Left occipital
    - "O2"    # Right occipital
  
  channel_mapping: null
```

---

### Option 3: Full Coverage (8 channels)

Record all available positions.

```yaml
brainaccess:
  channels:
    - "F3"
    - "F4"
    - "C3"
    - "C4"
    - "P3"
    - "P4"
    - "O1"
    - "O2"
  
  channel_mapping: null
```

---

### Option 4: Custom Midline Approximation

If you need midline positions (Fz, Cz, Pz) but the cap doesn't have them physically, you can:

**Option 4A: Use Left Hemisphere as Proxy**

```yaml
brainaccess:
  channels:
    - "Fz_approx"  # Using F3 as Fz approximation
    - "Cz_approx"  # Using C3 as Cz approximation
    - "Pz_approx"  # Using P3 as Pz approximation
    - "Oz_approx"  # Using O1 as Oz approximation
  
  channel_mapping:
    "Fz_approx": 0  # F3 cable (Channel 0)
    "Cz_approx": 2  # C3 cable (Channel 2)
    "Pz_approx": 4  # P3 cable (Channel 4)
    "Oz_approx": 6  # O1 cable (Channel 6)
```

**Option 4B: Average Left-Right Channels (Post-processing)**

Record bilateral pairs and average in analysis:

```yaml
brainaccess:
  channels:
    - "F3"
    - "F4"
    - "C3"
    - "C4"
    - "P3"
    - "P4"
  
  channel_mapping: null
```

Then in analysis:
```python
# Create virtual midline channels
eeg_df['Fz'] = (eeg_df['F3'] + eeg_df['F4']) / 2
eeg_df['Cz'] = (eeg_df['C3'] + eeg_df['C4']) / 2
eeg_df['Pz'] = (eeg_df['P3'] + eeg_df['P4']) / 2
```

---

### Option 5: Custom Physical Setup

If you physically move electrodes to different cap positions:

**Example: Move electrodes to midline positions**

1. Physically place electrodes at: Fz, Cz, Pz, Oz (midline)
2. Note which cables you used
3. Configure mapping:

```yaml
brainaccess:
  channels:
    - "Fz"    # Physically at Fz position
    - "Cz"    # Physically at Cz position
    - "Pz"    # Physically at Pz position
    - "Oz"    # Physically at Oz position
  
  channel_mapping:
    "Fz": 0   # Using F3 cable (25/30cm) at Fz position
    "Cz": 2   # Using C3 cable (20cm) at Cz position
    "Pz": 4   # Using P3 cable (15cm) at Pz position
    "Oz": 6   # Using O1 cable (10cm) at Oz position
```

**Note:** Check cable length is sufficient for custom positions!

---

## Choosing the Right Configuration

### For P300 CIT (Concealed Information Test)

**Minimum (2 channels):**
```yaml
channels: ["P3", "P4"]
```

**Recommended (4 channels):**
```yaml
channels: ["P3", "P4", "C3", "C4"]
```

**Optimal (6-8 channels):**
```yaml
channels: ["F3", "F4", "C3", "C4", "P3", "P4"]
# or add O1, O2 for full coverage
```

### Literature Support

- **Rosenfeld et al. (2020)**: P300 maximum at Pz, Cz
- **Abootalebi et al. (2009)**: Parietal regions (P3, Pz, P4) show strongest P300
- **Farwell & Donchin (1991)**: Midline parietal (Pz) optimal

### Practical Considerations

1. **More channels = Better**
   - More spatial coverage
   - Redundancy if one channel has artifacts
   - Can compute virtual midline channels

2. **Bilateral > Unilateral**
   - Reduces lateralization bias
   - Allows left-right comparison
   - Enables averaging for midline approximation

3. **Parietal is Critical**
   - P3/P4 are most important for P300
   - Don't skip parietal electrodes

---

## Testing Your Configuration

After configuring, test signal quality:

```bash
python scripts/check_signal_quality.py
```

Should show:
```
P3    : ✓ GOOD
P4    : ✓ GOOD
C3    : ✓ GOOD
C4    : ✓ GOOD
```

If any channel shows POOR:
1. Check physical electrode contact
2. Verify cable is connected to correct channel
3. Check electrode position matches configuration

---

## Configuration Template

Copy this to `config/experiment_config.yaml`:

```yaml
eeg:
  device_type: "brainaccess"
  enabled: true
  send_markers: true
  
  brainaccess:
    # 1. List electrodes you want to record
    channels:
      - "P3"
      - "P4"
      - "C3"
      - "C4"
    
    # 2. Define mapping (if using custom setup)
    #    null = use standard BrainAccess CAP positions
    #    dict = custom electrode_name -> channel_number mapping
    channel_mapping: null
    
    # 3. Device settings
    sampling_rate: 250
    connection_timeout: 10.0
    buffer_size: 360
```

---

## Troubleshooting

### Error: "Electrode X not in channel mapping"

**Cause:** Requested electrode name doesn't exist in physical setup.

**Solution:** Check spelling and available positions:
- Standard positions: F3, F4, C3, C4, P3, P4, O1, O2
- Use custom `channel_mapping` if using different names

### Warning: "Could not map electrodes: ['Pz', 'Cz']"

**Cause:** Requesting midline positions that don't exist on bilateral cap.

**Solution:** Use bilateral alternatives:
- Instead of Pz: use P3, P4
- Instead of Cz: use C3, C4
- Average them in post-processing

### Poor signal quality on all channels

**Cause:** Reference electrode (Fp1) has bad contact.

**Solution:**
1. Check Fp1 (REF) electrode first - it affects ALL channels
2. Ensure good contact at forehead
3. Clean skin with alcohol wipe
4. Slightly dampen electrode if using dry contacts

---

## References

- BrainAccess CAP Manual: https://www.brainaccess.ai/hardware/brainaccess-cap/
- 10-20 System: Jasper (1958)
- P300 optimal electrodes: Picton et al. (2000)

---

**Last Updated:** January 2025

