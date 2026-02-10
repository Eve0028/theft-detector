# P300-Based Concealed Information Test (CIT)

Experiment implementing a P300-based Concealed Information Test using the "mock crime" paradigm for EEG-based deception detection research.

## Overview

Participants view images of objects (probe = stolen item, irrelevants = neutral items) followed by digit strings while EEG is recorded. The experiment tests whether P300 amplitude differences can detect concealed knowledge of the stolen item.

**Key Features:**
- 400 trials (80 probe + 320 irrelevants) across 5 blocks
- Multiple image views per object with even rotation
- **Native BrainAccess SDK annotations** (<0.1 ms overhead per marker)
- **FIF format** with embedded event markers (MNE-Python standard)
- **Optimized performance** (42x faster markers, 6x less CPU)
- Complete behavioral data logging
- Configurable timing and parameters

**Trial Structure:**
1. Fixation cross (500 ms)
2. S1 - Image (400 ms) → Press `Z`
3. ISI (1000-1500 ms)
4. S2 - Digit string (300 ms) → Press `M` (target) or `N` (nontarget)
5. ITI (500-800 ms)

**Session Duration:** ~64 minutes (includes setup, breaks, debriefing)

---

## Quick Start

### 1. Installation

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Includes:
# - psychopy (experiment framework)
# - brainaccess (official BrainAccess API)
# - pylsl (behavioral markers)
# - numpy, pandas (data processing)
```

### 2. Prepare Images

```bash
# Normalize images (resize, brightness, contrast)
python scripts/normalize_images.py

# Generate metadata CSV
python scripts/generate_metadata.py
```

### 3. Configure

Edit `config/experiment_config.yaml`:
```yaml
participant:
  id: "P001"
  session: 1
  condition: "thief"  # or "control"

eeg:
  device_type: "brainaccess"
  enabled: true
  send_markers: true
  
  brainaccess:
    channels: ["P3", "P4", "C3", "C4"]  # Bilateral parietal-central
    channel_mapping: null  # Use standard BrainAccess CAP positions
    sampling_rate: 250
```

**Note:** BrainAccess CAP has fixed electrode positions (F3, F4, C3, C4, P3, P4, O1, O2). See `docs/ELECTRODE_CONFIGURATIONS.md` for setup options.

### 4. Setup BrainAccess Device

1. **Power on device** (hold button 2 seconds, red LED lights)
2. **Place cap** on participant's head (correct size: S/M/L)
3. **Check electrode configuration** in `config/experiment_config.yaml`
4. **Test connection**:
   ```bash
   python scripts/check_signal_quality.py
   ```
5. **Verify signal quality** - all channels should show "good" or "fair"

**Important:** Ensure Fp1 (REF) has good contact - it affects all channels!

See `docs/ELECTRODE_CONFIGURATIONS.md` for detailed electrode setup options.

### 5. Run

```bash
python src/experiment.py
```

Or double-click `run_experiment.bat` (Windows) / `./run_experiment.sh` (Linux/Mac)

The experiment will:
- Scan and connect to BrainAccess device via Bluetooth
- Check signal quality on all channels (warnings if poor)
- Start recording EEG data when experiment begins
- Save synchronized EEG + behavioral data to separate CSV files

---

## Project Structure

```
session/
├── config/
│   └── experiment_config.yaml    # All settings
├── src/
│   ├── experiment.py             # Main experiment (PsychoPy)
│   ├── trial_generator.py        # Trial sequence generation
│   ├── lsl_markers.py            # EEG synchronization
│   └── utils.py                  # Helper functions
├── scripts/
│   ├── normalize_images.py       # Image preprocessing
│   ├── generate_metadata.py      # Metadata generation
│   └── test_trial_generation.py # Unit tests
├── images/                       # Raw stimulus images
│   └── normalized/               # Processed images (auto-generated)
├── data/                         # Output (auto-generated)
│   ├── behavioral/               # CSV files with trial data
│   └── logs/                     # Session logs
└── requirements.txt
```

---

## Multiple Image Views

Each object can have multiple views (e.g., `probe_pendrive_view1.jpg`, `probe_pendrive_view2.jpg`).

**How it works:**
- Images are automatically grouped by object name
- 80 repetitions **per object** (not per image)
- For 2 views: each appears 40 times
- Views are evenly rotated throughout trials to prevent clustering

**Example:**
- Probe: pendrive (2 views) → 80 trials total
- Irrelevants: mouse, headphones, cable_usb, charger (2 views each) → 80 trials each
- **Total: 400 trials**

**Image Naming Convention:**
- `probe_objectname_view1.jpg`
- `irr_objectname_view1.jpg`

---

## Experiment Protocol

### For "Thief" Participants:
1. **Mock Crime** (5 min): Enter room, examine and take pendrive from box
2. **EEG Setup** (15 min): Mount device, check signal quality
3. **Instructions** (5 min): Explain task, practice responses
4. **Main Experiment** (~21 min): 5 blocks × 80 trials with breaks
5. **Debriefing** (5 min)

### For "Control" Participants:
1. **Room Visit** (2 min): Sign paper, do NOT open box or see pendrive
2. Same steps 2-5 as above

---

## Data Output

### Behavioral Data
CSV file in `data/behavioral/` with columns:
- Participant info: `participant_id`, `session_id`, `condition`
- Trial info: `block`, `trial_index`
- S1 data: `S1_type` (probe/irrelevant), `S1_object`, `S1_filename`, `S1_onset_time`, `S1_response_key`, `S1_RT`
- S2 data: `S2_type` (target/nontarget), `S2_string`, `S2_onset_time`, `S2_response_key`, `S2_RT`, `S2_correct`
- Timing: `ISI_duration`, `ITI_duration`
- **EEG sync: `LSL_*_marker` IDs for synchronization with EEG data**
- Timestamps: `timestamp_unix`, `timestamp_iso`

### EEG Data
FIF file (MNE-Python format) in `data/eeg/` with:
- **Continuous EEG**: Channels (P3, P4, C3, C4) at 250 Hz
- **Embedded annotations**: Event markers synchronized with <0.1 ms precision
- **Metadata**: Sampling rate, channel info, units

Format: `P001_S01_eeg_YYYYMMDD_HHMMSS.fif`

**Loading data:**
```python
import mne
raw = mne.io.read_raw_fif("data/eeg/P001_S01_eeg.fif", preload=True)
events, event_id = mne.events_from_annotations(raw)
epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.8)
```

**Legacy CSV format** still supported for compatibility. See `docs/EEG_DATA_FORMATS.md`.

### Data Analysis Tools

**Interactive Streamlit Application (🆕 Recommended):**
```bash
# Launch web-based analyzer
streamlit run scripts/eeg_analyzer_app.py
# or use launcher scripts
run_analyzer.bat  # Windows
./run_analyzer.sh # Linux/Mac
```

Features:
- 📂 Interactive file upload and inspection
- 📊 Signal quality assessment with visual indicators
- 🔧 Real-time filtering (notch, bandpass) with before/after comparison
- 📈 Epoch creation with configurable parameters
- 🎯 ERP analysis (probe vs irrelevant) with P300 quantification
- 🎲 **CTP-BAD bootstrap analysis** for guilty/innocent classification
- 📉 Export results (CSV, PNG, NPY)
- 🎨 Interactive visualizations and plots

See [`docs/STREAMLIT_APP_GUIDE.md`](docs/STREAMLIT_APP_GUIDE.md) for quick start guide.

**Command-Line Analysis Script:**
```bash
# Analyze single file with P300 pipeline
python scripts/example_mne_analysis.py --eeg data/eeg/P001_S01_*.fif --reject 300
```

**When to use each:**
- **Streamlit App**: Exploratory analysis, parameter tuning, single files, presentations
- **CLI Script**: Batch processing, automation, reproducible pipelines

### LSL Event Markers
Sent via separate LSL marker stream for sub-millisecond EEG synchronization:
- `fixation_onset`
- `S1_onset_probe` / `S1_onset_irrelevant`
- `S1_response`, `S2_onset_target` / `S2_onset_nontarget`
- `S2_response`, `ITI_start`
- `block_start` / `block_end`

**Setup Required:** Set `send_markers: true` in config and connect marker stream in BrainAccess Board. See [`docs/LSL_MARKERS_GUIDE.md`](docs/LSL_MARKERS_GUIDE.md) for detailed setup instructions.

---

## Configuration

All settings in `config/experiment_config.yaml`:

**Essential Settings:**
- `participant`: ID, session, condition (thief/control)
- `eeg.device_type`: "brainaccess", "muse_s", or "none"
- `eeg.enabled`: Enable/disable EEG recording
- `eeg.send_markers`: Enable/disable LSL markers
- `eeg.brainaccess.channels`: List of electrodes to record
- `display.fullscreen`: true/false
- `timing.*`: All durations (can adjust for pilot testing)

**Timing Parameters (defaults match pipeline):**
```yaml
timing:
  fixation_duration: 0.5
  s1_duration: 0.4
  isi_min: 1.0
  isi_max: 1.5
  s2_duration: 0.3
  iti_min: 0.5
  iti_max: 0.8
```

---

## Troubleshooting

### PsychoPy Installation Issues
If standard install fails:
```bash
pip install psychopy --no-deps
pip install psychtoolbox pillow numpy scipy matplotlib pandas openpyxl
```

### Test Without EEG
In config:
```yaml
eeg:
  enabled: false
  send_markers: false
  device_type: "none"
```

### Quick Test Run
In config, set:
```yaml
trials:
  num_blocks: 1  # Just 80 trials instead of 400
display:
  fullscreen: false
```

### BrainAccess Connection Issues

**Device not found:**
1. Device is powered on (red LED lit)
2. Bluetooth enabled on computer
3. Device is in range (~10 meters)
4. Try scanning with test script:
   ```bash
   python -c "from brainaccess.core import EEGManager; print(EEGManager().scan())"
   ```

**Poor signal quality:**
1. Check electrode contact (especially reference at Fp1)
2. Apply gel/saline to dry electrodes if needed
3. Ensure cap fits properly
4. Wait 2-3 minutes for signal to stabilize

**Library not installed:**
```bash
pip install brainaccess pylsl
```

**Check installed version:**
```bash
pip show brainaccess
# or
python -c "import brainaccess; print(brainaccess.__version__)"
```

### Verify Setup
```bash
python verify_setup.py
```
Checks all files are in place.

---

## Custom Stimuli

To use your own images:
1. Place images in `images/` directory
2. Follow naming: `probe_objectname_view1.jpg`, `irr_objectname_view2.jpg`, etc.
3. Update object names in `config/experiment_config.yaml`
4. Run normalization and metadata scripts

---

## Testing

### Unit Tests
```bash
python scripts/test_trial_generation.py
```
Tests: trial counts, view distribution, object repetitions, rotation, blocks.

### Trial Generation
```bash
python -c "from src.trial_generator import TrialGenerator; print('OK')"
```

### Dependencies
```bash
python -c "import psychopy; import pylsl; import numpy; import pandas; print('All OK')"
```

### Run Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Test BrainAccess integration (mock)
python -m pytest tests/test_brainaccess_integration.py -v -m mock

# Test with real device (requires BrainAccess connected)
python -m pytest tests/test_brainaccess_integration.py -v -m integration

# Test EEG-behavioral synchronization
python -m pytest tests/test_eeg_behavioral_sync.py -v
```

---

## BrainAccess Integration

### Device Setup

**Hardware:**
- BrainAccess MINI: 8-channel EEG device
- BrainAccess CAP: Dry-contact electrode cap
- 4 active electrodes: Pz, Cz, Fz, Fp1
- Sampling rate: 250 Hz
- Reference: Fp1 location (denoted 'R' on device)
- Bias: Fp1 location (denoted 'B' on device)

**Electrode Positions (10-20 system):**
- **Pz**: Parietal midline - primary channel for P300 detection
- **Cz**: Central midline - additional parietal activity
- **Fz**: Frontal midline - frontal activity, EOG monitoring
- **Fp1**: Left frontal pole - used as reference

### Software Integration

**Data Flow:**
1. `BrainAccessHandler` connects to device via official API
2. Device streams EEG data (250 Hz, 4 channels)
3. Experiment sends behavioral markers via LSL
4. Timestamps synchronize EEG and behavioral events
5. Data saved to separate CSV files: `eeg/` and `behavioral/`

**Key Components:**
- `brainaccess_handler.py`: Uses `brainaccess.core.EEGManager`
- `lsl_markers.py`: Behavioral event markers via LSL
- Timestamps: Unix timestamps for synchronization

**API Documentation:**
https://www.brainaccess.ai/documentation/python-api/3.6.0/

### Signal Quality Monitoring

The system continuously monitors signal quality:
- **Good**: Normal EEG amplitude (typical variation)
- **Fair**: Somewhat noisy but usable
- **Poor**: Flat signal (bad contact) or excessive noise (artifact)

Poor quality warnings are logged. Check electrode contacts if warnings occur.

### Data Synchronization

**Method:**
1. All LSL streams share synchronized timestamps
2. Behavioral CSV contains marker IDs for each event
3. EEG CSV contains LSL timestamps for each sample
4. Post-processing: match markers to nearest EEG samples

**Example synchronization:**
```python
# Find EEG samples at S1 onset
s1_timestamp = behavioral_df.loc[trial, 'S1_onset_time']
eeg_window = eeg_df[
    (eeg_df['timestamp'] >= s1_timestamp - 0.1) &
    (eeg_df['timestamp'] <= s1_timestamp + 0.8)
]
```

---

## Technical Details

### Trial Generation
- Images grouped by object name (extracted from filename)
- Repetitions distributed evenly across views
- Object lists interleaved for even rotation
- Constraint checking prevents excessive consecutive repetition
- S2 stimuli assigned with 20/80 target/nontarget ratio

### EEG Synchronization
- LSL markers sent at each event with metadata
- Timestamps logged in both experiment time and Unix time
- Marker IDs saved in behavioral CSV for post-hoc alignment

### Image Normalization
- Resize to 800×600 px (maintains aspect ratio)
- Normalize brightness: mean = 128
- Normalize contrast: RMS = 50
- Output: PNG format in `images/normalized/`

---

## Support & Documentation

- **Code Documentation**: All functions have detailed docstrings (reStructuredText)
- **Logs**: Check `data/logs/` for session logs and error messages
- **Config Comments**: YAML file has inline explanations

**Setup Guides:**
- **Electrode Configurations**: [`docs/ELECTRODE_CONFIGURATIONS.md`](docs/ELECTRODE_CONFIGURATIONS.md) - CAP setup and channel mappings


**Analysis Tools (🆕):**
- **⭐ Streamlit App Guide**: [`docs/STREAMLIT_APP_GUIDE.md`](docs/STREAMLIT_APP_GUIDE.md) - **NEW!** Interactive web-based analyzer quick start
**References:**
- Rosenfeld, J. P. (2020). P300 in detecting concealed information and deception: A review.
- BAD (Bootstrapped Amplitude Difference) method for P300 analysis
- Pipeline specification: `docs/TheftDetector/Eksperyment/Pipeline sesji - końcowy.md`

---

**Version:** 1.0  
**Python:** 3.8+  
**Dependencies:** PsychoPy 2023.2.3, pylsl 1.16.2, OpenCV, Pillow
