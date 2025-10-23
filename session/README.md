# P300-Based Concealed Information Test (CIT)

Experiment implementing a P300-based Concealed Information Test using the "mock crime" paradigm for EEG-based deception detection research.

## Overview

Participants view images of objects (probe = stolen item, irrelevants = neutral items) followed by digit strings while EEG is recorded. The experiment tests whether P300 amplitude differences can detect concealed knowledge of the stolen item.

**Key Features:**
- 400 trials (80 probe + 320 irrelevants) across 5 blocks
- Multiple image views per object with even rotation
- LSL integration for EEG synchronization (BrainAccess, Muse S)
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
  device_type: "brainaccess"  # or "muse_s"
  send_markers: true
```

### 4. Run

```bash
python src/experiment.py
```

Or double-click `run_experiment.bat` (Windows) / `./run_experiment.sh` (Linux/Mac)

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
- EEG sync: `LSL_*_marker` IDs for all events
- Timestamps: `timestamp_unix`, `timestamp_iso`

### EEG Data
Recorded by your EEG device software with embedded LSL markers:
- `fixation_onset`
- `S1_onset_probe` / `S1_onset_irrelevant`
- `S1_response`, `S2_onset_target` / `S2_onset_nontarget`
- `S2_response`, `ITI_start`
- `block_start` / `block_end`

---

## Configuration

All settings in `config/experiment_config.yaml`:

**Essential Settings:**
- `participant`: ID, session, condition (thief/control)
- `eeg.device_type`: "brainaccess", "muse_s", or "none"
- `eeg.send_markers`: Enable/disable LSL markers
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

### LSL Not Working
```bash
pip install liblsl
pip install pylsl
```
Check that EEG device is streaming via LSL before starting experiment.

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

**References:**
- Rosenfeld, J. P. (2020). P300 in detecting concealed information and deception: A review.
- BAD (Bootstrapped Amplitude Difference) method for P300 analysis
- Pipeline specification: `docs/TheftDetector/Eksperyment/Pipeline sesji - końcowy.md`

---

**Version:** 1.0  
**Python:** 3.8+  
**Dependencies:** PsychoPy 2023.2.3, pylsl 1.16.2, OpenCV, Pillow
