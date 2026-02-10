# Streamlit EEG Analyzer - Quick Start Guide

## What is This?

Interactive web application for analyzing EEG data stored in FIF format. Built with Streamlit and MNE-Python.

## Key Features

🎯 **No Coding Required** - Point-and-click interface  
📊 **Interactive Visualizations** - Real-time plots and adjustments  
🔧 **Complete Pipeline** - From raw data to P300 analysis  
🎲 **CTP-BAD Classification** - Statistical guilty/innocent determination  
💾 **Export Results** - CSV, PNG, NPY formats  
📱 **Web-Based** - Runs in your browser  

## Quick Start

### 1. Install Dependencies

```bash
cd session
pip install -r requirements.txt
```

### 2. Launch Application

**Windows:**
```bash
run_analyzer.bat
```

**Linux/macOS:**
```bash
chmod +x run_analyzer.sh
./run_analyzer.sh
```

**Manual:**
```bash
streamlit run scripts/eeg_analyzer_app.py
```

### 3. Open in Browser

Application opens automatically at: `http://localhost:8501`

## Workflow

### Step 1: Load Data (📂)

1. Click "Browse files" button
2. Select your `.fif` file
3. View file information and event markers

### Step 2: Check Quality (📊)

1. Review per-channel quality metrics
2. Check for flat or noisy signals
3. Visualize raw data and PSD

### Step 3: Preprocess (🔧)

1. Enable notch filter (50/60 Hz)
2. Set bandpass filter (0.5-40 Hz recommended)
3. Click "Apply Filters"
4. Compare before/after

### Step 4: Create Epochs (📈)

1. Set time window: -0.2 to 0.8 s
2. Enable baseline correction
3. Set rejection threshold: 300 µV
4. Click "Create Epochs"

### Step 5: Analyze ERPs (🎯)

1. Click "Compute ERPs"
2. View probe vs irrelevant waveforms
3. Check P300 amplitude (300-600ms)
4. **Run CTP-BAD bootstrap analysis** (guilty/innocent classification)
5. Interpret results

### Step 6: Export (📉)

1. Download CSV files (metrics, P300)
2. Export plots (PNG, 300 DPI)
3. Save epoch data (NPY)

## Interface Overview

### Navigation Pages

```
📂 Load Data       → Upload and inspect FIF files
📊 Signal Quality  → Quality assessment and visualization
🔧 Preprocessing   → Filtering and noise removal
📈 Epoching       → Create time-locked segments
🎯 ERP Analysis   → Compute and analyze ERPs
📉 Export Results → Download data and plots
```

### Interactive Controls

| Control | Purpose |
|---------|---------|
| File uploader | Select FIF file |
| Sliders | Adjust time windows, thresholds |
| Checkboxes | Enable/disable features |
| Buttons | Trigger computations |
| Expanders | View detailed information |

## What to Look For

### Good Signal Quality

```
Channel  Status  Quality  Std (µV)  P2P (µV)
Fz       🟢      GOOD     15.2      85.3
Cz       🟢      GOOD     16.2      70.1
Pz       🟢      GOOD     23.7      96.5
```

**Action:** Proceed with analysis ✓

### Poor Signal Quality

```
Channel  Status  Quality      Std (µV)  P2P (µV)
Fz       🔴      FLAT         0.8       3.2
Cz       🔴      VERY NOISY   250.5     1205.8
Pz       🟡      POOR         105.2     520.3
```

**Action:** Check recording - may need to re-record ⚠️

### Strong P300 Effect

```
Channel  Status  Probe (µV)  Irrelevant (µV)  Difference (µV)  Effect
Fz       🟡      38.45       32.12            6.33             Moderate
Cz       🟡      42.23       35.67            6.56             Moderate
Pz       🟢      48.90       32.34            16.56            Strong
```

**Interpretation:** Clear recognition effect, strongest at Pz ✓

### CTP-BAD Classification - Guilty

```
Overall Classification: GUILTY
Max Proportion: 97.8% at Pz

Channel  Bootstrap %  Classification  Confidence
Fz       92.3%        Guilty          Moderate
Cz       94.7%        Guilty          Moderate
Pz       97.8%        Guilty          High
```

**Interpretation:** Strong statistical evidence of probe recognition (guilty) ✓

### CTP-BAD Classification - Innocent

```
Overall Classification: INNOCENT
Max Proportion: 55.7% at Pz

Channel  Bootstrap %  Classification  Confidence
Fz       48.2%        Innocent        High
Cz       52.1%        Innocent        High
Pz       55.7%        Innocent        Moderate
```

**Interpretation:** No statistical evidence of probe recognition (innocent) ✓

### Weak/No P300 Effect

```
Channel  Status  Probe (µV)  Irrelevant (µV)  Difference (µV)  Effect
Fz       🔴      28.12       27.89            0.23             None
Cz       🔴      30.45       31.02            -0.57            None
Pz       🔴      32.67       33.21            -0.54            None
```

**Interpretation:** No recognition effect detected ⚠️

## Common Issues

### Problem: All Epochs Dropped

**Symptoms:**
- "Created 0 epochs" message
- No data available for ERP analysis

**Solutions:**
1. **Lower rejection threshold**
   - Try 500 µV instead of 300 µV
   - Or disable rejection temporarily
2. **Check signal quality**
   - Go to Signal Quality page
   - Look for flat or noisy channels
3. **Verify recording**
   - EEG cap may not have been worn
   - Re-record with proper setup

### Problem: No Event Markers

**Symptoms:**
- "No events found in file"
- Cannot create epochs

**Solutions:**
1. **Check file format**
   - File must be `.fif` with embedded annotations
   - Old files may lack markers
2. **Verify annotations**
   - Look at "Event Markers" section in Load Data page
   - Should show S1_onset events
3. **Re-record if needed**
   - Ensure LSL markers are enabled during recording

### Problem: Application Won't Start

**Symptoms:**
- Import errors
- "Module not found" messages

**Solutions:**
```bash
# Check Python version (requires 3.10 or 3.11)
python --version

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or install manually
pip install streamlit mne numpy scipy matplotlib pandas
```

## Tips for Best Results

### 1. Recording Quality

- ✅ Ensure good electrode contact (impedance <10 kΩ)
- ✅ Minimize movement during recording
- ✅ Reduce EMG artifacts (relax facial muscles)
- ✅ Check signal quality in real-time

### 2. Preprocessing

- ✅ Always apply notch filter (50/60 Hz)
- ✅ Use bandpass 0.5-40 Hz for P300
- ✅ Check before/after comparison
- ✅ Proceed with filtered data for epoching

### 3. Epoch Rejection

- ✅ Start with 300 µV threshold
- ✅ Adjust based on drop rate
- ✅ Keep at least 15-20 epochs
- ✅ Accept 10-30% drop rate

### 4. P300 Analysis

- ✅ Focus on Pz electrode (strongest effect)
- ✅ Use 300-600ms time window
- ✅ Look for probe > irrelevant pattern
- ✅ Expect 5-20 µV difference

## Output Files

### Exportable Data

| File | Format | Content |
|------|--------|---------|
| `eeg_summary.csv` | CSV | Basic file info |
| `signal_quality.csv` | CSV | Per-channel metrics |
| `p300_analysis.csv` | CSV | P300 amplitudes |
| `erp_analysis.png` | PNG | ERP waveforms (300 DPI) |
| `epochs_data.npy` | NPY | Epoch arrays |

### Using Exported Data

**Python:**
```python
import pandas as pd
import numpy as np

# Load P300 results
p300 = pd.read_csv('p300_analysis.csv')
print(p300)

# Load epoch data
epochs = np.load('epochs_data.npy')
print(epochs.shape)  # (n_epochs, n_channels, n_times)
```

**Excel/Sheets:**
- Open `.csv` files directly
- Import for statistics and plotting

## Comparison with CLI Script

### When to Use Streamlit App

✅ **Exploratory analysis** - Testing parameters, visualizing data  
✅ **Single file analysis** - Detailed inspection of one recording  
✅ **Interactive tuning** - Finding optimal filter/threshold settings  
✅ **Presentation** - Demonstrating results to others  
✅ **Learning** - Understanding the analysis pipeline  

### When to Use CLI Script

✅ **Batch processing** - Analyzing many files at once  
✅ **Automation** - Scripted pipelines  
✅ **Reproducibility** - Fixed parameters across sessions  
✅ **Speed** - Faster for routine analysis  
✅ **Server environments** - No GUI needed  

**Example batch processing:**
```bash
# Process all files in directory
for file in data/eeg/*.fif; do
    python scripts/example_mne_analysis.py --eeg "$file" --reject 300
done
```

## Advanced Features

### Custom P300 Window

If your data shows earlier/later P300:

```
ERP Analysis page:
- P300 window start: 0.25s  (earlier)
- P300 window end: 0.70s    (extended)
```

### Strict Quality Control

For publication-quality data:

```
Epoching page:
- Rejection threshold: 100 µV  (strict)
- Only high-quality epochs will be kept
```

### Extended Time Windows

For different ERP components:

```
Epoching page:
- Start time: -0.5s  (extended baseline)
- End time: 1.0s     (capture late components)
```

## Resources

### Example Code

- `scripts/example_mne_analysis.py` - CLI analysis pipeline
- `scripts/inspect_fif.py` - File inspection utility
- `scripts/eeg_analyzer_app.py` - Streamlit app source

### External Resources

- **MNE-Python**: https://mne.tools/
- **Streamlit**: https://docs.streamlit.io/
- **P300**: https://en.wikipedia.org/wiki/P300_(neuroscience)

## FAQ

**Q: Can I analyze files recorded with other systems?**  
A: Yes, if they can be converted to FIF format using MNE. Supports EDF, BDF, CNT, etc.

**Q: How much RAM do I need?**  
A: ~2-4 GB for typical recordings (<100 MB files). More for longer recordings.

**Q: Can I run this on a server?**  
A: Yes, but you'll need to configure port forwarding. Use `streamlit run --server.port 8501 --server.address 0.0.0.0`

**Q: Why are all my epochs dropped?**  
A: Usually poor signal quality or too strict rejection threshold. Check Signal Quality page first.

**Q: What if I don't see a P300 effect?**  
A: This is normal if the participant didn't recognize the probe, or if signal quality was poor.

**Q: Can I change the color scheme?**  
A: Yes, edit the CSS section in `eeg_analyzer_app.py` (lines 40-55).

## Support

For questions or issues:

1. Check this guide and `README_ANALYZER_APP.md`
2. Review example outputs in `docs/`
3. Consult MNE documentation
4. Check script source code for customization

---

**Last Updated:** 2026-02-03  
**Version:** 1.0  
**Application:** `scripts/eeg_analyzer_app.py`
