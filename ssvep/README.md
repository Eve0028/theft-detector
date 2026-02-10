# SSVEP Demo App

Standalone app to test the **BrainAccess MINI** (8-electrode) cap using a **Steady-State Visual Evoked Potential (SSVEP)** paradigm. Two squares flicker at different frequencies; when you look at one, the EEG shows a peak at that frequency and the app highlights the square (feedback).

## What is SSVEP?

When the retina is stimulated by a periodic visual stimulus (e.g. a flickering square), the brain generates oscillatory activity at the same frequency (and harmonics). This is measured with EEG; the response is strongest over occipital cortex. By using two stimuli at different frequencies (e.g. 8 Hz and 12 Hz), we can infer which one the user is attending to from the frequency spectrum of the EEG.

## Requirements

- BrainAccess Board running, cap connected
- Python 3.10+
- Install: `pip install -r requirements.txt`

## Run

```bash
cd ssvep
python app.py
```

Optional: `python app.py path/to/config.yaml`

- **Left square**: 8 Hz (default)  
- **Right square**: 12 Hz (default)  
- Look at one square; when the corresponding frequency is detected in the EEG, that square turns green.  
- **ESC** to quit.

## Configuration

Edit `config.yaml`:

- **eeg.channels**: Electrodes to use. For SSVEP, **O1** and **O2** (occipital) are best. You can add P3, P4, etc.
- **eeg.channel_mapping**: Maps electrode names to device channel numbers (0–7). Default cap layout:
  - F3:0, F4:1, C3:2, C4:3, P3:4, P4:5, O1:6, O2:7  
  Set to `null` to use this default. If you use custom positions (e.g. Oz at O1 cable), set e.g. `"Oz": 6`.
- **eeg.raw_to_uv_scale**: Physical units of the callback data are **not stated** in the official API docs (see links below). If your numbers are in the millions and you want µV-like scale, set this empirically (e.g. ~0.000122). Use `1.0` to leave values unchanged.
- **signal_check.filter_before_stats**: If `true`, mean/std/ptp are computed **after** the same bandpass as in preprocessing. BrainAccess Viewer shows filtered (e.g. 1–30 Hz) signal, so this makes the quality stats comparable to the Viewer. The quality table headers say “(µV)” only when you set `raw_to_uv_scale` to a calibrated factor; otherwise treat values as arbitrary units.
- **stimulus.frequency_left_hz / frequency_right_hz**: Flicker frequencies (avoid 50 Hz if you have line noise).
- **stimulus.analysis_window_seconds**: Longer = more stable detection, slower response (e.g. 3 s).
- **stimulus.detection_method**: `"fft"` or `"cca"`. CCA uses reference sin/cos at f and harmonics (`cca_n_harmonics`, `cca_components`, `cca_reg`).
- **display**: Colors, size, fullscreen.

## Preprocessing (and how it helps)

Applied in `ssvep_analysis.py` before frequency detection:

1. **Bandpass filter (default 5–30 Hz)**  
   Removes DC drift and high-frequency noise. SSVEP is typically in the 3.5–75 Hz range; a narrow band improves SNR and reduces artifacts (e.g. muscle, line noise).

2. **Common-average reference (CAR)**  
   Subtract the mean across channels from each channel. Reduces common noise (e.g. reference/bias) and can improve contrast of local activity.

3. **Detection method** (config: `stimulus.detection_method`):
   - **fft**: Power at stimulus frequency (and optional second harmonic) from FFT; compare power for left vs right target.
   - **cca**: Canonical Correlation Analysis. Reference signals are sin/cos at the target frequency and harmonics; we compute canonical correlation between the EEG segment and each reference. The target with higher correlation wins. CCA often gives better accuracy and robustness to noise.

4. **Smoothing**  
   The app requires a few consecutive analyses to agree before changing the feedback (fewer false flips).


## Signal quality: before vs after filter

- **Mean, std, ptp** are computed on the **same data** you get from the stream. If **signal_check.filter_before_stats** is `true` (default), we apply the **preprocessing bandpass** (e.g. 5–30 Hz) to that segment first, then compute stats. So the numbers are on **filtered** signal, comparable to what BrainAccess Viewer shows (Viewer uses e.g. 1–30 Hz filter). If `filter_before_stats` is `false`, stats are on **raw** (unfiltered) stream.

## Data units and BrainAccess documentation

- **What units does the Python API return?**  
  The **official BrainAccess API documentation does not state** whether the values in the chunk callback are in microvolts (µV) or raw ADC counts. So we cannot assume either.

- **Links (official):**
  - [BrainAccess Python API 3.6.1](https://www.brainaccess.ai/documentation/python-api/3.6.1/) – main docs
  - [Usage examples (3.5.0)](https://www.brainaccess.ai/documentation/python-api/3.5.0/usage.html) – minimal acquisition, `set_callback_chunk`, gain; no mention of units
  - [From Brain to Bytes](https://brainaccess.ai/from-brain-to-bytes-how-eeg-data-is-captured) – hardware pipeline (ADC, gain); no API units
  - [Neurotechnology BrainAccess API index](https://www.neurotechnology.com/brainaccess-documentation/PythonAPI/brainaccess.html) – EEGManager, `set_callback_chunk`, gain modes

- **What is “gain” set to 8 in BrainAccess Options?**  
  **Gain** is the **hardware amplification** (e.g. 8×) before the ADC. It improves SNR. The Viewer’s display already accounts for it; the Python callback data may be before or after scaling—the docs do not say.

## Files

- `app.py` – Pygame UI, flicker timing, calls EEG + analysis
- `eeg_stream.py` – BrainAccess connection and live buffer (config-based channels/mapping)
- `ssvep_analysis.py` – Bandpass, CAR; FFT or CCA detection at target frequencies
- `config.yaml` – Channels, mapping, frequencies, display, preprocessing options
- `requirements.txt` – brainaccess, numpy, PyYAML, pygame, scipy
