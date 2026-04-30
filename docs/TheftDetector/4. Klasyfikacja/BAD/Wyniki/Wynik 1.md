ROC-AUC
0.9333

F-β (LOOCV preds)
0.7692

Accuracy
72.7%

Sensitivity
66.7%

Specificity
80.0%

|                 | Pred GUILTY | Pred. INNOCENT |
| --------------- | ----------- | -------------- |
| Actual GUILTY   | 4           | 2              |
| Actual INNOCENT | 1           | 4              |


### Parametry
{
  "filter_preset": "aggressive",
  "notch_freqs": [
    50
  ],
  "hp_cutoff": 0.3,
  "lp_cutoff": 24,
  "filter_method": "iir",
  "iir_order": 3,
  "s2_tmin": -0.2,
  "s2_tmax": 0.8,
  "s2_baseline": true,
  "s2_detrend": "DC offset (0)",
  "s2_rejection": "autoreject",
  "s2_threshold_uv": null,
  "use_individual_window": true,
  "peak_channels": [
    "Pz"
  ],
  "peak_search_tmin": 0.25,
  "peak_search_tmax": 0.75,
  "window_margin": 0.15000000000000002,
  "s2_erp_lowpass_hz": 10.0,
  "manual_tmin": 0.25,
  "manual_tmax": 0.8,
  "s1_tmin": -0.2,
  "s1_tmax": 1.0,
  "s1_baseline": true,
  "s1_detrend": "DC offset (0)",
  "s1_rejection": "autoreject",
  "s1_threshold_uv": null,
  "s1_adaptive_k": 3.0,
  "s2_trial_rejection": true,
  "s2_max_rt": 1.0,
  "bad_channels": [
    "Pz"
  ],
  "amplitude_method": "Peak-to-Peak (Peak-Valley)",
  "p2p_tmax_negative": 0.9,
  "smoothing_method": "Low-pass (Butterworth)",
  "smoothing_lp_hz": 10.0,
  "smoothing_ma_ms": 100.0,
  "n_bootstrap": 1000,
  "guilty_threshold": 0.66,
  "target_stim": null,
  "baseline_stims": null,
  "s2_match_s1_preprocessing": true,
  "ar_n_jobs": 1
}