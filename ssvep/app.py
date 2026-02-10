"""
SSVEP demo app: three flickering squares + rest (static) square; optional calibration.

Run with BrainAccess Board running and cap on. Look at one square; the app highlights
it when the corresponding frequency is detected. Rest = look at static square or away.
F2 = open power spectrum in a separate window. Flicker is synced to monitor refresh rate
when display.refresh_rate_hz is set in config.
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pygame

from eeg_stream import EEGStream, load_config
from ssvep_analysis import (
    bandpass_filter,
    compute_power_spectrum,
    detect_ssvep_multi,
    get_smoothed_selection,
)
from signal_quality import (
    compute_channel_stats,
    overall_status,
)

CHART_DATA_PATH = Path(tempfile.gettempdir()) / "ssvep_power.npy"


def _run_confirm_before_signal_check(
    screen: pygame.surface.Surface,
    disp_cfg: dict,
) -> bool:
    """
    Ask user to confirm before starting signal quality check.
    Returns True on SPACE or button click, False on ESC/quit.
    """
    bg = tuple(disp_cfg.get("background_rgb", [30, 30, 35]))
    white = (255, 255, 255)
    gray = (180, 180, 180)
    font_m = pygame.font.Font(None, 36)
    font_l = pygame.font.Font(None, 28)
    w, h = screen.get_size()
    # Button rect (center)
    btn_w, btn_h = 320, 56
    btn_rect = pygame.Rect((w - btn_w) // 2, (h - btn_h) // 2, btn_w, btn_h)
    btn_color = (80, 140, 200)
    btn_hover = (100, 160, 220)

    while True:
        screen.fill(bg)
        title = font_m.render("EEG connected.", True, white)
        tr = title.get_rect(center=(w // 2, h // 2 - 80))
        screen.blit(title, tr)
        sub = font_l.render("Start signal quality check?", True, gray)
        sr = sub.get_rect(center=(w // 2, h // 2 - 45))
        screen.blit(sub, sr)
        # Button
        mouse = pygame.mouse.get_pos()
        hover = btn_rect.collidepoint(mouse)
        color = btn_hover if hover else btn_color
        pygame.draw.rect(screen, color, btn_rect, border_radius=8)
        lbl = font_l.render("Start check", True, white)
        lr = lbl.get_rect(center=btn_rect.center)
        screen.blit(lbl, lr)
        hint = font_l.render("or press SPACE  |  ESC = quit", True, gray)
        hr = hint.get_rect(center=(w // 2, h // 2 + 60))
        screen.blit(hint, hr)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_SPACE:
                    return True
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_rect.collidepoint(event.pos):
                    return True
        time.sleep(0.05)
    return False


def _run_signal_quality_screen(
    screen: pygame.surface.Surface,
    stats: list,
    status_msg: str,
    ok_to_proceed: bool,
    disp_cfg: dict,
) -> bool:
    """
    Draw signal quality table and status. Wait for SPACE (continue) or ESC (quit).
    Returns True to continue to main task, False to quit.
    """
    bg = tuple(disp_cfg.get("background_rgb", [30, 30, 35]))
    white = (255, 255, 255)
    gray = (180, 180, 180)
    green = (100, 255, 140)
    amber = (255, 200, 80)
    red = (255, 100, 100)
    font_l = pygame.font.Font(None, 28)
    font_m = pygame.font.Font(None, 36)
    font_s = pygame.font.Font(None, 22)
    line_h = 24
    col_w = 90

    while True:
        screen.fill(bg)
        y = 30
        title = font_m.render("Signal quality check", True, white)
        screen.blit(title, (30, y))
        y += 50
        status_color = green if ok_to_proceed else amber
        status = font_m.render(status_msg[:80], True, status_color)
        screen.blit(status, (30, y))
        y += 45
        # Header
        headers = ["Channel", "Mean (µV)", "Std (µV)", "Min", "Max", "PtP (µV)", "Quality"]
        for i, h in enumerate(headers):
            t = font_l.render(h, True, gray)
            screen.blit(t, (30 + i * col_w, y))
        y += line_h + 5
        for s in stats:
            q = s["quality"]
            qcolor = green if q == "good" else (amber if q == "fair" else red)
            row = [
                s["name"],
                f"{s['mean']:.1f}",
                f"{s['std']:.2f}",
                f"{s['min']:.1f}",
                f"{s['max']:.1f}",
                f"{s['ptp']:.1f}",
                q.upper(),
            ]
            for i, cell in enumerate(row):
                t = font_l.render(cell, True, white if i != 6 else qcolor)
                screen.blit(t, (30 + i * col_w, y))
            y += line_h
            snippet_str = ", ".join(f"{v:.1f}" for v in s["snippet"][:8])
            t = font_s.render(f"  snippet: [{snippet_str}]", True, gray)
            screen.blit(t, (30, y))
            y += line_h
        y += 20
        inst = font_m.render("SPACE = continue | ESC = quit", True, gray)
        screen.blit(inst, (30, y))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_SPACE:
                    return True
        time.sleep(0.05)
    return True


def _run_calibration(
    screen: pygame.surface.Surface,
    stream: EEGStream,
    freqs_hz: List[float],
    window_sec: float,
    rest_seconds: float,
    seconds_per_target: float,
    rest_margin_std: float,
    disp_cfg: dict,
    pre_cfg: dict,
    stim_cfg: dict,
    eeg_cfg: dict,
) -> Optional[float]:
    """
    Run calibration: user looks at LEFT, CENTER, RIGHT, then REST (gray square).
    Squares are shown and flicker during each phase so the user knows where to look.
    Returns rest_threshold or None on skip/failure.
    """
    import numpy as np

    bg = tuple(disp_cfg.get("background_rgb", [30, 30, 35]))
    white = (255, 255, 255)
    gray = (180, 180, 180)
    black = (0, 0, 0)
    font_m = pygame.font.Font(None, 42)
    font_l = pygame.font.Font(None, 28)
    sq_size = disp_cfg.get("square_size_px", 120)
    gap = 160
    color_left = tuple(disp_cfg.get("square_color_left", [255, 255, 255]))
    color_center = tuple(disp_cfg.get("square_color_center", [255, 255, 255]))
    color_right = tuple(disp_cfg.get("square_color_right", [255, 255, 255]))
    color_rest = tuple(disp_cfg.get("rest_square_color", [90, 90, 90]))
    flicker_black = disp_cfg.get("flicker_black", True)
    f_left, f_center, f_right = freqs_hz[0], freqs_hz[1], freqs_hz[2]
    dt_left = 0.5 / f_left
    dt_center = 0.5 / f_center
    dt_right = 0.5 / f_right

    fs = eeg_cfg["sampling_rate"]
    band_low = pre_cfg.get("bandpass_low_hz", 5.0)
    band_high = pre_cfg.get("bandpass_high_hz", 30.0)
    filter_order = pre_cfg.get("filter_order", 4)
    car = pre_cfg.get("common_average_reference", True)
    method = stim_cfg.get("detection_method", "cca")
    cca_n_harmonics = stim_cfg.get("cca_n_harmonics", 2)
    cca_components = stim_cfg.get("cca_components", 1)
    cca_reg = stim_cfg.get("cca_reg", 1e-4)
    freq_tol = stim_cfg.get("frequency_tolerance_hz", 0.5)
    use_h2 = stim_cfg.get("use_second_harmonic", True)

    phases = [
        ("LEFT", seconds_per_target),
        ("CENTER", seconds_per_target),
        ("RIGHT", seconds_per_target),
        ("REST (look at gray square)", rest_seconds),
    ]
    rest_max_scores: List[float] = []
    cal_clock = pygame.time.Clock()

    def draw_calibration_squares(
        w: int, h: int,
        state_l: int, state_c: int, state_r: int,
    ) -> tuple:
        cx, cy = w // 2, h // 2
        r_left = pygame.Rect(cx - gap - sq_size, cy - sq_size // 2, sq_size, sq_size)
        r_center = pygame.Rect(cx - sq_size // 2, cy - sq_size // 2, sq_size, sq_size)
        r_right = pygame.Rect(cx + gap, cy - sq_size // 2, sq_size, sq_size)
        r_rest = pygame.Rect(cx - sq_size // 2, cy + gap - sq_size // 2, sq_size, sq_size)
        for rect, base_color, state in [
            (r_left, color_left, state_l),
            (r_center, color_center, state_c),
            (r_right, color_right, state_r),
        ]:
            c = black if (state == 0 and flicker_black) else base_color
            pygame.draw.rect(screen, c, rect)
        pygame.draw.rect(screen, color_rest, r_rest)

    for phase_name, duration in phases:
        t_left, t_center, t_right = time.perf_counter(), time.perf_counter(), time.perf_counter()
        state_left, state_center, state_right = 1, 1, 1

        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            w, h = screen.get_size()
            now = time.perf_counter()
            if now - t_left >= dt_left:
                t_left = now
                state_left = 1 - state_left
            if now - t_center >= dt_center:
                t_center = now
                state_center = 1 - state_center
            if now - t_right >= dt_right:
                t_right = now
                state_right = 1 - state_right

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return None

            screen.fill(bg)
            draw_calibration_squares(w, h, state_left, state_center, state_right)
            title = font_m.render(f"Calibration: Look at {phase_name}", True, white)
            tr = title.get_rect(center=(w // 2, 40))
            screen.blit(title, tr)
            sub = font_l.render(f"({duration:.0f} s)  ESC = skip", True, gray)
            screen.blit(sub, (w // 2 - 80, h - 35))
            pygame.display.flip()

            data, _ = stream.get_recent(window_sec)
            if data is not None and data.shape[0] >= fs * 0.5:
                _, scores = detect_ssvep_multi(
                    data, fs, freqs_hz,
                    bandpass_low=band_low,
                    bandpass_high=band_high,
                    filter_order=filter_order,
                    car=car,
                    freq_tol_hz=freq_tol,
                    use_second_harmonic=use_h2,
                    method=method,
                    cca_n_harmonics=cca_n_harmonics,
                    cca_components=cca_components,
                    cca_reg=cca_reg,
                    rest_threshold=None,
                )
                if phase_name.startswith("REST") and len(scores) > 0:
                    rest_max_scores.append(max(scores))
            cal_clock.tick(120)

    if not rest_max_scores:
        return None
    rest_mean = float(np.mean(rest_max_scores))
    rest_std = float(np.std(rest_max_scores))
    if rest_std <= 0:
        rest_std = 1e-6
    rest_threshold = rest_mean + rest_margin_std * rest_std
    print(f"Calibration: rest threshold = {rest_threshold:.4f} (mean={rest_mean:.4f}, std={rest_std:.4f})")
    return rest_threshold


def main() -> None:
    config_path = Path(__file__).parent / "config.yaml"
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    if "eeg" not in config:
        print("Config must contain 'eeg' section")
        sys.exit(1)

    eeg_cfg = config["eeg"]
    pre_cfg = config.get("preprocessing", {})
    stim_cfg = config["stimulus"]
    disp_cfg = config["display"]

    # EEG (raw_to_uv_scale converts raw ADC to µV to match BrainAccess Viewer)
    stream = EEGStream(
        channels=eeg_cfg["channels"],
        channel_mapping=eeg_cfg.get("channel_mapping"),
        sampling_rate=eeg_cfg["sampling_rate"],
        buffer_seconds=eeg_cfg.get("buffer_seconds", 60),
        raw_to_uv_scale=eeg_cfg.get("raw_to_uv_scale", 1.0),
    )
    if not stream.connect():
        print("Could not connect to BrainAccess. Is BrainAccess Board running?")
        sys.exit(1)
    print("EEG connected.")

    # Pygame early so we can show confirmation before signal check
    pygame.init()
    flags = pygame.FULLSCREEN if disp_cfg.get("fullscreen") else 0
    flags |= pygame.RESIZABLE
    size = (disp_cfg.get("width", 1280), disp_cfg.get("height", 720))
    screen = pygame.display.set_mode(size, flags)
    pygame.display.set_caption("SSVEP Demo - Look at one square")
    clock = pygame.time.Clock()
    refresh_rate_hz = disp_cfg.get("refresh_rate_hz")  # None = uncapped; set for flicker sync

    # User confirmation before starting signal quality check
    sig_cfg = config.get("signal_check", {})
    show_quality_screen = sig_cfg.get("show_screen", True)
    if show_quality_screen:
        if not _run_confirm_before_signal_check(screen, disp_cfg):
            stream.disconnect()
            pygame.quit()
            return
    print("Collecting signal for quality check...")

    # Signal quality: collect data then show stats
    collect_sec = sig_cfg.get("collect_seconds", 2.5)
    time.sleep(collect_sec)
    data, ch_names = stream.get_recent(collect_sec)
    # Optionally filter before stats so values match BrainAccess Viewer (which shows filtered signal)
    if data is not None and sig_cfg.get("filter_before_stats", False):
        # bandpass_filter expects time on last axis; data is (samples, channels)
        data = bandpass_filter(
            data.T,
            low_hz=pre_cfg.get("bandpass_low_hz", 5.0),
            high_hz=pre_cfg.get("bandpass_high_hz", 30.0),
            fs=eeg_cfg["sampling_rate"],
            order=pre_cfg.get("filter_order", 4),
        ).T
    stats = compute_channel_stats(data, ch_names, snippet_len=10) if data is not None else []
    status_msg, ok_to_proceed = overall_status(stats)
    if not stats:
        print("Warning: no data for quality check.")
    else:
        print("Channel stats:", [(s["name"], s["mean"], s["std"], s["quality"]) for s in stats])
        print("Overall:", status_msg)

    if show_quality_screen and stats:
        if not _run_signal_quality_screen(screen, stats, status_msg, ok_to_proceed, disp_cfg):
            stream.disconnect()
            pygame.quit()
            return

    bg = tuple(disp_cfg.get("background_rgb", [30, 30, 35]))
    sq_size = disp_cfg.get("square_size_px", 120)
    color_left = tuple(disp_cfg.get("square_color_left", [80, 120, 200]))
    color_center = tuple(disp_cfg.get("square_color_center", [120, 120, 180]))
    color_right = tuple(disp_cfg.get("square_color_right", [200, 120, 80]))
    color_rest = tuple(disp_cfg.get("rest_square_color", [90, 90, 90]))
    color_detected = tuple(disp_cfg.get("detected_color", [100, 255, 140]))
    flicker_black = disp_cfg.get("flicker_black", True)
    black = (0, 0, 0)

    f_left = stim_cfg["frequency_left_hz"]
    f_center = stim_cfg["frequency_center_hz"]
    f_right = stim_cfg["frequency_right_hz"]
    freqs_hz = [f_left, f_center, f_right]
    rest_enabled = stim_cfg.get("rest_enabled", True)
    window_sec = stim_cfg["analysis_window_seconds"]
    detection_method = stim_cfg.get("detection_method", "fft")
    freq_tol = stim_cfg.get("frequency_tolerance_hz", 0.5)
    use_h2 = stim_cfg.get("use_second_harmonic", True)
    cca_n_harmonics = stim_cfg.get("cca_n_harmonics", 2)
    cca_components = stim_cfg.get("cca_components", 1)
    cca_reg = stim_cfg.get("cca_reg", 1e-4)
    fs = eeg_cfg["sampling_rate"]
    band_low = pre_cfg.get("bandpass_low_hz", 5.0)
    band_high = pre_cfg.get("bandpass_high_hz", 30.0)
    filter_order = pre_cfg.get("filter_order", 4)
    car = pre_cfg.get("common_average_reference", True)

    # Power chart: opened in separate window (F2); data written to CHART_DATA_PATH
    chart_freq_min = disp_cfg.get("power_chart_freq_min", 5.0)
    chart_freq_max = disp_cfg.get("power_chart_freq_max", 16.0)
    chart_step_hz = disp_cfg.get("power_chart_step_hz", 0.5)

    gap = 160
    # Flicker: when refresh_rate_hz is set, flip by frame count for exact monitor sync
    if refresh_rate_hz and refresh_rate_hz > 0:
        frames_per_half_left = max(1, int(round(refresh_rate_hz / (2 * f_left))))
        frames_per_half_center = max(1, int(round(refresh_rate_hz / (2 * f_center))))
        frames_per_half_right = max(1, int(round(refresh_rate_hz / (2 * f_right))))
    else:
        frames_per_half_left = frames_per_half_center = frames_per_half_right = None
    frame_count = 0

    # Flicker state for left, center, right (rest is static)
    t_left, t_center, t_right = 0.0, 0.0, 0.0
    state_left, state_center, state_right = 1, 1, 1
    dt_left = 0.5 / f_left
    dt_center = 0.5 / f_center
    dt_right = 0.5 / f_right

    # Calibration: rest threshold (max score below = classify as rest)
    rest_threshold: Optional[float] = None
    cal_cfg = config.get("calibration", {})
    if cal_cfg.get("enabled", False) and rest_enabled:
        rest_threshold = _run_calibration(
            screen, stream, freqs_hz,
            window_sec=window_sec,
            rest_seconds=cal_cfg.get("rest_seconds", 3.0),
            seconds_per_target=cal_cfg.get("seconds_per_target", 4.0),
            rest_margin_std=cal_cfg.get("rest_margin_std", 1.5),
            disp_cfg=disp_cfg,
            pre_cfg=pre_cfg,
            stim_cfg=stim_cfg,
            eeg_cfg=eeg_cfg,
        )
    # 4 classes: 0=left, 1=center, 2=right, 3=rest
    history: list = []
    smooth_count = 2
    n_classes = 4 if rest_enabled else 3
    chart_process: Optional[subprocess.Popen] = None

    try:
        while True:
            # Flicker: frame-based (sync to monitor) or time-based
            if frames_per_half_left is not None:
                if frame_count % frames_per_half_left == 0:
                    state_left = 1 - state_left
                if frame_count % frames_per_half_center == 0:
                    state_center = 1 - state_center
                if frame_count % frames_per_half_right == 0:
                    state_right = 1 - state_right
            else:
                time_sec = pygame.time.get_ticks() / 1000.0
                if time_sec - t_left >= dt_left:
                    t_left = time_sec
                    state_left = 1 - state_left
                if time_sec - t_center >= dt_center:
                    t_center = time_sec
                    state_center = 1 - state_center
                if time_sec - t_right >= dt_right:
                    t_right = time_sec
                    state_right = 1 - state_right
            frame_count += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    if event.key == pygame.K_F2:
                        # Open power chart in separate window
                        if chart_process is None or chart_process.poll() is not None:
                            try:
                                chart_script = Path(__file__).parent / "chart_viewer.py"
                                chart_process = subprocess.Popen(
                                    [sys.executable, str(chart_script), str(CHART_DATA_PATH)],
                                    cwd=str(Path(__file__).parent),
                                )
                            except Exception:
                                pass
                if event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode((event.w, event.h), flags)

            # EEG analysis (3 targets + optional rest) + power spectrum for chart file
            data, _ = stream.get_recent(window_sec)
            freqs_plot = None
            powers_plot = None
            if data is not None and data.shape[0] >= fs * 0.5:
                freqs_plot, powers_plot = compute_power_spectrum(
                    data, fs,
                    freq_min_hz=chart_freq_min,
                    freq_max_hz=chart_freq_max,
                    step_hz=chart_step_hz,
                    bandpass_low=band_low,
                    bandpass_high=band_high,
                    filter_order=filter_order,
                    car=car,
                )
                # Write to file for separate chart window (F2)
                if freqs_plot is not None and powers_plot is not None:
                    try:
                        np.save(CHART_DATA_PATH, {"freqs": freqs_plot, "powers": powers_plot})
                    except Exception:
                        pass
            detected_idx: Optional[int] = None
            if data is not None and data.shape[0] >= fs * 0.5:
                idx, _ = detect_ssvep_multi(
                    data, fs, freqs_hz,
                    bandpass_low=band_low,
                    bandpass_high=band_high,
                    filter_order=filter_order,
                    car=car,
                    freq_tol_hz=freq_tol,
                    use_second_harmonic=use_h2,
                    method=detection_method,
                    cca_n_harmonics=cca_n_harmonics,
                    cca_components=cca_components,
                    cca_reg=cca_reg,
                    rest_threshold=rest_threshold if rest_enabled else None,
                )
                # Map detector -1 (rest) -> 3 for history
                hist_val = 3 if idx == -1 else idx
                history.append(hist_val)
                if len(history) > 25:
                    history.pop(0)
                detected_idx = get_smoothed_selection(
                    history, min_agreements=smooth_count, n_classes=n_classes
                )

            # Draw (layout from current window size for resizability)
            w, h = screen.get_size()
            cx, cy = w // 2, h // 2
            rect_left = pygame.Rect(cx - gap - sq_size, cy - sq_size // 2, sq_size, sq_size)
            rect_center = pygame.Rect(cx - sq_size // 2, cy - sq_size // 2, sq_size, sq_size)
            rect_right = pygame.Rect(cx + gap, cy - sq_size // 2, sq_size, sq_size)
            rect_rest = pygame.Rect(cx - sq_size // 2, cy + gap - sq_size // 2, sq_size, sq_size)

            screen.fill(bg)

            def draw_square(rect: pygame.Rect, base_color: tuple, state: int, is_detected: bool) -> None:
                if state == 0 and flicker_black:
                    c = black
                elif is_detected:
                    c = color_detected
                else:
                    c = base_color
                pygame.draw.rect(screen, c, rect)

            draw_square(rect_left, color_left, state_left, detected_idx == 0)
            draw_square(rect_center, color_center, state_center, detected_idx == 1)
            draw_square(rect_right, color_right, state_right, detected_idx == 2)
            if rest_enabled:
                c_rest = color_detected if detected_idx == 3 else color_rest
                pygame.draw.rect(screen, c_rest, rect_rest)

            # Labels
            font = pygame.font.Font(None, 36)
            screen.blit(font.render(f"{f_left} Hz", True, (200, 200, 200)), (rect_left.centerx - 20, rect_left.bottom + 5))
            screen.blit(font.render(f"{f_center} Hz", True, (200, 200, 200)), (rect_center.centerx - 20, rect_center.bottom + 5))
            screen.blit(font.render(f"{f_right} Hz", True, (200, 200, 200)), (rect_right.centerx - 20, rect_right.bottom + 5))
            if rest_enabled:
                screen.blit(font.render("REST", True, (160, 160, 160)), (rect_rest.centerx - 25, rect_rest.bottom + 5))
            inst = font.render("Look at one square. ESC = quit.  F2 = Power chart", True, (180, 180, 180))
            screen.blit(inst, (w // 2 - 180, h - 40))

            pygame.display.flip()
            clock.tick(refresh_rate_hz if refresh_rate_hz and refresh_rate_hz > 0 else 120)
    finally:
        if chart_process is not None and chart_process.poll() is None:
            chart_process.terminate()
        stream.disconnect()
        pygame.quit()


if __name__ == "__main__":
    main()
