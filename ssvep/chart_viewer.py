"""
Standalone power spectrum viewer. Reads freqs/powers from a file and updates a plot.
Launched by the SSVEP app (e.g. on F2) so the chart runs in a separate window.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as _plt
    plt = _plt
except ImportError:
    plt = None  # type: ignore[assignment]

DEFAULT_PATH = Path(tempfile.gettempdir()) / "ssvep_power.npy"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    if plt is None:
        print("matplotlib required for chart viewer: pip install matplotlib")
        sys.exit(1)
    fig, ax = plt.subplots(figsize=(8, 3), facecolor="#14161a")
    ax.set_facecolor("#14161a")
    ax.tick_params(colors="#aaa")
    ax.spines["bottom"].set_color("#4a6b3a")
    ax.spines["top"].set_color("#4a6b3a")
    ax.spines["left"].set_color("#4a6b3a")
    ax.spines["right"].set_color("#4a6b3a")
    ax.set_xlabel("Frequency (Hz)", color="#aaa")
    ax.set_ylabel("Power", color="#aaa")
    ax.set_title("Power spectrum (SSVEP)", color="#ccc")
    line, = ax.plot([], [], color="#50b4dc", linewidth=1.5)
    ax.set_xlim(5, 16)
    ax.set_ylim(0, 1)
    if fig.canvas.manager is not None:
        getattr(fig.canvas.manager, "set_window_title", lambda _: None)("SSVEP Power")
    plt.ion()
    plt.show(block=False)
    try:
        while plt.get_fignums():
            try:
                if path.exists():
                    data = np.load(path, allow_pickle=True)
                    if isinstance(data, np.ndarray) and data.shape == ():
                        data = data.item()
                    if isinstance(data, dict) and "freqs" in data and "powers" in data:
                        freqs = np.asarray(data["freqs"])
                        powers = np.asarray(data["powers"])
                        if len(freqs) and len(powers):
                            line.set_data(freqs, powers)
                            p_max = float(np.max(powers)) or 1e-12
                            ax.set_ylim(0, p_max * 1.05)
                            ax.figure.canvas.draw_idle()
            except Exception:
                pass
            plt.pause(0.2)
    except Exception:
        pass
    try:
        plt.close(fig)
    except Exception:
        pass


if __name__ == "__main__":
    main()
