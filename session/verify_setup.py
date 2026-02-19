"""
Setup Verification Script
==========================

Verifies project structure, image files and trial-math consistency.
All stimulus checks are derived dynamically from ``config/experiment_config.yaml``
and the ``images/`` directory — no hardcoded filenames.

Usage
-----
::

    python verify_setup.py

"""

import os
import sys
from pathlib import Path
from typing import Dict, List

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def check_file(filepath: str, description: str) -> bool:
    """Check whether a file exists and print the result."""
    if os.path.exists(filepath):
        print(f"  [OK]      {description}")
        return True
    print(f"  [MISSING] {description}: {filepath}")
    return False


def check_dir(dirpath: str, description: str) -> bool:
    """Check whether a directory exists and print the result."""
    if os.path.isdir(dirpath):
        print(f"  [OK]      {description}")
        return True
    print(f"  [MISSING] {dirpath} — {description}")
    return False


# ---------------------------------------------------------------------------
# Config / image helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Load YAML config. Returns empty dict if unavailable."""
    if not YAML_AVAILABLE:
        print("  [WARNING] PyYAML not installed — skipping config-based checks")
        return {}
    if not config_path.exists():
        print(f"  [WARNING] Config not found: {config_path}")
        return {}
    with open(config_path, 'r', encoding='utf-8') as fh:
        return yaml.safe_load(fh) or {}


def discover_views(images_dir: Path, prefix: str, obj_name: str) -> List[Path]:
    """Return sorted list of view files for one object (probe or irr)."""
    if not images_dir.is_dir():
        return []
    extensions = ('jpg', 'jpeg', 'png')
    views = []
    for ext in extensions:
        views.extend(sorted(images_dir.glob(f'{prefix}_{obj_name}_view*.{ext}')))
    return sorted(set(views))


def check_images_from_config(
    config: dict,
    images_dir: Path,
) -> tuple:
    """
    Check image files based on config objects.

    Returns
    -------
    tuple
        (checks_list, views_per_probe_obj, views_per_irr_obj)
        where views dicts map object name -> number of views found.
    """
    stim = config.get('stimuli', {}).get('images', {})
    probe_obj: str = stim.get('probe_object', '')
    irr_objs: List[str] = stim.get('irrelevant_objects', [])

    print(f"  Probe object  (config): {probe_obj!r}")
    print(f"  Irr objects   (config): {irr_objs}")
    print()

    checks = []
    probe_views: Dict[str, int] = {}
    irr_views: Dict[str, int] = {}

    # Probe
    if probe_obj:
        views = discover_views(images_dir, 'probe', probe_obj)
        n = len(views)
        probe_views[probe_obj] = n
        label = f"probe_{probe_obj}: {n} view(s) found"
        if n > 0:
            print(f"  [OK]      {label}")
            checks.append(True)
        else:
            print(f"  [MISSING] {label} — expected ≥1 file matching "
                  f"probe_{probe_obj}_view*.jpg in {images_dir}")
            checks.append(False)

    # Irrelevants
    for obj in irr_objs:
        views = discover_views(images_dir, 'irr', obj)
        n = len(views)
        irr_views[obj] = n
        label = f"irr_{obj}: {n} view(s) found"
        if n > 0:
            print(f"  [OK]      {label}")
            checks.append(True)
        else:
            print(f"  [MISSING] {label} — expected ≥1 file matching "
                  f"irr_{obj}_view*.jpg in {images_dir}")
            checks.append(False)

    return checks, probe_views, irr_views


def check_trial_math(config: dict, irr_objs: List[str]) -> bool:
    """Verify that total trials are evenly divisible into blocks."""
    t = config.get('trials', {})
    probe_reps: int = t.get('probe_repetitions', 80)
    irr_reps: int = t.get('irrelevant_repetitions', 80)
    num_blocks: int = t.get('num_blocks', 5)
    n_irr: int = len(irr_objs)

    total = probe_reps + n_irr * irr_reps
    probe_pct = probe_reps / total * 100 if total else 0

    print(f"  Formula : {probe_reps} probe + {n_irr} × {irr_reps} irr "
          f"= {total} trials total")
    print(f"  Probe   : {probe_reps}/{total} = {probe_pct:.1f}%")

    if total % num_blocks == 0:
        print(f"  Blocks  : {total} ÷ {num_blocks} = {total // num_blocks} trials/block  [OK]")
        return True
    else:
        remainder = total % num_blocks
        print(f"  Blocks  : {total} ÷ {num_blocks} = {total / num_blocks:.1f} "
              f"(remainder {remainder})  [WARNING — uneven blocks]")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run all verification checks."""
    root = Path(__file__).parent
    config_path = root / 'config' / 'experiment_config.yaml'
    images_dir = root / 'images'

    print("=" * 60)
    print("P300-CIT PROJECT SETUP VERIFICATION")
    print("=" * 60)

    checks: List[bool] = []

    # ------------------------------------------------------------------
    # [1] Directories
    # ------------------------------------------------------------------
    print("\n[1] Directory structure...")
    checks.append(check_dir(str(root / 'config'), "config/"))
    checks.append(check_dir(str(root / 'src'), "src/"))
    checks.append(check_dir(str(root / 'scripts'), "scripts/"))
    checks.append(check_dir(str(root / 'images'), "images/"))

    # ------------------------------------------------------------------
    # [2] Core files
    # ------------------------------------------------------------------
    print("\n[2] Core files...")
    checks.append(check_file(str(root / 'requirements.txt'), "requirements.txt"))
    checks.append(check_file(str(root / 'README.md'), "README.md"))

    # ------------------------------------------------------------------
    # [3] Configuration
    # ------------------------------------------------------------------
    print("\n[3] Configuration...")
    checks.append(check_file(str(config_path), "experiment_config.yaml"))

    # ------------------------------------------------------------------
    # [4] Source code
    # ------------------------------------------------------------------
    print("\n[4] Source code...")
    for fname, desc in [
        ('src/__init__.py', "Source package init"),
        ('src/experiment.py', "Main experiment script"),
        ('src/trial_generator.py', "Trial generator"),
        ('src/lsl_markers.py', "LSL markers module"),
        ('src/brainaccess_handler.py', "BrainAccess handler"),
        ('src/utils.py', "Utils module"),
    ]:
        checks.append(check_file(str(root / fname), desc))

    # ------------------------------------------------------------------
    # [5] Helper scripts
    # ------------------------------------------------------------------
    print("\n[5] Scripts...")
    for fname, desc in [
        ('scripts/normalize_images.py', "Image normalization"),
        ('scripts/generate_metadata.py', "Metadata generation"),
        ('scripts/test_trial_generation.py', "Trial generation tests"),
        ('scripts/eeg_analyzer_app.py', "EEG analyzer (Streamlit)"),
    ]:
        checks.append(check_file(str(root / fname), desc))

    # ------------------------------------------------------------------
    # [6] Launcher scripts
    # ------------------------------------------------------------------
    print("\n[6] Launchers...")
    checks.append(check_file(str(root / 'run_experiment.bat'), "run_experiment.bat (Windows)"))
    checks.append(check_file(str(root / 'run_experiment.sh'), "run_experiment.sh (Linux/Mac)"))

    # ------------------------------------------------------------------
    # [7] Images — derived from config
    # ------------------------------------------------------------------
    print("\n[7] Stimulus images (from config)...")
    config = load_config(config_path)
    if config:
        img_checks, probe_views, irr_views = check_images_from_config(config, images_dir)
        checks.extend(img_checks)

        irr_objs = config.get('stimuli', {}).get('images', {}).get('irrelevant_objects', [])

        # ------------------------------------------------------------------
        # [8] Trial math
        # ------------------------------------------------------------------
        print("\n[8] Trial math consistency...")
        checks.append(check_trial_math(config, irr_objs))

        # Info: view summary
        print("\n  View summary:")
        for obj, n in probe_views.items():
            print(f"    probe_{obj}: {n} view(s)")
        for obj, n in irr_views.items():
            print(f"    irr_{obj}: {n} view(s)")
    else:
        # Fallback: just count images
        n_images = sum(
            1 for f in images_dir.iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
        ) if images_dir.is_dir() else 0
        print(f"  Found {n_images} image file(s) in images/  "
              "(install PyYAML for config-based checks)")
        checks.append(n_images > 0)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total = len(checks)
    passed = sum(checks)
    print(f"\nPassed: {passed}/{total} checks")

    if passed == total:
        print("\n[SUCCESS] All checks passed!")
        print("\nNext steps:")
        print("  1. python -m venv venv && venv\\Scripts\\activate")
        print("  2. pip install -r requirements.txt")
        print("  3. python scripts/normalize_images.py")
        print("  4. python scripts/test_trial_generation.py")
        print("  5. python src/experiment.py")
        return 0

    print("\n[WARNING] Some checks failed — please resolve missing items above.")
    return 1


if __name__ == '__main__':
    sys.exit(main())
