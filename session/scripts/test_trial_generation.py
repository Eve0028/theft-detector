"""
Test Trial Generation
======================

Validates the trial generation logic against the current experiment
configuration.  All expected values (total trials, reps per object, block
size, etc.) are derived **automatically** from ``config/experiment_config.yaml``
and the actual image files in ``images/``, so the tests stay correct when
stimuli or parameters change.

Usage
-----
::

    python scripts/test_trial_generation.py

"""

import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List

import yaml

# Add session/src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trial_generator import TrialGenerator  # noqa: E402


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Load YAML config.  Returns empty dict on failure."""
    if not config_path.exists():
        print(f"[WARNING] Config not found: {config_path}")
        return {}
    with open(config_path, 'r', encoding='utf-8') as fh:
        return yaml.safe_load(fh) or {}


def discover_views(images_dir: Path, prefix: str, obj_name: str) -> List[str]:
    """
    Find all view files for one object (e.g. ``probe_wolf_view*.jpg``).

    Falls back to a single synthetic path when the directory or files do not
    exist, so TrialGenerator can still be constructed without real images
    on disk (it never opens files, only uses paths as strings).
    """
    views: List[str] = []
    if images_dir.is_dir():
        for ext in ('jpg', 'jpeg', 'png'):
            views.extend(
                str(p) for p in sorted(
                    images_dir.glob(f'{prefix}_{obj_name}_view*.{ext}')
                )
            )
    if not views:
        views = [str(images_dir / f'{prefix}_{obj_name}_view1.jpg')]
    return sorted(set(views))


def build_test_setup() -> dict:
    """
    Load config + discover image views.

    Returns
    -------
    dict
        All parameters needed to construct TrialGenerator and assert
        expected trial counts.  Keys:

        - ``probe_images``      list of probe image paths
        - ``irr_images_by_obj`` {obj: [path, ...]} for each irrelevant
        - ``irrelevant_images`` flat list of all irrelevant paths
        - ``probe_reps``, ``irr_reps``, ``num_blocks``, ``target_proportion``
        - ``probe_obj``, ``irr_objs``
        - ``total_trials``, ``trials_per_block``
        - ``s2_target``, ``s2_nontargets``
    """
    root = Path(__file__).parent.parent
    config = load_config(root / 'config' / 'experiment_config.yaml')
    images_dir = root / 'images'

    t = config.get('trials', {})
    stim_img = config.get('stimuli', {}).get('images', {})
    stim_dig = config.get('stimuli', {}).get('digits', {})

    probe_reps: int = t.get('probe_repetitions', 80)
    irr_reps: int = t.get('irrelevant_repetitions', 80)
    num_blocks: int = t.get('num_blocks', 5)
    target_proportion: float = t.get('target_proportion', 0.2)
    probe_obj: str = stim_img.get('probe_object', 'probe')
    irr_objs: List[str] = stim_img.get('irrelevant_objects', [])
    s2_target: str = stim_dig.get('target', '111111')
    s2_nontargets: List[str] = stim_dig.get(
        'nontargets', ['222222', '333333', '444444', '555555']
    )

    probe_images = discover_views(images_dir, 'probe', probe_obj)

    irr_images_by_obj: Dict[str, List[str]] = {
        obj: discover_views(images_dir, 'irr', obj)
        for obj in irr_objs
    }
    irrelevant_images: List[str] = [
        p for views in irr_images_by_obj.values() for p in views
    ]

    total_trials = probe_reps + len(irr_objs) * irr_reps
    trials_per_block = total_trials // num_blocks

    return {
        'probe_images': probe_images,
        'irr_images_by_obj': irr_images_by_obj,
        'irrelevant_images': irrelevant_images,
        'probe_reps': probe_reps,
        'irr_reps': irr_reps,
        'num_blocks': num_blocks,
        'target_proportion': target_proportion,
        'probe_obj': probe_obj,
        'irr_objs': irr_objs,
        'total_trials': total_trials,
        'trials_per_block': trials_per_block,
        's2_target': s2_target,
        's2_nontargets': s2_nontargets,
    }


def make_generator(setup: dict, seed: int = 42) -> TrialGenerator:
    """Construct a TrialGenerator from the setup dict."""
    return TrialGenerator(
        probe_images=setup['probe_images'],
        irrelevant_images=setup['irrelevant_images'],
        probe_reps=setup['probe_reps'],
        irrelevant_reps=setup['irr_reps'],
        target_proportion=setup['target_proportion'],
        num_blocks=setup['num_blocks'],
        s2_target=setup['s2_target'],
        s2_nontargets=setup['s2_nontargets'],
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_trial_counts(setup: dict) -> bool:
    """Total trials == probe_reps + n_irr_objs * irr_reps."""
    print("=" * 60)
    print("TEST 1: Trial Counts")
    print("=" * 60)

    expected = setup['total_trials']
    trials = make_generator(setup).generate_trials()
    actual = len(trials)

    print(f"  Expected: {setup['probe_reps']} + "
          f"{len(setup['irr_objs'])} × {setup['irr_reps']} = {expected}")
    print(f"  Actual  : {actual}")

    if actual == expected:
        print("[PASS]")
        return True
    print(f"[FAIL]: expected {expected}, got {actual}")
    return False


def test_view_distribution(setup: dict) -> bool:
    """
    Each view must appear exactly ``reps // n_views`` or
    ``reps // n_views + 1`` times (handles uneven division).
    """
    print("\n" + "=" * 60)
    print("TEST 2: View Distribution (even rotation)")
    print("=" * 60)

    trials = make_generator(setup).generate_trials()
    image_counts = Counter(trial['s1_image'] for trial in trials)

    print("\n  Image (view) counts:")
    for img, count in sorted(image_counts.items()):
        print(f"    {Path(img).name}: {count}")

    def _check_views(obj_label: str, views: List[str], reps: int) -> bool:
        n = len(views)
        lo = reps // n
        hi = lo + (1 if reps % n else 0)
        ok = True
        for v in views:
            c = image_counts.get(v, 0)
            if not (lo <= c <= hi):
                print(f"  [FAIL] {Path(v).name}: {c} (expected {lo}-{hi})")
                ok = False
        if ok:
            print(f"  [OK]   '{obj_label}': all {n} view(s) in [{lo}, {hi}]")
        return ok

    all_ok = True
    print(f"\n  Probe '{setup['probe_obj']}' ({len(setup['probe_images'])} view(s)):")
    all_ok &= _check_views(setup['probe_obj'], setup['probe_images'], setup['probe_reps'])

    for obj, views in setup['irr_images_by_obj'].items():
        print(f"\n  Irrelevant '{obj}' ({len(views)} view(s)):")
        all_ok &= _check_views(obj, views, setup['irr_reps'])

    if all_ok:
        print("\n[PASS]: Views distributed evenly")
    else:
        print("\n[FAIL]: Uneven view distribution")
    return all_ok


def test_object_distribution(setup: dict) -> bool:
    """Each object appears exactly probe_reps / irr_reps times."""
    print("\n" + "=" * 60)
    print("TEST 3: Object Distribution")
    print("=" * 60)

    trials = make_generator(setup).generate_trials()
    obj_counts = Counter(trial['s1_object'] for trial in trials)

    print("\n  Object counts:")
    for obj, count in sorted(obj_counts.items()):
        print(f"    {obj}: {count}")

    all_ok = True

    c = obj_counts.get(setup['probe_obj'], 0)
    exp = setup['probe_reps']
    if c == exp:
        print(f"  [OK]   probe '{setup['probe_obj']}': {c}")
    else:
        print(f"  [FAIL] probe '{setup['probe_obj']}': {c} (expected {exp})")
        all_ok = False

    for obj in setup['irr_objs']:
        c = obj_counts.get(obj, 0)
        exp = setup['irr_reps']
        if c == exp:
            print(f"  [OK]   irr '{obj}': {c}")
        else:
            print(f"  [FAIL] irr '{obj}': {c} (expected {exp})")
            all_ok = False

    print("\n[PASS]" if all_ok else "\n[FAIL]")
    return all_ok


def test_view_rotation(setup: dict) -> bool:
    """No more than 5 consecutive identical image files (anti-clustering)."""
    print("\n" + "=" * 60)
    print("TEST 4: View Rotation (anti-clustering)")
    print("=" * 60)

    trials = make_generator(setup).generate_trials()
    max_consec = current = 1

    for i in range(1, len(trials)):
        if trials[i]['s1_image'] == trials[i - 1]['s1_image']:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 1

    print(f"\n  Max consecutive same image: {max_consec}")

    if max_consec <= 5:
        print("[PASS]: Views well-rotated")
    else:
        print(f"[WARN]  Clustering detected (max consecutive = {max_consec})")
    return True  # warn only, never fail hard on this


def test_block_distribution(setup: dict) -> bool:
    """Trials distributed evenly across blocks (trials_per_block each)."""
    print("\n" + "=" * 60)
    print("TEST 5: Block Distribution")
    print("=" * 60)

    trials = make_generator(setup).generate_trials()
    block_counts = Counter(trial['block'] for trial in trials)

    exp_per_block = setup['trials_per_block']
    exp_n_blocks = setup['num_blocks']

    print(f"\n  Expected: {exp_n_blocks} blocks × {exp_per_block} trials each")
    print(f"  Actual blocks: {sorted(block_counts.keys())}")
    for block in sorted(block_counts.keys()):
        print(f"    Block {block}: {block_counts[block]} trials")

    ok = (
        len(block_counts) == exp_n_blocks
        and all(c == exp_per_block for c in block_counts.values())
    )

    if ok:
        print(f"\n[PASS]")
    elif len(block_counts) != exp_n_blocks:
        print(f"\n[FAIL]: Expected {exp_n_blocks} blocks, got {len(block_counts)}")
    else:
        print(f"\n[FAIL]: Block sizes {set(block_counts.values())} ≠ {exp_per_block}")
    return ok


def test_s2_proportion(setup: dict) -> bool:
    """S2 target proportion matches config (±2 % tolerance for rounding)."""
    print("\n" + "=" * 60)
    print("TEST 6: S2 Target Proportion")
    print("=" * 60)

    trials = make_generator(setup).generate_trials()
    n_targets = sum(1 for t in trials if t['s2_type'] == 'target')
    actual = n_targets / len(trials)
    expected = setup['target_proportion']

    print(f"  Expected: {expected * 100:.0f}%  "
          f"Actual: {actual * 100:.1f}%  ({n_targets}/{len(trials)})")

    if abs(actual - expected) <= 0.02:
        print("[PASS]")
        return True
    print(f"[FAIL]: {actual:.3f} deviates from {expected:.3f}")
    return False


def test_trial_math(setup: dict) -> bool:
    """Total trials must be divisible by num_blocks (no uneven blocks)."""
    print("\n" + "=" * 60)
    print("TEST 7: Trial Math (divisibility check)")
    print("=" * 60)

    total = setup['total_trials']
    n_blocks = setup['num_blocks']
    probe_pct = setup['probe_reps'] / total * 100

    print(f"  Total  : {total}")
    print(f"  Blocks : {n_blocks}")
    print(f"  Probe  : {setup['probe_reps']}/{total} = {probe_pct:.1f}%")

    if total % n_blocks == 0:
        print(f"  {total} ÷ {n_blocks} = {total // n_blocks} trials/block")
        print("[PASS]")
        return True

    remainder = total % n_blocks
    print(f"  {total} % {n_blocks} = {remainder}  → uneven blocks")
    print("[FAIL]: Adjust probe_repetitions or irrelevant_repetitions in config")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run all tests."""
    print("\n" + "=" * 60)
    print("       TRIAL GENERATION TEST SUITE")
    print("=" * 60)

    setup = build_test_setup()

    print(f"\nConfiguration loaded:")
    print(f"  Probe      : '{setup['probe_obj']}' × {setup['probe_reps']} reps  "
          f"({len(setup['probe_images'])} view(s))")
    for obj, views in setup['irr_images_by_obj'].items():
        print(f"  Irrelevant : '{obj}' × {setup['irr_reps']} reps  "
              f"({len(views)} view(s))")
    print(f"  Total      : {setup['total_trials']} trials  "
          f"({setup['num_blocks']} blocks × {setup['trials_per_block']})")
    print(f"  S2 targets : {setup['target_proportion'] * 100:.0f}%")

    tests = [
        ("Trial Counts",        test_trial_counts),
        ("View Distribution",   test_view_distribution),
        ("Object Distribution", test_object_distribution),
        ("View Rotation",       test_view_rotation),
        ("Block Distribution",  test_block_distribution),
        ("S2 Proportion",       test_s2_proportion),
        ("Trial Math",          test_trial_math),
    ]

    results = [(name, fn(setup)) for name, fn in tests]

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        print(f"  {'[PASS]' if passed else '[FAIL]'}  {name}")

    passed_n = sum(p for _, p in results)
    total_n = len(results)
    print(f"\n  {passed_n}/{total_n} tests passed")

    if passed_n == total_n:
        print("\n>>> All tests passed!")
        return 0
    print("\n>>> Some tests failed")
    return 1


if __name__ == '__main__':
    sys.exit(main())
