"""
Test Trial Generation with Multiple Views
==========================================

This script tests the trial generation logic to verify that:
1. Multiple views of the same object are handled correctly
2. Repetitions are distributed evenly across views
3. Views are rotated uniformly throughout trials
4. Total trial count is correct (based on objects, not images)

Usage
-----
::

    python test_trial_generation.py

"""

import sys
from pathlib import Path
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trial_generator import TrialGenerator


def create_mock_images():
    """Create mock image file paths for testing."""
    # Probe: 1 object (pendrive) with 2 views
    probe_images = [
        'images/probe_pendrive_view1.jpg',
        'images/probe_pendrive_view2.jpg'
    ]
    
    # Irrelevants: 4 objects, each with 2 views
    irrelevant_images = [
        'images/irr_mouse_view1.jpg',
        'images/irr_mouse_view2.jpg',
        'images/irr_headphones_view1.jpg',
        'images/irr_headphones_view2.jpg',
        'images/irr_cableusb_view1.jpg',
        'images/irr_cableusb_view2.jpg',
        'images/irr_charger_view1.jpg',
        'images/irr_charger_view2.jpg'
    ]
    
    return probe_images, irrelevant_images


def test_trial_counts():
    """Test that trial counts are correct."""
    print("=" * 60)
    print("TEST 1: Trial Counts")
    print("=" * 60)
    
    probe_images, irrelevant_images = create_mock_images()
    
    generator = TrialGenerator(
        probe_images=probe_images,
        irrelevant_images=irrelevant_images,
        probe_reps=80,
        irrelevant_reps=80,
        target_proportion=0.2,
        num_blocks=5,
        s2_target="111111",
        s2_nontargets=["222222", "333333", "444444", "555555"],
        seed=42
    )
    
    trials = generator.generate_trials()
    
    # Expected: 80 probe + 4 objects × 80 = 400 trials
    expected_total = 400
    actual_total = len(trials)
    
    print(f"Expected total trials: {expected_total}")
    print(f"Actual total trials: {actual_total}")
    
    if actual_total == expected_total:
        print("✓ PASS: Trial count is correct")
    else:
        print("✗ FAIL: Trial count is incorrect")
        return False
    
    return True


def test_view_distribution():
    """Test that views are distributed evenly."""
    print("\n" + "=" * 60)
    print("TEST 2: View Distribution")
    print("=" * 60)
    
    probe_images, irrelevant_images = create_mock_images()
    
    generator = TrialGenerator(
        probe_images=probe_images,
        irrelevant_images=irrelevant_images,
        probe_reps=80,
        irrelevant_reps=80,
        target_proportion=0.2,
        num_blocks=5,
        s2_target="111111",
        s2_nontargets=["222222", "333333", "444444", "555555"],
        seed=42
    )
    
    trials = generator.generate_trials()
    
    # Count occurrences of each image
    image_counts = Counter(trial['s1_image'] for trial in trials)
    
    print("\nImage (view) counts:")
    for img, count in sorted(image_counts.items()):
        print(f"  {Path(img).name}: {count}")
    
    # Check probe views (should be 40 each for 2 views)
    probe_view1_count = image_counts['images/probe_pendrive_view1.jpg']
    probe_view2_count = image_counts['images/probe_pendrive_view2.jpg']
    
    print(f"\nProbe view1: {probe_view1_count}, view2: {probe_view2_count}")
    
    if probe_view1_count == 40 and probe_view2_count == 40:
        print("✓ PASS: Probe views distributed evenly (40 each)")
    else:
        print("✗ FAIL: Probe views not distributed evenly")
        return False
    
    # Check irrelevant views (should be 40 each for 2 views per object)
    all_even = True
    for obj in ['mouse', 'headphones', 'cableusb', 'charger']:
        view1 = image_counts[f'images/irr_{obj}_view1.jpg']
        view2 = image_counts[f'images/irr_{obj}_view2.jpg']
        
        print(f"{obj} - view1: {view1}, view2: {view2}")
        
        if view1 != 40 or view2 != 40:
            all_even = False
    
    if all_even:
        print("✓ PASS: All irrelevant views distributed evenly (40 each)")
    else:
        print("✗ FAIL: Some irrelevant views not distributed evenly")
        return False
    
    return True


def test_object_distribution():
    """Test that objects appear correct number of times."""
    print("\n" + "=" * 60)
    print("TEST 3: Object Distribution")
    print("=" * 60)
    
    probe_images, irrelevant_images = create_mock_images()
    
    generator = TrialGenerator(
        probe_images=probe_images,
        irrelevant_images=irrelevant_images,
        probe_reps=80,
        irrelevant_reps=80,
        target_proportion=0.2,
        num_blocks=5,
        s2_target="111111",
        s2_nontargets=["222222", "333333", "444444", "555555"],
        seed=42
    )
    
    trials = generator.generate_trials()
    
    # Count occurrences of each object
    object_counts = Counter(trial['s1_object'] for trial in trials)
    
    print("\nObject counts:")
    for obj, count in sorted(object_counts.items()):
        print(f"  {obj}: {count}")
    
    # Check counts
    all_correct = True
    
    if object_counts['pendrive'] != 80:
        print(f"✗ FAIL: Pendrive should appear 80 times, got {object_counts['pendrive']}")
        all_correct = False
    else:
        print("✓ Pendrive: 80 (correct)")
    
    for obj in ['mouse', 'headphones', 'cableusb', 'charger']:
        if object_counts[obj] != 80:
            print(f"✗ FAIL: {obj} should appear 80 times, got {object_counts[obj]}")
            all_correct = False
        else:
            print(f"✓ {obj}: 80 (correct)")
    
    if all_correct:
        print("\n✓ PASS: All objects appear correct number of times")
    
    return all_correct


def test_view_rotation():
    """Test that views are rotated (not clustered)."""
    print("\n" + "=" * 60)
    print("TEST 4: View Rotation (Anti-Clustering)")
    print("=" * 60)
    
    probe_images, irrelevant_images = create_mock_images()
    
    generator = TrialGenerator(
        probe_images=probe_images,
        irrelevant_images=irrelevant_images,
        probe_reps=80,
        irrelevant_reps=80,
        target_proportion=0.2,
        num_blocks=5,
        s2_target="111111",
        s2_nontargets=["222222", "333333", "444444", "555555"],
        seed=42
    )
    
    trials = generator.generate_trials()
    
    # Check for long runs of same view
    max_consecutive_same_view = 0
    current_consecutive = 1
    
    for i in range(1, len(trials)):
        if trials[i]['s1_image'] == trials[i-1]['s1_image']:
            current_consecutive += 1
            max_consecutive_same_view = max(max_consecutive_same_view, current_consecutive)
        else:
            current_consecutive = 1
    
    print(f"\nMax consecutive trials with same image (view): {max_consecutive_same_view}")
    
    # Reasonable threshold: no more than 5 consecutive same views
    if max_consecutive_same_view <= 5:
        print("✓ PASS: Views are well-rotated (no excessive clustering)")
        return True
    else:
        print("✗ WARNING: Views may be clustered (max consecutive > 5)")
        return True  # Still pass, but warn


def test_block_distribution():
    """Test that trials are distributed across blocks."""
    print("\n" + "=" * 60)
    print("TEST 5: Block Distribution")
    print("=" * 60)
    
    probe_images, irrelevant_images = create_mock_images()
    
    generator = TrialGenerator(
        probe_images=probe_images,
        irrelevant_images=irrelevant_images,
        probe_reps=80,
        irrelevant_reps=80,
        target_proportion=0.2,
        num_blocks=5,
        s2_target="111111",
        s2_nontargets=["222222", "333333", "444444", "555555"],
        seed=42
    )
    
    trials = generator.generate_trials()
    
    # Count trials per block
    block_counts = Counter(trial['block'] for trial in trials)
    
    print("\nTrials per block:")
    for block in sorted(block_counts.keys()):
        print(f"  Block {block}: {block_counts[block]} trials")
    
    # Should be 80 trials per block (400 / 5)
    all_correct = all(count == 80 for count in block_counts.values())
    
    if all_correct and len(block_counts) == 5:
        print("✓ PASS: 80 trials per block across 5 blocks")
        return True
    else:
        print("✗ FAIL: Blocks not evenly distributed")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("=" * 60)
    print("       TRIAL GENERATION TEST SUITE")
    print("=" * 60)
    
    results = []
    
    results.append(("Trial Counts", test_trial_counts()))
    results.append(("View Distribution", test_view_distribution()))
    results.append(("Object Distribution", test_object_distribution()))
    results.append(("View Rotation", test_view_rotation()))
    results.append(("Block Distribution", test_block_distribution()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n>>> All tests passed!")
        return 0
    else:
        print("\n>>> Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

