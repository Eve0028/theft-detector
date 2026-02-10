"""
Setup Verification Script
==========================

Verifies that the project structure is complete and all files are in place.
Run this before setting up venv to ensure everything is ready.

Usage
-----
::

    python verify_setup.py

"""

import os
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists and report."""
    if os.path.exists(filepath):
        print(f"  [OK] {description}")
        return True
    else:
        print(f"  [MISSING] {description}: {filepath}")
        return False


def check_directory_exists(dirpath, description):
    """Check if a directory exists and report."""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        print(f"  [OK] {description}")
        return True
    else:
        print(f"  [MISSING] {description}: {dirpath}")
        return False


def count_images(image_dir):
    """Count image files in directory."""
    if not os.path.exists(image_dir):
        return 0
    
    extensions = ('.jpg', '.jpeg', '.png')
    count = 0
    for f in os.listdir(image_dir):
        if f.lower().endswith(extensions):
            count += 1
    return count


def main():
    """Run verification checks."""
    print("=" * 60)
    print("P300-CIT PROJECT SETUP VERIFICATION")
    print("=" * 60)
    
    all_checks = []
    
    # Check directories
    print("\n[1] Checking directory structure...")
    all_checks.append(check_directory_exists("config", "Config directory"))
    all_checks.append(check_directory_exists("src", "Source code directory"))
    all_checks.append(check_directory_exists("scripts", "Scripts directory"))
    all_checks.append(check_directory_exists("images", "Images directory"))
    
    # Check core files
    print("\n[2] Checking core files...")
    all_checks.append(check_file_exists("requirements.txt", "Requirements file"))
    all_checks.append(check_file_exists("README.md", "README (complete documentation)"))
    all_checks.append(check_file_exists(".gitignore", "Git ignore file"))
    
    # Check config files
    print("\n[3] Checking configuration files...")
    all_checks.append(check_file_exists("config/experiment_config.yaml", "Experiment config"))
    
    # Check source code
    print("\n[4] Checking source code...")
    all_checks.append(check_file_exists("src/__init__.py", "Source package init"))
    all_checks.append(check_file_exists("src/experiment.py", "Main experiment script"))
    all_checks.append(check_file_exists("src/trial_generator.py", "Trial generator"))
    all_checks.append(check_file_exists("src/lsl_markers.py", "LSL markers module"))
    all_checks.append(check_file_exists("src/utils.py", "Utils module"))
    
    # Check scripts
    print("\n[5] Checking helper scripts...")
    all_checks.append(check_file_exists("scripts/__init__.py", "Scripts package init"))
    all_checks.append(check_file_exists("scripts/normalize_images.py", "Image normalization script"))
    all_checks.append(check_file_exists("scripts/generate_metadata.py", "Metadata generation script"))
    all_checks.append(check_file_exists("scripts/test_trial_generation.py", "Trial generation test"))
    
    # Check launcher scripts
    print("\n[6] Checking launcher scripts...")
    all_checks.append(check_file_exists("run_experiment.bat", "Windows launcher"))
    all_checks.append(check_file_exists("run_experiment.sh", "Linux/Mac launcher"))
    
    # Check images
    print("\n[7] Checking stimulus images...")
    image_count = count_images("images")
    print(f"  Found {image_count} image files in images/")
    
    if image_count >= 10:  # Expected: 2 views × 5 objects = 10 images
        print(f"  [OK] Sufficient images found ({image_count})")
        all_checks.append(True)
    elif image_count > 0:
        print(f"  [WARNING] Only {image_count} images found (expected ~10)")
        all_checks.append(True)
    else:
        print(f"  [ERROR] No images found!")
        all_checks.append(False)
    
    # Check for expected image files
    expected_images = [
        "probe_pendrive_view1.jpg",
        "probe_pendrive_view2.jpg",
        "irr_mouse_view1.jpg",
        "irr_mouse_view2.jpg",
        "irr_headphones_view1.jpg",
        "irr_headphones_view2.jpg",
        "irr_cableusb_view1.jpg",
        "irr_cableusb_view2.jpg",
        "irr_charger_view1.jpg",
        "irr_charger_view2.jpg"
    ]
    
    print("\n[8] Checking specific stimulus images...")
    for img in expected_images:
        img_path = os.path.join("images", img)
        exists = os.path.exists(img_path)
        if exists:
            print(f"  [OK] {img}")
        else:
            print(f"  [MISSING] {img}")
        all_checks.append(exists)
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    total_checks = len(all_checks)
    passed_checks = sum(1 for check in all_checks if check)
    
    print(f"\nPassed: {passed_checks}/{total_checks} checks")
    
    if passed_checks == total_checks:
        print("\n[SUCCESS] All checks passed!")
        print("\nNext steps:")
        print("1. Create virtual environment: python -m venv venv")
        print("2. Activate venv:")
        print("   - Windows: venv\\Scripts\\activate")
        print("   - Linux/Mac: source venv/bin/activate")
        print("3. Install dependencies: pip install -r requirements.txt")
        print("4. Normalize images: python scripts/normalize_images.py")
        print("5. Generate metadata: python scripts/generate_metadata.py")
        print("6. Run test: python scripts/test_trial_generation.py")
        print("\nSee README.md for complete documentation.")
        return 0
    else:
        print("\n[WARNING] Some checks failed!")
        print("Please ensure all files are in place before proceeding.")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
