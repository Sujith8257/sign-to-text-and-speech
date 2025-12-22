"""
Script to identify and delete unwanted files
"""
from pathlib import Path
import os

# Files to delete
files_to_delete = [
    "inference_indian.py",  # Outdated, has hardcoded paths that don't exist
    "class_names.json",      # Not used by app.py (labels are hardcoded)
    "generate_dataset_visualization.py",  # Utility script, already generated the image
    "requirements_training.txt",  # Redundant, requirements.txt covers everything
]

print("=" * 60)
print("Cleaning up unwanted files...")
print("=" * 60)

deleted_count = 0
not_found_count = 0

for file_path in files_to_delete:
    path = Path(file_path)
    if path.exists():
        try:
            size = path.stat().st_size
            path.unlink()
            print(f"[DELETED] {file_path} ({size} bytes)")
            deleted_count += 1
        except Exception as e:
            print(f"[ERROR] Could not delete {file_path}: {e}")
    else:
        print(f"[SKIP] {file_path} (not found)")
        not_found_count += 1

print("=" * 60)
print(f"Summary: {deleted_count} files deleted, {not_found_count} not found")
print("=" * 60)
