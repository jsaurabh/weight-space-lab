"""Prepare the common VM directory without uploading any candidate or test data."""
from pathlib import Path

Path("/content/qwen-comparison").mkdir(exist_ok=True)
print("Comparison VM directory ready")
