import sys
from pathlib import Path

# root scripts (record_dataset, run_control, ...) are importable in tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
