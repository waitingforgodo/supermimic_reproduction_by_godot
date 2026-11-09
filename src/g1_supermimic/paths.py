"""Portable project data and output locations."""
import os
from pathlib import Path
ROOT=Path(os.environ.get('G1_SUPERMIMIC_HOME',Path.cwd())).expanduser().resolve()
RUNS=ROOT/'outputs/runs'
