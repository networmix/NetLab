from __future__ import annotations

import sys
from pathlib import Path

# Import netlab from the repository when running tests directly.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
