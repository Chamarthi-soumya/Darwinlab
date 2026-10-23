import sys
from pathlib import Path

# Add project root to sys.path so tests can import src.* cleanly
root = Path(__file__).parent.resolve()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
