"""pytest conftest — configures pytest for the test suite."""

import sys
from pathlib import Path

# Thêm thư mục gốc của project (cha của thư mục `tests/`) vào sys.path
# Điều này cho phép các file test gọi `from src.xxx import yyy` bình thường.
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))
