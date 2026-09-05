import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_deck.tif"


@pytest.fixture(scope="session")
def synthetic_deck() -> Path:
    """합성 정사영상. 용량이 커서 저장소에 넣지 않고 없으면 생성한다."""
    if not FIXTURE.is_file():
        subprocess.run(
            [sys.executable, str(ROOT / "tools" / "make_synthetic.py"), "-o", str(FIXTURE)],
            check=True,
            cwd=ROOT,
        )
    return FIXTURE
