"""Fixed query-id subsets, committed as text files so every run uses the same queries."""
import random
from pathlib import Path

from .course import CourseData, paths

SUBSETS_DIR = Path(__file__).resolve().parents[1] / "subsets"
SEED = 0
# name -> (split, number of queries; None = all)
SUBSETS = {
    "practice-smoke20": ("practice", 20),
    "practice-20pct": ("practice", 70),
    "dev-20pct": ("dev", 25),
    "practice-all": ("practice", None),
    "dev-all": ("dev", None),
}


def load_subset(name) -> list[str]:
    path = SUBSETS_DIR / f"{name}.txt"
    if not path.exists():
        raise ValueError(f"unknown subset {name!r}; files are in {SUBSETS_DIR}")
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def make_subsets(out_dir=SUBSETS_DIR):
    """Write every subset file; ids keep the manifest order."""
    data = CourseData(paths.data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (split, n) in SUBSETS.items():
        ids = [row["query_id"] for row in data.audio_items(split)]
        if n is not None:
            chosen = set(random.Random(SEED).sample(ids, n))
            ids = [q for q in ids if q in chosen]
        (out_dir / f"{name}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
