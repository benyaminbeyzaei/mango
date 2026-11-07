import pytest

from mlsrag.course import paths

needs_data = pytest.mark.skipif(
    not (paths.data_dir / "corpus.jsonl").exists(), reason="course data not installed"
)
