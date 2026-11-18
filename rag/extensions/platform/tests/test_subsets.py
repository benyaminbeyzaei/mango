import pytest

from mlsrag.subsets import SUBSETS, SUBSETS_DIR, check_subset, load_subset, make_subsets

from conftest import needs_data


def test_subset_sizes():
    sizes = {name: len(load_subset(name)) for name in SUBSETS}
    assert sizes == {
        "practice-20": 20,
        "practice-70": 70,
        "test-25": 25,
        "practice": 350,
        "test": 125,
    }


@needs_data
def test_make_subsets_reproduces_committed_files(tmp_path):
    make_subsets(tmp_path)
    for name in SUBSETS:
        assert (tmp_path / f"{name}.txt").read_text() == (SUBSETS_DIR / f"{name}.txt").read_text()


def test_check_subset_refuses_full_splits_without_flag():
    for name in ("practice", "test"):
        with pytest.raises(ValueError, match="--allow-full"):
            check_subset(name, allow_full=False)
        check_subset(name, allow_full=True)
    check_subset("practice-20", allow_full=False)
