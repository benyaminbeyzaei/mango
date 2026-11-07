from mlsrag.subsets import SUBSETS, SUBSETS_DIR, load_subset, make_subsets

from conftest import needs_data


def test_subset_sizes():
    sizes = {name: len(load_subset(name)) for name in SUBSETS}
    assert sizes == {
        "practice-smoke20": 20,
        "practice-20pct": 70,
        "dev-20pct": 25,
        "practice-all": 350,
        "dev-all": 125,
    }


@needs_data
def test_make_subsets_reproduces_committed_files(tmp_path):
    make_subsets(tmp_path)
    for name in SUBSETS:
        assert (tmp_path / f"{name}.txt").read_text() == (SUBSETS_DIR / f"{name}.txt").read_text()
