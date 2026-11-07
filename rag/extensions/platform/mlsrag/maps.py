"""Linear maps from one corpus encoder's space to another's, fitted on the aligned corpus vectors."""
import json
import time

import numpy as np
from scipy.linalg import orthogonal_procrustes

from .course import CourseData, paths

MAPS_DIR = paths.cache_dir / "maps"


def map_path(src, dst, method):
    return MAPS_DIR / f"{src}-to-{dst}-{method}.npy"


def normalise(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def fit(a, b, method):
    """W such that a @ W approximates b (rows are aligned records)."""
    if method == "lstsq":
        return np.linalg.lstsq(a, b, rcond=None)[0]
    if method == "procrustes":
        return orthogonal_procrustes(a, b)[0]
    raise ValueError(f"unknown map method {method!r}")


def mean_cosine(a, b, w):
    return float(np.mean(np.sum(normalise(a @ w) * b, axis=1)))


def fit_map(src, dst, method, exclude_dev=True):
    """Fit, save the map and a sidecar JSON. The cosines are measured on the dev-relevant records,
    which are held out only with exclude_dev (otherwise `dev_rows_in_fit` says they are in-sample)."""
    emb = paths.data_dir / "embeddings"
    a = np.load(emb / f"corpus_{src}.npy").astype(np.float32)
    b = np.load(emb / f"corpus_{dst}.npy").astype(np.float32)
    ids_a = json.loads((emb / f"corpus_{src}_ids.json").read_text())
    ids_b = json.loads((emb / f"corpus_{dst}_ids.json").read_text())
    assert ids_a == ids_b, "corpus vectors are not in the same row order"

    dev_ids = {rid for rels in CourseData(paths.data_dir).load_qrels("dev").values() for rid in rels}
    dev = np.array([rid in dev_ids for rid in ids_a])
    train = ~dev if exclude_dev else np.ones(len(ids_a), dtype=bool)

    start = time.perf_counter()
    w = fit(a[train], b[train], method).astype(np.float32)
    fit_s = time.perf_counter() - start

    info = {
        "src": src, "dst": dst, "method": method, "dev_rows_in_fit": not exclude_dev,
        "fit_seconds": round(fit_s, 3),
        "rows_fit": int(train.sum()), "dev_rows": int(dev.sum()),
        "dev_rows_cosine_mapped": mean_cosine(a[dev], b[dev], w),
        "dev_rows_cosine_unmapped": mean_cosine(a[dev], b[dev], np.eye(a.shape[1])),
    }
    out = map_path(src, dst, method)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, w)
    out.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return out, info
