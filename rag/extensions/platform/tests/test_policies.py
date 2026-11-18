import numpy as np
import pytest

from mlsrag import maps, stages
from mlsrag.config import RunConfig
from mlsrag.maps import fit, mean_cosine
from mlsrag.stages import BGE_QUERY_PREFIX, GatewayEncoder, ReEncode, Translate, build_encoder

from conftest import needs_data


class FakeGateway:
    """Encodes text to a fixed toy vector and remembers what it was asked."""

    def __init__(self):
        self.calls = []

    def encoder(self, name):
        return None

    def embed(self, name, texts, role="query"):
        self.calls.append((name, texts[0]))
        return np.array([[3.0, 4.0]], dtype=np.float32)


def cfg(policy, **params):
    return RunConfig(name="x", query_encoder="gte-small", corpus_index="enc_b",
                     mismatch_policy=policy, params=params)


def test_as_is_and_matching_use_the_query_encoder():
    g = FakeGateway()
    enc = build_encoder(cfg("as-is"), g)
    assert isinstance(enc, GatewayEncoder) and enc.name == "gte-small"
    matching = RunConfig(name="x", mismatch_policy="translate")  # gte-small on enc_a: nothing to do
    assert isinstance(build_encoder(matching, g), GatewayEncoder)


def test_re_encode_picks_the_index_encoder_and_prefix():
    g = FakeGateway()
    build_encoder(cfg("re-encode"), g).encode("hello")
    build_encoder(cfg("re-encode", bge_prefix=True), g).encode("hello")
    assert g.calls == [("bge-small", "hello"), ("bge-small", BGE_QUERY_PREFIX + "hello")]
    back = RunConfig(name="x", query_encoder="bge-small", corpus_index="enc_a", mismatch_policy="re-encode")
    enc = build_encoder(back, g)
    assert isinstance(enc, ReEncode) and enc.inner.name == "gte-small"
    with pytest.raises(ValueError, match="bge-small"):
        build_encoder(back.model_copy(update={"params": {"bge_prefix": True}}), g)


def test_translate_applies_the_map_and_normalises(tmp_path, monkeypatch):
    w = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)  # swap the two coordinates
    np.save(tmp_path / "m.npy", w)
    enc = Translate(GatewayEncoder(FakeGateway(), "gte-small"), tmp_path / "m.npy")
    np.testing.assert_allclose(enc.encode("q"), [0.8, 0.6], rtol=1e-6)

    monkeypatch.setattr(stages, "map_path", lambda src, dst, method: tmp_path / f"{src}-{dst}-{method}.npy")
    with pytest.raises(FileNotFoundError, match="fit-map"):
        build_encoder(cfg("translate", map="procrustes"), FakeGateway())
    np.save(tmp_path / "enc_a-enc_b-procrustes.npy", w)
    assert isinstance(build_encoder(cfg("translate", map="procrustes"), FakeGateway()), Translate)


def test_fit_recovers_a_known_rotation():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((200, 8))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    r = np.linalg.qr(rng.standard_normal((8, 8)))[0]
    for method in ("lstsq", "procrustes"):
        w = fit(a, a @ r, method)
        np.testing.assert_allclose(w, r, atol=1e-6)
        assert mean_cosine(a, a @ r, w) == pytest.approx(1.0)


@needs_data
def test_fit_map_excludes_dev_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(maps, "MAPS_DIR", tmp_path)
    out, info = maps.fit_map("enc_a", "enc_b", "procrustes", exclude_dev=True)
    assert out == tmp_path / "enc_a-to-enc_b-procrustes.npy" and out.with_suffix(".json").exists()
    assert not info["dev_rows_in_fit"] and info["rows_fit"] + info["dev_rows"] == 24000
    assert info["dev_rows_cosine_mapped"] > info["dev_rows_cosine_unmapped"]
