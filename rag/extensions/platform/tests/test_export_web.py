"""The static file paths must match web/src/api/client.ts (staticPath)."""
from mlsrag.export_web import static_path


def test_static_path():
    assert static_path("/api/results/R/queries", {"offset": 100, "limit": 100}) == \
        "/static-api/results/R/queries/limit-100_offset-100.json"
    assert static_path("/api/configs") == "/static-api/configs.json"
