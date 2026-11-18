"""The course code in rag/src, imported the way rag/scripts do it, plus the project paths."""
import sys
from dataclasses import dataclass
from pathlib import Path

RAG_DIR = Path(__file__).resolve().parents[3]
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from src.context_builder import SimpleContextBuilder  # noqa: F401
from src.data import CourseData  # noqa: F401
from src.encoder import LocalEncoder  # noqa: F401
from src.metrics import mrr_at_k, ndcg_at_k, recall_at_k  # noqa: F401
from src.retriever import ExactDenseRetriever  # noqa: F401


@dataclass(frozen=True)
class Paths:
    rag_dir: Path
    data_dir: Path
    models_dir: Path
    cache_dir: Path
    results_dir: Path


paths = Paths(
    rag_dir=RAG_DIR,
    data_dir=RAG_DIR / "data" / "rag_corpus",
    models_dir=RAG_DIR / "models",
    cache_dir=RAG_DIR / ".cache",
    results_dir=RAG_DIR / "extensions" / "results",
)
