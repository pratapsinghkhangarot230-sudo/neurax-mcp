"""Local text embeddings with fastembed. The model must output 384 dims to match the schema."""

from functools import cache

from fastembed import TextEmbedding

from neurax import config


@cache
def _model() -> TextEmbedding:
    # Loaded once per process: first call downloads/loads the ONNX model (slow).
    return TextEmbedding(model_name=config.EMBED_MODEL, cache_dir=config.MODEL_CACHE)


def embed(texts: list[str]) -> list[list[float]]:
    return [vec.tolist() for vec in _model().embed(texts)]


def to_pgvector(vec: list[float]) -> str:
    """pgvector's text format, so we can pass vectors as `%s::vector` without an extra package."""
    return "[" + ",".join(map(str, vec)) + "]"
