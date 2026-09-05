"""sentence-transformers 模型封装：懒加载 + 文本向量化 + L2 归一化。

独立的 EMBEDDER 类，供 store.py 在锁外调用（推理慢，不能占锁）。
模型默认 BAAI/bge-small-zh-v1.5（512 维，中文语义检索）。
"""

import os

_MODEL_NAME = os.environ.get("VECTOR_MODEL", "BAAI/bge-small-zh-v1.5")

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(_MODEL_NAME)
    return _embedder


def is_model_loaded() -> bool:
    return _embedder is not None


def get_dim() -> int:
    """返回模型输出维度（需模型已加载，否则返回 None -> 由调用方 503）。"""
    if _embedder is None:
        return None
    return _embedder.get_sentence_embedding_dimension()


def embed_texts(texts) -> list:
    """把一批文本编码为 L2 归一化后的向量列表（Python list of list[float]）。

    normalize_embeddings=True 做 L2 归一化。配合 faiss IndexFlatIP（内积）
    即得余弦相似度。
    """
    import numpy as np
    if not texts:
        return []
    emb = _get_embedder().encode(
        list(texts),
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )
    return np.asarray(emb, dtype="float32").tolist()


def embed_one(text: str) -> list:
    vecs = embed_texts([text])
    return vecs[0] if vecs else []