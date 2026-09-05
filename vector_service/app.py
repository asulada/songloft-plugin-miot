"""FastAPI 向量语义检索服务。

对外提供：
  GET  /health                健康检查 + 统计
  POST /api/embed             文本 -> L2 归一化向量
  POST /api/search            query -> top-k 相似条目（歌曲/歌单混排，得分降序）
  POST /api/upsert            按 (type, ref_id) 增量 upsert（可单条或批量）
  POST /api/delete_by_type    按 type + ref_ids 删除
  POST /api/rebuild           全量原子重建（doRefresh 后由插件调用）

鉴权：VECTOR_SERVICE_TOKEN 环境变量（可选）。设置则要求 Authorization: Bearer <token>，
否则返回 401。插件始终带该头。
"""

import os
import threading

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from text_embed import embed_texts, is_model_loaded, get_dim
from store import VectorStore

app = FastAPI(title="Songloft Vector Search", version="0.1")

# ===== 配置 =====
TOKEN = os.environ.get("VECTOR_SERVICE_TOKEN", "").strip()
DATA_DIR = os.environ.get("VECTOR_DATA_DIR", "./data")

MODEL_WARMUP_LOCK = threading.Lock()
_model_warmed = False


def _require_auth(authorization=None):
    if not TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    provided = authorization[len("Bearer "):].strip()
    # 常数时间比较防时序攻击
    if not _const_eq(provided, TOKEN):
        raise HTTPException(status_code=401, detail="invalid token")


def _const_eq(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    return sum(x ^ y for x, y in zip(a.encode(), b.encode())) == 0


def _warm_model():
    """懒加载并预热模型（读写路径低频，允许首次阻塞该请求，其它请求 503）。"""
    global _model_warmed
    if _model_warmed:
        return True
    with MODEL_WARMUP_LOCK:
        if _model_warmed:
            return True
        try:
            dim = get_dim()  # 触发加载
            _store = app.state.store
            if dim and dim != _store._dim and _store.count() == 0:
                _store._dim = dim
            _model_warmed = True
            return True
        except Exception as exc:
            print(f"[VectorService] 模型加载失败: {exc}")
            return False


@app.on_event("startup")
async def _startup():
    dim = 512
    from text_embed import get_dim as _g
    try:
        loaded_dim = _g()
        if loaded_dim:
            dim = loaded_dim
    except Exception:
        pass
    app.state.store = VectorStore(DATA_DIR, dim=dim)
    print(f"[VectorService] 已就绪 data_dir={DATA_DIR} dim={app.state.store._dim} count={app.state.store.count()}")


# ===== 请求/响应模型 =====

class EmbedReq(BaseModel):
    texts: list = Field(default_factory=list)
    text: str | None = None


class SearchReq(BaseModel):
    query: str
    top_k: int = 5


class UpsertEntry(BaseModel):
    type: str
    ref_id: int
    title: str = ""
    artist: str = ""
    album: str = ""


class UpsertReq(BaseModel):
    entries: list = Field(default_factory=list)
    # 兼容单条形态
    type: str | None = None
    ref_id: int | None = None
    title: str = ""
    artist: str = ""
    album: str = ""


class DeleteReq(BaseModel):
    type: str
    ref_ids: list = Field(default_factory=list)


class RebuildReq(BaseModel):
    songs: list = Field(default_factory=list)
    playlists: list = Field(default_factory=list)


def _ok(data=None, msg="ok"):
    return {"code": 0, "msg": msg, "data": data}


# ===== 路由 =====

@app.get("/health")
def health(authorization: str | None = Header(default=None)):
    _require_auth(authorization)
    store = app.state.store
    return _ok({
        "dim": store._dim,
        "count": store.count(),
        "model_loaded": is_model_loaded(),
    })


@app.post("/api/embed")
def embed_route(body: EmbedReq, authorization: str | None = Header(default=None)):
    _require_auth(authorization)
    if not _warm_model():
        raise HTTPException(status_code=503, detail="model not loaded")
    texts = body.texts if body.texts else ([body.text] if body.text else [])
    if not texts:
        raise HTTPException(status_code=400, detail="empty texts")
    vecs = embed_texts(texts)
    return _ok({"dim": get_dim(), "vectors": vecs})


@app.post("/api/search")
def search_route(body: SearchReq, authorization: str | None = Header(default=None)):
    _require_auth(authorization)
    if not _warm_model():
        raise HTTPException(status_code=503, detail="model not loaded")
    if not (body.query or "").strip():
        raise HTTPException(status_code=400, detail="empty query")
    top_k = max(1, min(20, int(body.top_k)))
    store = app.state.store
    q_vec = embed_texts([body.query])[0]
    results = store.search(q_vec, top_k)
    return _ok({"results": results})


@app.post("/api/upsert")
def upsert_route(body: UpsertReq, authorization: str | None = Header(default=None)):
    _require_auth(authorization)
    if not _warm_model():
        raise HTTPException(status_code=503, detail="model not loaded")
    entries = body.entries
    if not entries and body.type and body.ref_id is not None:
        entries = [{
            "type": body.type,
            "ref_id": body.ref_id,
            "title": body.title,
            "artist": body.artist,
            "album": body.album,
        }]
    if not entries:
        raise HTTPException(status_code=400, detail="empty entries")
    texts = []
    for e in entries:
        if e.get("type") == "playlist":
            texts.append(f"[歌单] {e.get('title','')}".strip())
        else:
            texts.append(f"{e.get('title','')} {e.get('artist','')} {e.get('album','')}".strip())
    vecs = embed_texts(texts)
    store = app.state.store
    to_store = []
    for e, vec in zip(entries, vecs):
        to_store.append({
            "type": e["type"],
            "ref_id": int(e["ref_id"]),
            "title": e.get("title", ""),
            "artist": e.get("artist", ""),
            "album": e.get("album", ""),
            "vector": vec,
        })
    n = store.upsert(to_store)
    return _ok({"count": n})


@app.post("/api/delete_by_type")
def delete_route(body: DeleteReq, authorization: str | None = Header(default=None)):
    _require_auth(authorization)
    store = app.state.store
    ids = [int(x) for x in body.ref_ids if x is not None]
    n = store.delete_by_type(body.type, ids)
    return _ok({"count": n})


@app.post("/api/rebuild")
def rebuild_route(body: RebuildReq, authorization: str | None = Header(default=None)):
    _require_auth(authorization)
    if not _warm_model():
        raise HTTPException(status_code=503, detail="model not loaded")
    entries = []
    for s in body.songs:
        entries.append({
            "type": "song",
            "ref_id": int(s["id"]),
            "title": s.get("title", ""),
            "artist": s.get("artist", ""),
            "album": s.get("album", ""),
            "vector": None,
        })
    for p in body.playlists:
        entries.append({
            "type": "playlist",
            "ref_id": int(p["id"]),
            "title": p.get("name", ""),
            "artist": "",
            "album": "",
            "vector": None,
        })
    if not entries:
        # 空库重建：清空但不编码
        store = app.state.store
        n = store.rebuild([])
        return _ok({"count": n})

    texts = []
    for e in entries:
        if e["type"] == "playlist":
            texts.append(f"[歌单] {e['title']}")
        else:
            texts.append(f"{e['title']} {e['artist']} {e['album']}".strip())
    vecs = embed_texts(texts)
    store = app.state.store
    for e, vec in zip(entries, vecs):
        e["vector"] = vec
    n = store.rebuild(entries)
    return _ok({"count": n})