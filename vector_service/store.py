"""VectorStore：faiss IndexFlatIP + 记录元数据持久化，线程安全。

内存中同时维护：
  - faiss index：歌曲/歌单统一向量库，512 维 float32
  - meta 列表：与 faiss 顺序对齐的 [{type, ref_id, title, artist, album}]

持久化：
  - data/index.bin：faiss 序列化索引
  - data/meta.json：元数据数组（按 faiss 顺序）

所有 faiss 增/删/搜/重建 + 元数据读写都在 self._lock 保护下；
embedding（慢推理）由调用方在锁外完成后传入 vector 字段。
"""

import json
import os
import tempfile
import threading

import numpy as np


class VectorStore:
    def __init__(self, data_dir: str = "./data", dim: int = 512):
        self._data_dir = data_dir
        self._dim = dim
        self._lock = threading.RLock()
        self._idx = None
        self._meta = []  # [ {type, ref_id, title, artist, album} ]
        self._id_index = {}  # "type:ref_id" -> position in faiss
        os.makedirs(data_dir, exist_ok=True)
        self._load_persisted()

    # ===== 持久化 =====

    def _index_path(self):
        return os.path.join(self._data_dir, "index.bin")

    def _meta_path(self):
        return os.path.join(self._data_dir, "meta.json")

    def _load_persisted(self):
        idx_path, meta_path = self._index_path(), self._meta_path()
        if os.path.exists(idx_path) and os.path.exists(meta_path):
            try:
                import faiss
                self._idx = faiss.read_index(idx_path)
                with open(meta_path, "r", encoding="utf-8") as f:
                    self._meta = json.load(f)
                self._id_index = {
                    f"{m['type']}:{m['ref_id']}": i
                    for i, m in enumerate(self._meta)
                }
            except Exception as exc:  # 索引损坏 → 重置为空库，不崩溃
                self._idx = None
                self._meta = []
                self._id_index = {}
                print(f"[VectorStore] 持久化文件不可用，重置为空库: {exc}")
        if self._idx is None:
            self._fresh_index()

    def _fresh_index(self):
        import faiss
        self._idx = faiss.IndexFlatIP(self._dim)

    def _persist(self):
        idx_path, meta_path = self._index_path(), self._meta_path()
        fd_idx, tmp_idx = tempfile.mkstemp(dir=self._data_dir, suffix=".tmp")
        fd_meta, tmp_meta = tempfile.mkstemp(dir=self._data_dir, suffix=".tmp")
        try:
            os.close(fd_idx)
            os.close(fd_meta)
            import faiss
            faiss.write_index(self._idx, tmp_idx)
            with open(tmp_meta, "w", encoding="utf-8") as f:
                json.dump(self._meta, f, ensure_ascii=False)
            os.replace(tmp_idx, idx_path)
            os.replace(tmp_meta, meta_path)
        finally:
            for p in (tmp_idx, tmp_meta):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass

    # ===== 查询 =====

    def count(self) -> int:
        with self._lock:
            return len(self._meta)

    def search(self, query_vec, top_k: int) -> list:
        """返回 [{type, ref_id, title, artist, album, score}]，得分降序。"""
        with self._lock:
            if self._idx is None or self._idx.ntotal == 0:
                return []
            q = np.asarray([query_vec], dtype="float32")
            k = min(top_k, self._idx.ntotal)
            scores, ilocs = self._idx.search(q, k)
            out = []
            for j in range(k):
                pos = int(ilocs[0][j])
                if pos < 0 or pos >= len(self._meta):
                    continue
                m = self._meta[pos]
                out.append({
                    "type": m["type"],
                    "ref_id": m["ref_id"],
                    "title": m.get("title", ""),
                    "artist": m.get("artist", ""),
                    "album": m.get("album", ""),
                    "score": round(float(scores[0][j]), 6),
                })
            return out

    # ===== 写操作 =====

    def upsert(self, entries):
        """按 (type, ref_id) upsert；entries 每条含 vector。返回处理条数。"""
        with self._lock:
            count = 0
            for e in entries:
                vec = np.asarray([e["vector"]], dtype="float32")
                meta_entry = {
                    "type": e["type"],
                    "ref_id": e["ref_id"],
                    "title": e.get("title", ""),
                    "artist": e.get("artist", ""),
                    "album": e.get("album", ""),
                }
                key = f"{e['type']}:{e['ref_id']}"
                if key in self._id_index:
                    self._expunge_pos(self._id_index[key])
                self._append(vec[0], meta_entry)
                count += 1
            if count > 0:
                self._persist()
            return count

    def delete_by_type(self, etype, ref_ids):
        with self._lock:
            removed = 0
            for rid in ref_ids:
                key = f"{etype}:{rid}"
                if key in self._id_index:
                    self._expunge_pos(self._id_index[key])
                    removed += 1
            if removed > 0:
                self._persist()
            return removed

    def rebuild(self, entries):
        """原子整体重建。entries 每条含 vector。返回条数。"""
        with self._lock:
            self._fresh_index()
            self._meta = []
            self._id_index = {}
            vecs = []
            for e in entries:
                vecs.append(e["vector"])
                self._meta.append({
                    "type": e["type"],
                    "ref_id": e["ref_id"],
                    "title": e.get("title", ""),
                    "artist": e.get("artist", ""),
                    "album": e.get("album", ""),
                })
            if vecs:
                arr = np.asarray(vecs, dtype="float32")
                self._idx.add(arr)
                self._id_index = {
                    f"{m['type']}:{m['ref_id']}": i
                    for i, m in enumerate(self._meta)
                }
            self._persist()
            return len(self._meta)

    # ===== 内部 =====

    def _append(self, vec, meta_entry):
        self._idx.add(np.asarray([vec], dtype="float32"))
        pos = len(self._meta)
        self._meta.append(meta_entry)
        self._id_index[f"{meta_entry['type']}:{meta_entry['ref_id']}"] = pos

    def _expunge_pos(self, pos):
        """移除位置 pos 处的向量与记录。faiss 序被破坏后整体重建同序索引（写路径低频）。"""
        del self._meta[pos]

        vecs = []
        read_idx = self._idx
        for j in range(read_idx.ntotal):
            if j == pos:
                continue
            vecs.append(read_idx.reconstruct(j))

        self._fresh_index()
        if vecs:
            self._idx.add(np.asarray(vecs, dtype="float32"))
        self._id_index = {
            f"{m['type']}:{m['ref_id']}": i
            for i, m in enumerate(self._meta)
        }