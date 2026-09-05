# Songloft 外部向量语义检索服务

独立的 FastAPI + faiss + sentence-transformers 服务，为 Songloft MIoT 语音插件提供**外部语义召回**：
语音搜歌在本地字面匹配 miss 时，把 query 语义化后在本地歌曲/歌单库中按余弦相似度召回。

## 思想

- `bge-small-zh-v1.5`（512 维中文向量），两侧（入库文本 / 查询文本）都做 L2 归一化
- `faiss-cpu` 的 `IndexFlatIP`（内积 + L2 归一化 = 余弦相似度）
- 歌曲条目与歌单条目统一建库（`type: "song" | "playlist"`），检索时**歌单+歌曲混排、得分降序**

## 安装与运行

```bash
cd vector_service
pip install -r requirements.txt

# 首次运行会自动下载模型 BAAI/bge-small-zh-v1.5（约 100MB）
# 若下载受限，可设镜像：
#   export HF_ENDPOINT=https://hf-mirror.com

uvicorn app:app --host 127.0.0.1 --port 8710
```

可选环境变量：

| 变量 | 说明 |
|---|---|
| `VECTOR_SERVICE_TOKEN` | 可选共享密钥；设置后所有接口需 `Authorization: Bearer <token>` |
| `VECTOR_DATA_DIR` | 持久化目录，默认 `./data`（gitignore） |
| `VECTOR_MODEL` | 模型名，默认 `BAAI/bge-small-zh-v1.5` |

## systemd 部署（Debian 12）

把服务托给 systemd 管理，日志写文件并带时间戳，不写 journald。相关文件在 `deploy/systemd/`：

```
vector_service/deploy/systemd/
├── songloft-vector.service          # systemd unit
└── log_conf.json                    # uvicorn 日志配置（asctime 时间戳 + 落文件）
```

### 安装

```bash
sudo cp /mnt/nfts1/py/vector_service/deploy/systemd/songloft-vector.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now songloft-vector

# 状态 + 日志
sudo systemctl status songloft-vector --no-pager
tail -f /run/songloft-vector/error.log
tail -f /run/songloft-vector/access.log
```

### 行为要点

- **日志路径**：`/run/songloft-vector/{error,access}.log`（RotatingFileHandler，10MB × 3 备份）。
- **时间戳**：`log_conf.json` 的 formatter 带 `%(asctime)s`，格式 `%Y-%m-%d %H:%M:%S %z`，`access` 与 `error` 均带。
- **监听**：`--host 0.0.0.0 --port 8710`，保证插件容器可经网关注口访问。
- **目录生命周期**：`/run` 是 tmpfs，`ExecStartPre=mkdir -p` 只保证目录存在；**不随服务停止清理**，仅当容器/宿主重启（重新挂载 tmpfs）才清空。日志因此是**临时诊断用途**。

| 操作 | `/run/songloft-vector` 日志 |
|---|---|
| `systemctl restart songloft-vector` | 保留 |
| `docker restart <容器>` / `docker stop/start` | 清空（tmpfs 重挂） |
| 宿主机关机/重启 | 清空 |

### 需要持久化日志？

把 `log_conf.json` 里的 `/run/songloft-vector/…` 改为 `/var/log/songloft-vector/…`，并自行保证目录存在（可新建独立目录并替换 unit 里 `ExecStartPre=/bin/mkdir -p /run/songloft-vector` 为 `mkdir -p /var/log/songloft-vector`），即可在容器重启后保留。

## API（`/api`，JSON）

| 端点 | 请求 | 返回 |
|---|---|---|
| `GET /health` | - | `{dim,count,model_loaded}` |
| `POST /embed` | `{"texts":[...]}` 或 `{"text":".."}` | `{dim, vectors:[[...]...]}` |
| `POST /search` | `{"query":"儿童歌曲","top_k":5}` | `{results:[{type,ref_id,title,artist,album,score}]}` 得分降序 |
| `POST /upsert` | 单条 `{type,ref_id,title,...}` 或批量 `{entries:[..]}` | `{count}` |
| `POST /delete_by_type` | `{type:"song",ref_ids:[..]}` | `{count}` |
| `POST /rebuild` | `{songs:[{id,title,artist,album}],playlists:[{id,name}]}` | `{count}` |

示例：

```bash
curl -s http://127.0.0.1:8710/health
curl -s -X POST http://127.0.0.1:8710/api/rebuild -H 'Content-Type: application/json' \
  -d '{"songs":[{"id":1,"title":"儿歌","artist":"\u513f\u54e5","album":""}],"playlists":[{"id":10,"name":"儿歌"}]}'
curl -s -X POST http://127.0.0.1:8710/api/search -H 'Content-Type: application/json' \
  -d '{"query":"儿童歌曲","top_k":5}'
```

## 数据与持久化

- `data/index.bin`：faiss 索引；`data/meta.json`：元数据数组（按 faiss 顺序）。
- 写路径用临时文件 `os.replace` 原子落盘。文件损坏时服务启动会重置为空库（不崩溃）。
- 内存量级：512 维 float32 ≈ `count*512*4` 字节，万级 ≈ 20MB。

## 与 Songloft 插件的关系

由插件的 `IndexingManager` 在 `doRefresh` / `addImportedSong` 时推送建库 / 增量 upsert；
语音本地 literal miss 时插件调用 `/search` 语义召回，并按 `ref_id` 直接播放。
服务不可达时插件**静默降级**为现有外搜，不影响播放。