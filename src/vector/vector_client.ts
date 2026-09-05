// 向量语义检索客户端
// 对接 vector_service/（FastAPI + bge-small-zh-v1.5 + faiss）。
// 纪律：绝不让语音链路抛错——所有方法 try/catch，失败返回 [] / false / null，
// 首次失败 warn 日志（[VectorSearcher] ...）。
//
// URL 只能是完整 http(s) 地址（配置填写），**不做 loopback fallback**。

/// <reference types="@songloft/plugin-sdk" />

import type { ConfigManager } from '../config/manager';

/** 向量召回的命中条目（歌单/歌曲统一，得分降序） */
export interface VectorHit {
  type: 'playlist' | 'song';
  refId: number;
  title: string;
  artist: string;
  album?: string;
  score: number;
}

/** 建库/增量推送的条目（无 vector，由服务端编码） */
export interface VectorEntry {
  type: 'playlist' | 'song';
  ref_id: number;
  title: string;
  artist?: string;
  album?: string;
}

/** 供 /rebuild 的输入：歌曲数组 + 歌单数组 */
export interface VectorLibrary {
  songs: Array<{ id: number; title: string; artist?: string; album?: string }>;
  playlists: Array<{ id: number; name: string }>;
}

const BATCH_SIZE = 200;

export class VectorSearcher {
  private configManager: ConfigManager;
  /** 并发 rebuild/全量推送守卫：只放一次在飞 */
  private syncInFlight: Promise<void> | null = null;
  /** 冷期静默：同一次故障窗口只 warn 一次，避免刷屏 */
  private lastWarnKey = '';
  private lastWarnAt = 0;

  constructor(configManager: ConfigManager) {
    this.configManager = configManager;
  }

  /** 是否已启用（总开关 + URL 非空） */
  async isEnabled(): Promise<boolean> {
    try {
      const cfg = await this.configManager.getConfig();
      return !!cfg.vector_service_enabled && (cfg.vector_service_url || '').trim() !== '';
    } catch {
      return false;
    }
  }

  /** 认证 Token：配置 token 或插件 token */
  private async resolveToken(): Promise<string> {
    try {
      const cfg = await this.configManager.getConfig();
      const t = (cfg.vector_service_token || '').trim();
      if (t) return t;
    } catch {}
    try {
      const pluginToken = await songloft.plugin.getToken();
      return `Bearer ${pluginToken}`;
    } catch {
      return '';
    }
  }

  private async getEnabledConfig(): Promise<{ url: string; topK: number; timeoutMs: number } | null> {
    try {
      const cfg = await this.configManager.getConfig();
      if (!cfg.vector_service_enabled) return null;
      const url = (cfg.vector_service_url || '').trim();
      if (!url) return null;
      const topK = Math.max(1, Math.min(20, cfg.vector_service_top_k ?? 5));
      const timeoutSec = Math.max(1, Math.min(60, cfg.vector_service_timeout ?? 3));
      return { url, topK, timeoutMs: timeoutSec * 1000 };
    } catch {
      return null;
    }
  }

  /** 统一 POST：超时（Promise.race，QuickJS 无 AbortController）+ json + code===0 校验。失败返回 null。 */
  private async post(path: string, body: unknown, timeoutMs: number): Promise<any | null> {
    const cfg = await this.getEnabledConfig();
    if (!cfg) return null;
    const token = await this.resolveToken();
    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error('AbortError')), timeoutMs);
    });
    try {
      const fetchPromise = fetch(cfg.url + path, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: token } : {}),
        },
        body: JSON.stringify(body),
      });
      const resp = await Promise.race([fetchPromise, timeoutPromise]);
      const text = await resp.text();
      let parsed: any;
      try {
        parsed = JSON.parse(text);
      } catch {
        this.warnOnce('post', `bad json status=${resp.status}`);
        return null;
      }
      if (!parsed || parsed.code !== 0) {
        this.warnOnce('post', `code=${parsed?.code} msg=${parsed?.msg} path=${path}`);
        return null;
      }
      return parsed.data ?? null;
    } catch (e: any) {
      if (e?.message === 'AbortError') {
        this.warnOnce('post', `timeout>${timeoutMs / 1000}s ${path}`);
      } else {
        this.warnOnce('post', `fetch error: ${String(e)} ${path}`);
      }
      return null;
    }
  }

  /** 冷期去重的 warn 日志（同一 key 60s 内仅一条） */
  private warnOnce(key: string, message: string): void {
    const now = Date.now();
    if (this.lastWarnKey === key && now - this.lastWarnAt < 60_000) {
      return;
    }
    this.lastWarnKey = key;
    this.lastWarnAt = now;
    songloft.log.warn(`[VectorSearcher] ${message}`);
  }

  /**
   * 语义召回：query -> top-k（歌单+歌曲混排，得分降序）。
   * 永不抛出；失败/服务不可达返回 []。
   */
  async search(query: string, topK?: number): Promise<VectorHit[]> {
    const cfg = await this.getEnabledConfig();
    if (!cfg) return [];
    const k = topK ?? cfg.topK;
    const data = await this.post('/api/search', { query, top_k: k }, cfg.timeoutMs);
    if (!data || !Array.isArray(data.results)) return [];
    return data.results
      .filter((r: any) => r && r.ref_id !== undefined)
      .map((r: any) => ({
        type: r.type === 'playlist' ? ('playlist' as const) : ('song' as const),
        refId: Number(r.ref_id),
        title: String(r.title || ''),
        artist: String(r.artist || ''),
        album: r.album ? String(r.album) : undefined,
        score: Number(r.score || 0),
      }));
  }

  /** 增量 upsert 一批（addImportedSong 等）。fire-and-forget 由调用方决定。 */
  async pushEntries(entries: VectorEntry[]): Promise<void> {
    const cfg = await this.getEnabledConfig();
    if (!cfg || entries.length === 0) return;
    // 分批发，避免超大 body
    for (let i = 0; i < entries.length; i += BATCH_SIZE) {
      const chunk = entries.slice(i, i + BATCH_SIZE);
      await this.post('/api/upsert', { entries: chunk }, cfg.timeoutMs);
    }
  }

  /**
   * 全量重建（doRefresh 后由指纹驱动调用）。
   * fire-and-forget；内部 atomic /rebuild，>BATCH 则退化分块 /upsert。
   * 用 syncInFlight 守卫：并发同步只放一次。
   */
  syncAll(lib: VectorLibrary): Promise<void> {
    if (this.syncInFlight) {
      return this.syncInFlight;
    }
    this.syncInFlight = this.doSyncAll(lib).finally(() => {
      this.syncInFlight = null;
    });
    return this.syncInFlight;
  }

  private async doSyncAll(lib: VectorLibrary): Promise<void> {
    const cfg = await this.getEnabledConfig();
    if (!cfg) return;

    const total = lib.songs.length + lib.playlists.length;
    if (total === 0) {
      // 空库重建：清空服务端
      await this.post('/api/rebuild', { songs: [], playlists: [] }, cfg.timeoutMs);
      return;
    }

    // 预估载荷：每首歌约 80-120 字节（id+title+artist+album）
    const estBytes = total * 120;
    if (estBytes > 2 * 1024 * 1024) {
      // 分批 upsert：先拆歌曲，再拆歌单
      const songEntries: VectorEntry[] = lib.songs.map((s) => ({
        type: 'song',
        ref_id: s.id,
        title: s.title,
        artist: s.artist,
        album: s.album,
      }));
      const playlistEntries: VectorEntry[] = lib.playlists.map((p) => ({
        type: 'playlist',
        ref_id: p.id,
        title: p.name,
      }));
      await this.pushEntries(songEntries);
      await this.pushEntries(playlistEntries);
      return;
    }

    const data = await this.post('/api/rebuild', lib, cfg.timeoutMs);
    if (data === null) {
      // rebuild 失败：退化为分块 upsert 兜底
      songloft.log.warn('[VectorSearcher] /api/rebuild failed, fallback to chunked /api/upsert');
      const songEntries: VectorEntry[] = lib.songs.map((s) => ({
        type: 'song',
        ref_id: s.id,
        title: s.title,
        artist: s.artist,
        album: s.album,
      }));
      const playlistEntries: VectorEntry[] = lib.playlists.map((p) => ({
        type: 'playlist',
        ref_id: p.id,
        title: p.name,
      }));
      await this.pushEntries(songEntries);
      await this.pushEntries(playlistEntries);
    }
  }
}