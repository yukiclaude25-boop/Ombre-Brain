请把项目里的 embedding_engine.py 整个文件替换成下面这个新版本（用于把 Gemini 向量调用从 OpenAI 兼容接口改成 Google 原生接口，用 ?key= 认证，兼容 AIza 和 AQ. 两种 key）。

要求：
1. 用下面的内容【完整覆盖】 embedding_engine.py，保持 UTF-8 编码、LF 换行。
2. 不要删掉 find_similar_buckets 方法（server.py 依赖它）。
3. 替换后运行 `python -m py_compile embedding_engine.py` 确认能编译通过。
4. 不要改动其他任何文件。

新版 embedding_engine.py 完整内容如下：

# ============================================================
# Module: Embedding Engine (embedding_engine.py)
# 模块：向量化引擎
#
# Generates embeddings via Gemini API (Google native by default,
# OpenAI-compatible for custom endpoints), stores them in SQLite,
# and provides cosine similarity search.
# 通过 Gemini API（默认 Google 原生接口，自定义端点走 OpenAI 兼容）
# 生成 embedding，存储在 SQLite 中，提供余弦相似度搜索。
#
# Depended on by: server.py, bucket_manager.py
# 被谁依赖：server.py, bucket_manager.py
# ============================================================

import os
import json
import math
import sqlite3
import logging

import httpx

# Optional: keep openai import for users who explicitly set a non-Google base_url
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

logger = logging.getLogger("ombre_brain.embedding")

# Google's native embedding endpoint (default)
_GOOGLE_NATIVE_BASE = "https://generativelanguage.googleapis.com/v1beta"


class EmbeddingEngine:
    """
    Embedding generation + SQLite vector storage + cosine search.
    Supports both Google native API (default) and OpenAI-compatible endpoints.
    向量生成 + SQLite 向量存储 + 余弦搜索。
    """

    def __init__(self, config: dict):
        dehy_cfg = config.get("dehydration", {})
        embed_cfg = config.get("embedding", {})

        if embed_cfg.get("independent"):
            self.api_key = str(embed_cfg.get("api_key") or "").strip()
        else:
            self.api_key = (
                embed_cfg.get("api_key") or dehy_cfg.get("api_key") or ""
            ).strip()

        # Determine base_url and mode
        user_base_url = (
            (embed_cfg.get("base_url") or "").strip()
            or (dehy_cfg.get("base_url") or "").strip()
        )

        # If user set a custom base_url that is NOT Google's, use OpenAI-compatible mode
        if user_base_url and "generativelanguage.googleapis.com" not in user_base_url:
            self.mode = "openai_compat"
            self.base_url = user_base_url
        else:
            self.mode = "google_native"
            self.base_url = _GOOGLE_NATIVE_BASE

        self.model = embed_cfg.get("model", "gemini-embedding-001")
        self.enabled = bool(self.api_key) and embed_cfg.get("enabled", True)
        self.last_error = ""
        self.last_error_details = {}

        # --- SQLite path: buckets_dir/embeddings.db ---
        db_path = os.path.join(config["buckets_dir"], "embeddings.db")
        self.db_path = db_path

        # --- Initialize client (only for OpenAI-compat mode) ---
        self.client = None
        if self.enabled and self.mode == "openai_compat":
            if AsyncOpenAI is not None:
                self.client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=30.0,
                )
            else:
                logger.warning("openai package not installed; falling back to google_native mode")
                self.mode = "google_native"

        # --- Initialize SQLite ---
        self._init_db()

    def _init_db(self):
        """Create embeddings table if not exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                bucket_id TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                model TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            )
        """)
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(embeddings)").fetchall()
        }
        if "model" not in columns:
            conn.execute(
                "ALTER TABLE embeddings ADD COLUMN model TEXT NOT NULL DEFAULT ''"
            )
        conn.commit()
        conn.close()

    async def generate_and_store(self, bucket_id: str, content: str) -> bool:
        """
        Generate embedding for content and store in SQLite.
        为内容生成 embedding 并存入 SQLite。
        Returns True on success, False on failure.
        """
        if not self.enabled or not content or not content.strip():
            return False

        try:
            embedding = await self._generate_embedding(content)
            if not embedding:
                return False
            self._store_embedding(bucket_id, embedding)
            self.last_error = ""
            self.last_error_details = {}
            return True
        except Exception as e:
            self._capture_error(e)
            logger.warning(f"Embedding generation failed for {bucket_id}: {e}")
            return False

    async def _generate_embedding(self, text: str) -> list[float]:
        """Call API to generate embedding vector. Supports both Google native and OpenAI-compat."""
        # Truncate to avoid token limits
        truncated = text[:2000]

        if self.mode == "google_native":
            return await self._generate_embedding_google_native(truncated)
        else:
            return await self._generate_embedding_openai_compat(truncated)

    async def _generate_embedding_google_native(self, text: str) -> list[float]:
        """Call Google's native embedding API with ?key= auth (works with all key formats)."""
        url = f"{self.base_url}/models/{self.model}:embedContent?key={self.api_key}"
        body = {
            "model": f"models/{self.model}",
            "content": {
                "parts": [{"text": text}]
            }
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=body)
                if response.status_code != 200:
                    self.last_error = f"Google API error: {response.status_code}"
                    self.last_error_details = {
                        "request_url": str(url).split("?")[0],  # Don't log key
                        "status_code": response.status_code,
                        "response_body": response.text[:2000],
                    }
                    logger.warning(f"Embedding API call failed: Error code: {response.status_code} - {response.text[:200]}")
                    return []
                data = response.json()
                values = data.get("embedding", {}).get("values", [])
                return values
        except Exception as e:
            self._capture_error(e)
            logger.warning(f"Embedding API call failed: {e}")
            return []

    async def _generate_embedding_openai_compat(self, text: str) -> list[float]:
        """Call OpenAI-compatible embedding API (for non-Google providers)."""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            if response.data and len(response.data) > 0:
                return response.data[0].embedding
            return []
        except Exception as e:
            self._capture_error(e)
            logger.warning(f"Embedding API call failed: {e}")
            return []

    def _capture_error(self, error: Exception) -> None:
        """Keep upstream diagnostics without retaining credentials or headers."""
        response = getattr(error, "response", None)
        request = getattr(error, "request", None)
        if request is None and response is not None:
            request = getattr(response, "request", None)

        self.last_error = f"{type(error).__name__}: {error}"[:500]
        self.last_error_details = {
            "request_url": str(getattr(request, "url", "")),
            "status_code": getattr(response, "status_code", None),
            "response_body": getattr(response, "text", "")[:2000],
        }

    def _store_embedding(self, bucket_id: str, embedding: list[float]):
        """Store embedding in SQLite."""
        from utils import now_iso
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT OR REPLACE INTO embeddings
                (bucket_id, embedding, model, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (bucket_id, json.dumps(embedding), self.model, now_iso()),
        )
        conn.commit()
        conn.close()

    def delete_embedding(self, bucket_id: str):
        """Remove embedding when bucket is deleted."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM embeddings WHERE bucket_id = ?", (bucket_id,))
        conn.commit()
        conn.close()

    async def get_embedding(self, bucket_id: str) -> list[float] | None:
        """Retrieve stored embedding for a bucket. Returns None if not found."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            """
            SELECT embedding FROM embeddings
            WHERE bucket_id = ? AND model = ?
            """,
            (bucket_id, self.model),
        ).fetchone()
        conn.close()
        if row:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                return None
        return None

    async def search_similar(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Search for buckets similar to query text.
        Returns list of (bucket_id, similarity_score) sorted by score desc.
        搜索与查询文本相似的桶。返回 (bucket_id, 相似度分数) 列表。
        """
        if not self.enabled:
            return []

        try:
            query_embedding = await self._generate_embedding(query)
            if not query_embedding:
                return []
        except Exception as e:
            logger.warning(f"Query embedding failed: {e}")
            return []

        # Load all embeddings from SQLite
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT bucket_id, embedding FROM embeddings WHERE model = ?",
            (self.model,),
        ).fetchall()
        conn.close()

        if not rows:
            return []

        # Calculate cosine similarity
        results = []
        for bucket_id, emb_json in rows:
            try:
                stored_embedding = json.loads(emb_json)
                sim = self._cosine_similarity(query_embedding, stored_embedding)
                results.append((bucket_id, sim))
            except (json.JSONDecodeError, Exception):
                continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def find_similar_buckets(
        self, bucket_id: str, top_k: int = 3, min_sim: float = 0.5
    ) -> list[tuple[str, float]]:
        """
        Find buckets most similar to an existing bucket's stored embedding.
        基于已存储的 embedding，找出与某个桶最相似的若干桶（排除自身）。
        Returns (bucket_id, similarity) sorted desc; empty if unavailable.
        """
        if not self.enabled:
            return []

        target = await self.get_embedding(bucket_id)
        if not target:
            return []

        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT bucket_id, embedding FROM embeddings WHERE model = ?",
            (self.model,),
        ).fetchall()
        conn.close()

        results = []
        for other_id, emb_json in rows:
            if other_id == bucket_id:
                continue
            try:
                other = json.loads(emb_json)
                sim = self._cosine_similarity(target, other)
                if sim >= min_sim:
                    results.append((other_id, sim))
            except (json.JSONDecodeError, Exception):
                continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

替换并编译通过后，请提交并推送这个改动。
