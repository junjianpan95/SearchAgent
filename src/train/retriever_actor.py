"""共享检索器 Ray actor：供 RL rollout 的多轮工具调用使用。

索引（BM25 + BGE-M3 向量 + 104K 语料）由 `scripts/build_index.py` 预先构建。
RL 时每个 AgentLoopWorker 都会发起 Search，若各自加载一份索引会重复占用数 GB
内存（本机 WSL 26GB 已被 vLLM + FSDP 占满），因此用命名 actor 做全局限定单例。

与评测 `scripts/run_compare.py` 用同一个 `CachedHybridRetriever`，
保证训练期观测与评测期观测来自同一检索分布。

## 为什么不用 sentence_transformers

1. WSL 的 verl 环境是 `transformers 5.12.1`，而 sentence_transformers 6.1.0
   要求 `transformers<5.0.0`，装进来会破坏已经调好的版本矩阵（该矩阵很脆弱，
   之前修过一轮依赖地狱）；环境又无法联网 pip。
2. 其实没必要：bge-m3 的 `1_Pooling/config.json` 明确写着
       pooling_mode_cls_token: true, pooling_mode_mean_tokens: false
   即 **取 CLS token 再 L2 归一化**，用原生 transformers 十几行就等价实现。

一致性由 `verify_encoder()` 保证：对语料里的已知文本重新编码，
与 `embeddings.npy` 中对应行的余弦相似度应 ≈ 1.0。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import ray

ROOT = Path(__file__).resolve().parent.parent.parent


class BGEM3ClsEncoder:
    """bge-m3 的 CLS pooling + L2 normalize 查询编码器（等价于 SentenceTransformer）。"""

    def __init__(self, model_path: str | None = None, device: str = "cpu", max_length: int = 512):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_length = max_length
        mp = model_path or str(ROOT / "data" / "models" / "bge-m3")
        self.tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(mp, trust_remote_code=True).to(device)
        self.model.eval()

    def encode(self, texts: list[str], normalize_embeddings: bool = True,
               batch_size: int = 8):
        import numpy as np

        torch = self.torch
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i: i + batch_size]
                enc = self.tok(batch, padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state[:, 0]  # CLS
                if normalize_embeddings:
                    h = torch.nn.functional.normalize(h, p=2, dim=1)
                outs.append(h.float().cpu().numpy())
        return np.vstack(outs)


def _make_retriever(device: str):
    """构造检索器，并把 dense 编码器替换成本地 transformers 实现。"""
    from src.retrieval.index_loader import CachedHybridRetriever

    class _LocalRetriever(CachedHybridRetriever):
        def _ensure_model(self):
            if self._model is None:
                self._model = BGEM3ClsEncoder(device=self.device)
            return self._model

    return _LocalRetriever(device=device)


@ray.remote
class RetrieverActor:
    def __init__(self, device: str = "cpu"):
        self.device = device or os.getenv("SA_RETRIEVER_DEVICE", "cpu")
        self._r = _make_retriever(self.device)

    def ping(self) -> bool:
        """探活：确保索引已加载完成。"""
        return True

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []
        hits = self._r.search(query, top_k=top_k)
        out = []
        for h in hits:
            out.append(h.to_dict() if hasattr(h, "to_dict") else
                       {"page": h.page, "text": h.text, "score": float(h.score)})
        return out

    def read(self, page: str) -> str:
        if not page:
            return ""
        try:
            return self._r.read(page)
        except Exception:
            return ""

    def stats(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "passages": len(getattr(self._r, "_passages", []) or []),
        }

    def verify_encoder(self, n: int = 5) -> dict[str, Any]:
        """自检：重编码语料文本，与缓存向量比对余弦相似度。

        若本地 transformers 实现与建索引时的 SentenceTransformer 语义不同，
        相似度会显著低于 1.0，据此可立刻发现问题。
        """
        import numpy as np

        passages = getattr(self._r, "_passages", []) or []
        emb = getattr(self._r, "_emb", None)
        if not passages or emb is None:
            return {"ok": False, "reason": "索引未加载"}

        texts = [p["text"] for p in passages[:n]]
        re_enc = self._r._ensure_model().encode(texts, normalize_embeddings=True)
        sims = [float(np.dot(re_enc[i], emb[i])) for i in range(len(texts))]
        return {
            "ok": min(sims) > 0.999,
            "cosine": sims,
            "mean": float(np.mean(sims)),
            "n_passages": len(passages),
            "emb_shape": list(emb.shape),
        }
