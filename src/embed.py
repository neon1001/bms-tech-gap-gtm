"""정제본 text_embed → bge-m3 문장 벡터.

입력:  data/interim/patents_clean.parquet
출력:  data/interim/emb_bge_m3.npy           (N, 1024) float32, L2 정규화, 정제본과 같은 행 순서
       data/interim/emb_bge_m3_pub_no.txt    각 행의 공개번호 (행 순서 확인용)
       data/interim/emb_bge_m3_meta.json     모델·리비전·설정·소요 시간
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
MODEL = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"   # 재현용 고정
MAX_LEN = 8192
BATCH = 32


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load(device):
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    assert tok.padding_side == "right"   # CLS가 0번 위치에 있어야 함
    model = AutoModel.from_pretrained(MODEL, revision=REVISION).eval().to(device)
    return tok, model


@torch.no_grad()
def embed(texts, tok, model, device, batch=BATCH):
    """bge-m3 공식 설정(1_Pooling/config.json, modules.json): CLS 토큰 pooling → L2 정규화."""
    lengths = np.array([len(ids) for ids in tok(texts)["input_ids"]])
    order = np.argsort(lengths)          # 비슷한 길이끼리 묶어 패딩 낭비를 줄임
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for i in range(0, len(texts), batch):
        idx = order[i:i + batch]
        b = tok([texts[j] for j in idx], padding=True, truncation=True,
                max_length=MAX_LEN, return_tensors="pt").to(device)
        cls = model(**b).last_hidden_state[:, 0]
        out[idx] = torch.nn.functional.normalize(cls, dim=-1).float().cpu().numpy()
    return out, lengths


def main():
    df = pd.read_parquet(INTERIM / "patents_clean.parquet")
    texts = df["text_embed"].tolist()
    device = pick_device()
    tok, model = load(device)
    t0 = time.time()
    emb, lengths = embed(texts, tok, model, device)
    secs = time.time() - t0

    np.save(INTERIM / "emb_bge_m3.npy", emb)
    (INTERIM / "emb_bge_m3_pub_no.txt").write_text("\n".join(df["pub_no"]) + "\n", encoding="utf-8")
    meta = {
        "model": MODEL, "revision": REVISION, "pooling": "cls", "normalized": True,
        "device": device, "batch": BATCH, "max_length": MAX_LEN,
        "n": len(texts), "dim": int(emb.shape[1]),
        "tokens_median": int(np.median(lengths)), "tokens_max": int(lengths.max()),
        "n_truncated": int((lengths > MAX_LEN).sum()),
        "seconds": round(secs, 1), "created": datetime.now().isoformat(timespec="seconds"),
        "source": "data/interim/patents_clean.parquet (text_embed)",
    }
    (INTERIM / "emb_bge_m3_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(texts):,}건 임베딩 완료: {emb.shape}, {device}, {secs / 60:.1f}분")


if __name__ == "__main__":
    main()
