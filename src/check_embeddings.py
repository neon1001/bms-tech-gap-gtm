"""bge-m3 임베딩 점검: 무결성 + 기술 구조를 잡았는지(이웃 특허 CPC 일치도).

입력:  data/interim/patents_clean.parquet, emb_bge_m3.npy, emb_bge_m3_pub_no.txt
출력:  data/interim/emb_check.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
K = 10
rng = np.random.default_rng(0)


def cpc_sets(cpc, level):
    """'H01M10/48, G01R31/392' → main group {'H01M10','G01R31'} 또는 subclass {'H01M','G01R'}.
    Y 섹션(Y02E 등)은 분야 태그라 제외한다."""
    out = set()
    for c in str(cpc).split(","):
        c = c.strip()
        if c and not c.startswith("Y"):
            out.add(c.split("/")[0] if level == "group" else c[:4])
    return out


def knn(X, k=K, chunk=1024):
    """코사인 최근접 k개 (X는 행이 정규화된 dense 또는 sparse 행렬)."""
    n = X.shape[0]
    idx, sim = np.zeros((n, k), dtype=int), np.zeros((n, k), dtype=np.float32)
    for s in range(0, n, chunk):
        S = X[s:s + chunk] @ X.T
        S = S.toarray() if hasattr(S, "toarray") else np.asarray(S)
        rows = np.arange(S.shape[0])
        S[rows, s + rows] = -np.inf                       # 자기 자신 제외
        part = np.argpartition(-S, k, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, 1)
        o = np.argsort(-ps, 1)
        idx[s:s + chunk], sim[s:s + chunk] = np.take_along_axis(part, o, 1), np.take_along_axis(ps, o, 1)
    return idx, sim


def share_rate(sets, pairs_i, pairs_j):
    return float(np.mean([bool(sets[i] & sets[j]) for i, j in zip(pairs_i, pairs_j)]))


def cpc_agreement(idx, sets):
    n, k = idx.shape
    return share_rate(sets, np.repeat(np.arange(n), k), idx.ravel())


def main():
    df = pd.read_parquet(INTERIM / "patents_clean.parquet")
    emb = np.load(INTERIM / "emb_bge_m3.npy")
    ids = (INTERIM / "emb_bge_m3_pub_no.txt").read_text(encoding="utf-8").split()
    n = len(df)
    norms = np.linalg.norm(emb, axis=1)
    rep = {"integrity": {
        "shape": list(emb.shape), "nan": bool(np.isnan(emb).any()),
        "norm_min": round(float(norms.min()), 5), "norm_max": round(float(norms.max()), 5),
        "row_order_matches": ids == df["pub_no"].tolist()}}

    idx, sim = knn(emb)
    ri, rj = rng.integers(0, n, 200_000), rng.integers(0, n, 200_000)
    keep = ri != rj
    ri, rj = ri[keep], rj[keep]
    rand_cos = (emb[ri] * emb[rj]).sum(1)
    nn1 = sim[:, 0]
    pairs = {tuple(sorted((i, int(j)))) for i, j in zip(np.arange(n), idx[:, 0]) if sim[i, 0] >= 0.98}
    rep["similarity"] = {
        "random_pair_cos_mean": round(float(rand_cos.mean()), 3),
        "nn1_cos_p10_p50_p90": [round(float(x), 3) for x in np.quantile(nn1, [.1, .5, .9])],
        "near_duplicate_pairs_cos_ge_0.98": len(pairs)}

    # 비교 기준: 글자 2~3-gram TF-IDF (형태소 분석 없는 단순 어휘 기반 방법)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=5, sublinear_tf=True)
    Xc = char.fit_transform(df["text_embed"])
    idx_c, _ = knn(Xc)

    rep["cpc_agreement"] = {}
    for level in ["subclass", "group"]:
        sets = [cpc_sets(c, level) for c in df["cpc"]]
        base = share_rate(sets, ri, rj)
        bge = cpc_agreement(idx, sets)
        chr_ = cpc_agreement(idx_c, sets)
        rep["cpc_agreement"][level] = {
            "random_pairs": round(base, 3), "char_tfidf_knn10": round(chr_, 3), "bge_m3_knn10": round(bge, 3),
            "bge_lift_vs_random": round(bge / base, 2)}

    samples = []
    for i in rng.choice(n, 3, replace=False):
        samples.append({"query": df["title_ko"][i],
                        "neighbors": [[df["title_ko"][j], round(float(s), 3)] for j, s in zip(idx[i, :3], sim[i, :3])]})
    rep["samples"] = samples

    (INTERIM / "emb_check.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
