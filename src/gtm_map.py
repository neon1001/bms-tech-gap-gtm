"""4단계: bge-m3 임베딩으로 GTM 기술맵 학습 + 맵 점검.

입력:  data/interim/emb_bge_m3.npy, patents_clean.parquet
출력:  data/interim/gtm/grid{20,25,30}/model.npz         GTM 파라미터 (GTM.load로 다시 불러옴)
                                        patents.parquet   특허별 cell·격자 위치·2D 좌표·최대 R
                                        cell_vectors.npy  cell별 원공간 벡터 y_k (빈 cell 포함, 7단계용)
       data/interim/gtm/summary.json                     격자별 학습·점검 결과
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from check_embeddings import cpc_sets, knn
from gtm import GTM

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
OUT = INTERIM / "gtm"
GRIDS = [20, 25, 30]
RBF_GRID, RBF_WIDTH, ALPHA = 10, 1.0, 1e-3
rng = np.random.default_rng(0)


def check_map(cell, grid, nn_idx, sets):
    """이웃 보존: 원공간 최근접 10개가 맵에서 같은 cell 또는 바로 옆(8칸)에 놓인 비율.
    CPC 모임: 같은 cell 안의 특허 쌍이 CPC 서브클래스를 공유하는 비율. 둘 다 무작위 기준선과 비교."""
    n, k = nn_idx.shape
    r, c = cell // grid, cell % grid
    near = lambda i, j: np.maximum(np.abs(r[i] - r[j]), np.abs(c[i] - c[j])) <= 1
    ii = np.repeat(np.arange(n), k)
    rand = rng.integers(0, n, n * k)
    keep = rand != ii

    pi, pj = [], []
    members = pd.Series(np.arange(n)).groupby(cell).apply(np.array)
    for m in members:
        if len(m) < 2:
            continue
        a = rng.choice(m, min(len(m), 50), replace=False)
        b = np.array([rng.choice(m[m != x]) for x in a])
        pi.extend(a), pj.extend(b)
    ri, rj = rng.integers(0, n, len(pi)), rng.integers(0, n, len(pi))
    share = lambda I, J: float(np.mean([bool(sets[x] & sets[y]) for x, y in zip(I, J) if x != y]))

    counts = np.bincount(cell, minlength=grid * grid)
    return {
        "empty_cell_ratio": round(float((counts == 0).mean()), 3),
        "patents_per_occupied_cell_median": int(np.median(counts[counts > 0])),
        "patents_per_cell_max": int(counts.max()),
        "neighbors_kept_nearby": round(float(near(ii, nn_idx.ravel()).mean()), 3),
        "neighbors_kept_nearby_random": round(float(near(ii[keep], rand[keep]).mean()), 3),
        "same_cell_cpc_share": round(share(pi, pj), 3),
        "same_cell_cpc_share_random": round(share(ri, rj), 3),
    }


def main():
    df = pd.read_parquet(INTERIM / "patents_clean.parquet")
    E = np.load(INTERIM / "emb_bge_m3.npy").astype(np.float64)
    nn_idx, _ = knn(E.astype(np.float32))
    sets = [cpc_sets(c, "subclass") for c in df["cpc"]]
    summary = {}
    for grid in GRIDS:
        t0 = time.time()
        g = GTM(grid=grid, rbf_grid=RBF_GRID, rbf_width=RBF_WIDTH, alpha=ALPHA, n_iter=300, tol=1e-5).fit(E)
        secs = time.time() - t0
        R, cell, z = g.transform(E)
        ll = np.array(g.loglik)

        d = OUT / f"grid{grid}"
        d.mkdir(parents=True, exist_ok=True)
        g.save(d / "model.npz")
        np.save(d / "cell_vectors.npy", g.cell_vectors().astype(np.float32))
        pd.DataFrame({
            "pub_no": df["pub_no"], "cell": cell, "row": cell // grid, "col": cell % grid,
            "z1": z[:, 0], "z2": z[:, 1], "r_max": R.max(0),
        }).to_parquet(d / "patents.parquet", index=False)

        summary[f"{grid}x{grid}"] = {
            "seconds": round(secs, 1), "iterations": len(ll),
            "loglik_monotonic": bool(np.all(np.diff(ll) >= -1e-6 * np.abs(ll[:-1]))),
            "nan": bool(np.isnan(ll).any()), "r_max_mean": round(float(R.max(0).mean()), 3),
            **check_map(cell, grid, nn_idx, sets),
        }
        print(f"{grid}x{grid}:", summary[f"{grid}x{grid}"])
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
