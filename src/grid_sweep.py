"""4단계 보조: 격자 크기별로 빈 cell이 '우연히 빈 칸'일 가능성을 비교한다.

안쪽 cell마다 주변 8칸 평균 특허 수 m을 구하면, 특허가 무작위로 흩어졌을 때 그 cell이 0건일 확률은
e^(-m)이다(포아송 근사). 이를 안쪽 cell 전체에 더한 값이 '우연히 빌 것으로 기대되는 cell 수'다.

입력:  data/interim/emb_bge_m3.npy
출력:  data/interim/gtm/grid_sweep.csv
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from check_embeddings import knn
from gtm import GTM

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
GRIDS = [20, 25, 30, 35, 40, 50]


def neighbor_mean(C):
    grid = C.shape[0]
    P = np.pad(C.astype(float), 1, constant_values=np.nan)
    shifts = [P[1 + di:grid + 1 + di, 1 + dj:grid + 1 + dj]
              for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
    return np.nanmean(np.stack(shifts), 0)


def main():
    E = np.load(INTERIM / "emb_bge_m3.npy").astype(np.float64)
    nn, _ = knn(E.astype(np.float32))
    rows = []
    for grid in GRIDS:
        t0 = time.time()
        g = GTM(grid=grid, rbf_grid=10, rbf_width=1.0, alpha=1e-3, n_iter=300, tol=1e-5).fit(E)
        _, cell, _ = g.transform(E)
        secs = time.time() - t0
        C = np.bincount(cell, minlength=grid * grid).reshape(grid, grid)
        nb = neighbor_mean(C)
        inner = np.zeros_like(C, bool)
        inner[1:-1, 1:-1] = True
        empty = (C == 0) & inner
        r, c = cell // grid, cell % grid
        near = np.maximum(np.abs(r[:, None] - r[nn]), np.abs(c[:, None] - c[nn])) <= 1
        rows.append({
            "grid": grid, "cells": grid * grid, "seconds": round(secs, 1),
            "inner_mean_patents": round(float(C[inner].mean()), 1),
            "inner_empty": int(empty.sum()),
            "expected_chance_empty": round(float(np.exp(-nb[inner]).sum()), 1),
            "empty_with_neighbor_mean_ge5": int((empty & (nb >= 5)).sum()),
            "neighbors_kept_nearby": round(float(near.mean()), 3),
        })
        print(rows[-1], flush=True)
    (INTERIM / "gtm").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(INTERIM / "gtm/grid_sweep.csv", index=False)


if __name__ == "__main__":
    main()
