"""4단계 점검 그림: GTM 맵을 cell 격자 위에 칠한다.

GTM은 특허를 거의 한 cell에 확정 배정하므로(최대 R 평균 0.99) 산점도 대신 cell 히트맵으로 그린다.
입력:  data/interim/gtm/grid{G}/patents.parquet, patents_clean.parquet
출력:  outputs/figures/gtm{G}_density_year.png, gtm{G}_cpc.png, gtm{G}_firms.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from check_embeddings import cpc_sets

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
FIG = ROOT / "outputs/figures"

# 참조 팔레트 (dataviz references/palette.md, light)
SURFACE, INK, INK2, MUTED, GRIDLINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq", BLUE)
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f19a99", "#c53030"])

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo", "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRIDLINE, "axes.titlesize": 12, "axes.titleweight": "bold",
})


def to_grid(values, cells, grid, agg):
    s = pd.Series(values).groupby(cells).agg(agg)
    out = np.full(grid * grid, np.nan)
    out[s.index.to_numpy()] = s.to_numpy()
    return out.reshape(grid, grid)          # [row(y), col(x)], origin='lower'로 그림


def panel(ax, M, cmap, title, empty, vmin=None, vmax=None, label=""):
    im = ax.imshow(M, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    er, ec = np.where(empty)
    ax.scatter(ec, er, marker="x", s=14, linewidths=1.2, color=INK)
    ax.set_title(title, loc="left", pad=8)
    ax.set_xticks([]), ax.set_yticks([])
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=9)
    cb.set_label(label, color=INK2, fontsize=9)
    return ax


def main(grid=25):
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(INTERIM / "patents_clean.parquet")
    m = pd.read_parquet(INTERIM / f"gtm/grid{grid}/patents.parquet")
    assert m["pub_no"].tolist() == df["pub_no"].tolist()
    cell = m["cell"].to_numpy()
    counts = np.bincount(cell, minlength=grid * grid).reshape(grid, grid)
    empty = counts == 0
    n_empty = int(empty.sum())

    # 1) 밀도 + 평균 출원연도
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    panel(axes[0], np.where(empty, np.nan, counts), SEQ,
          f"cell별 특허 수 — 빈 cell {n_empty}개 ({n_empty / grid**2:.1%})", empty, label="특허 수")
    year = to_grid(df["year"].to_numpy(), cell, grid, "mean")
    panel(axes[1], year, DIV, "cell별 평균 출원연도 — 파랑=오래됨, 빨강=최근", empty,
          vmin=2013, vmax=2019, label="평균 출원연도")
    fig.suptitle(f"GTM 기술맵 {grid}×{grid} · 10,900건 (2011~2020) · × = 특허 0건 cell", x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / f"gtm{grid}_density_year.png", dpi=160)
    plt.close(fig)

    # 2) 주요 CPC 서브클래스별: cell 안에서 그 CPC를 가진 특허 비율
    sets = [cpc_sets(c, "subclass") for c in df["cpc"]]
    top = pd.Series([s for x in sets for s in x]).value_counts().head(6)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))
    for ax, (sub, n) in zip(axes.ravel(), top.items()):
        has = np.array([sub in x for x in sets], dtype=float)
        panel(ax, to_grid(has, cell, grid, "mean"), SEQ, f"{sub} · {n:,}건", empty, vmin=0, vmax=1,
              label="cell 안 비율")
    fig.suptitle("CPC 서브클래스마다 모이는 영역이 뚜렷하다 (cell 안에서 그 분류를 가진 특허 비율 · × = 특허 0건 cell)",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / f"gtm{grid}_cpc.png", dpi=160)
    plt.close(fig)

    # 3) 주요 기업별 특허 수 (패널마다 색 범위가 다름)
    firms = df["firm"].str.split("|").explode().value_counts().head(4)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    for ax, (firm, n) in zip(axes, firms.items()):
        has = df["firm"].str.split("|").map(lambda xs: firm in xs).to_numpy(dtype=float)
        M = to_grid(has, cell, grid, "sum")
        panel(ax, np.where(M > 0, M, np.nan), SEQ, f"{firm.replace('주식회사', '').strip()} · {n:,}건", empty,
              label="특허 수 (패널별 범위)")
    fig.suptitle("주요 기업마다 특허가 놓인 영역이 다르다 (색 범위는 패널마다 다름 · × = 특허 0건 cell)", x=0.01,
                 ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG / f"gtm{grid}_firms.png", dpi=160)
    plt.close(fig)
    print("저장:", *sorted(p.name for p in FIG.glob(f"gtm{grid}_*.png")))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 25)
