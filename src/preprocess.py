"""Wisdomain 원본 CSV → 분석용 정제본.

입력:  data/raw/CSV2610154712.csv
출력:  data/interim/patents_clean.parquet   중복 제거된 정제본
       data/interim/dedup_log.csv           제거된 행과 남긴 행의 대응표
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/CSV2610154712.csv"
OUT = ROOT / "data/interim"

# 기업 통합 규칙: 기아 출원 693건 중 692건이 현대차와 공동출원이라 한 기업으로 본다
FIRM_MERGE = {"현대자동차 주식회사": "현대차그룹", "기아 주식회사": "현대차그룹"}

# SBERT 입력용 상투 문구 (내용어는 남기고 고정 표현만 제거)
BOILER = [
    r"본\s*(?:발명|문서|개시)(?:의|에\s*개시된)?\s*(?:일|다양한|몇몇)?\s*(?:실시\s*예|실시\s*형태|실시\s*양태)들?에\s*따른\s*",
    r"본\s*(?:발명|개시)에\s*따른\s*",
    r"^\s*본\s*(?:발명|개시)은\s*,?\s*",
]


def load_raw(path=RAW):
    # 맨 위 4줄은 작성일·검색 DB·검색식·빈 줄
    return pd.read_csv(path, skiprows=4, dtype=str, encoding="utf-8-sig")


def clean_boilerplate(text):
    for p in BOILER:
        text = re.sub(p, "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_firms(owner):
    names = [FIRM_MERGE.get(n.strip(), n.strip()) for n in str(owner).split("|") if n.strip()]
    return "|".join(dict.fromkeys(names))   # 순서 유지 중복 제거


def build(df):
    title = df["명칭(원문)"].str.strip()
    out = pd.DataFrame({
        "pub_no": df["번호"],
        "app_no": df["출원번호"],
        "app_date": pd.to_datetime(df["출원일"], format="%Y.%m.%d"),
        "doc_kind": df["번호"].str.extract(r"([A-Z])\d?$")[0],          # A=특허, U=실용신안
        "title_ko": title.str.replace(r"\{.*\}\s*$", "", regex=True).str.strip(),
        "title_en": title.str.extract(r"\{(.*)\}\s*$")[0],
        "abstract": df["요약(원문)"].str.strip(),
        "applicant": df["출원인대표명"].fillna(df["출원인"]),
        "owner": df["현재소유자"],
        "cpc": df["공통특허분류"],
        "ipc": df["국제특허분류"],
        "family_no": df["패밀리 번호"],
        "right_status": df["권리 현황"],
        "final_status": df["최종 상태"],
        "cited_kr": pd.to_numeric(df["자국피인용횟수"]).astype("Int64"),
        "rating": df["특허평가등급"],
        "rating_score": pd.to_numeric(df["특허평가점수"]),
        "n_claims": pd.to_numeric(df["전체 청구항수"]).astype("Int64"),
    })
    out["year"] = out["app_date"].dt.year
    out["firm"] = out["owner"].map(normalize_firms)
    out["text_embed"] = out["title_ko"] + ". " + out["abstract"].map(clean_boilerplate)
    return out


def dedup(df):
    """요약이 같은 행(공백 차이 무시)은 가장 먼저 출원된 1건만 남긴다."""
    key = df["abstract"].str.replace(r"\s+", " ", regex=True)
    order = df.assign(_key=key).sort_values(["app_date", "app_no"])
    keep = order.drop_duplicates("_key", keep="first")
    kept_id = keep.set_index("_key")["pub_no"]
    dropped = order.loc[~order.index.isin(keep.index)]
    log = pd.DataFrame({"dropped_pub_no": dropped["pub_no"], "kept_pub_no": dropped["_key"].map(kept_id).values})
    return keep.drop(columns="_key").sort_index().reset_index(drop=True), log


def main():
    raw = load_raw()
    clean, log = dedup(build(raw))
    OUT.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(OUT / "patents_clean.parquet", index=False)
    log.to_csv(OUT / "dedup_log.csv", index=False, encoding="utf-8-sig")
    print(f"원본 {len(raw):,}건 → 중복 제거 후 {len(clean):,}건 (제거 {len(log):,}건)")


if __name__ == "__main__":
    main()
