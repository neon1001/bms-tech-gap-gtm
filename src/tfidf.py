"""정제본 → TF-IDF 행렬 (공백 해석용).

텍스트 정제·Kiwi 형태소 분석·불용어·벡터라이저 설정은 팀원 코드(team.py)의 TF-IDF 부분을 그대로 옮겼다.
바꾼 점은 하나다: 숫자(SN) 품사는 남기지 않는다. 숫자 토큰은 대부분 '제1'·'제2' 같은 서수에서 나온 잡음이었다.
행은 정제본과 같은 순서라서 임베딩·GTM 결과와 행끼리 바로 대응한다.

입력:  data/interim/patents_clean.parquet
출력:  data/interim/tfidf_matrix.npz        (N, V) 희소 행렬, L2 정규화
       data/interim/tfidf_vectorizer.pkl
       data/interim/tfidf_vocabulary.csv    키워드, IDF, 등장 특허 수
       data/interim/tfidf_tokens.parquet    공개번호별 토큰 (8단계 공출현 분석용)
"""
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from kiwipiepy import Kiwi
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"

# 너무 일반적이라 기술 의미를 구분하기 어려운 표현 (team.py 그대로)
STOPWORDS = {
    # 특허 문서 공통 표현
    "발명", "본", "실시예", "실시", "예", "상기", "따른", "통해", "이용", "사용", "제공",
    "포함", "구성", "각각", "복수", "소정", "해당", "관련", "가능", "경우", "위해", "따라",
    # 너무 일반적인 기술 문서 표현
    "장치", "방법", "시스템", "부", "부재", "단계", "정보", "값", "기능",
}
KEEP_TAGS = ("NNG", "NNP", "SL")   # 일반명사, 고유명사, 영어


def basic_clean(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^가-힣a-z0-9+\-./% ]", " ", text)   # 한글·영어·숫자·+ - . / % 만 유지
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text, kiwi):
    tokens = []
    for t in kiwi.tokenize(text):
        word = t.form.lower().strip()
        if not t.tag.startswith(KEEP_TAGS):
            continue
        # 한국어 1글자는 버리고, BMS·EV 같은 영문 약어는 남긴다
        if len(word) < 2 and not re.search(r"[a-z0-9]", word):
            continue
        if word in STOPWORDS:
            continue
        tokens.append(word)
    return tokens


def build_vectorizer():
    return TfidfVectorizer(
        tokenizer=str.split, token_pattern=None, lowercase=False,   # Kiwi로 이미 토큰화함
        ngram_range=(1, 2), min_df=5, max_df=0.70, max_features=30000,
        sublinear_tf=True, norm="l2",
    )


def main():
    df = pd.read_parquet(INTERIM / "patents_clean.parquet")
    kiwi = Kiwi()
    text = (df["title_ko"] + " " + df["abstract"]).map(basic_clean)
    docs = [" ".join(tokenize(t, kiwi)) for t in text]

    vec = build_vectorizer()
    X = vec.fit_transform(docs)

    emb_ids = INTERIM / "emb_bge_m3_pub_no.txt"
    if emb_ids.exists():   # 임베딩과 행 순서가 같아야 7단계 역매핑 계산이 맞는다
        assert emb_ids.read_text(encoding="utf-8").split() == df["pub_no"].tolist()

    save_npz(INTERIM / "tfidf_matrix.npz", X)
    joblib.dump(vec, INTERIM / "tfidf_vectorizer.pkl")
    vocab = pd.DataFrame({
        "keyword": vec.get_feature_names_out(),
        "idf": vec.idf_,
        "n_docs": np.asarray((X > 0).sum(axis=0)).ravel(),
    }).sort_values("idf").reset_index(drop=True)
    vocab.to_csv(INTERIM / "tfidf_vocabulary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"pub_no": df["pub_no"], "tfidf_text": docs}).to_parquet(INTERIM / "tfidf_tokens.parquet", index=False)
    print(f"TF-IDF 행렬 {X.shape} (특허 {X.shape[0]:,}건 × 키워드 {X.shape[1]:,}개)")


if __name__ == "__main__":
    main()
