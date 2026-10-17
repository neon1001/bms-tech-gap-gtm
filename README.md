# bms-tech-gap-gtm

전기차 BMS 특허로 GTM 기술맵을 만들고 기술공백을 찾는 프로젝트의 코드입니다. 진행 중입니다.

## 진행 상태

| 단계 | 스크립트 | 상태 |
| --- | --- | --- |
| 1. 데이터 정제 | `src/preprocess.py` | 완료 |
| 2. 문장 임베딩 (bge-m3) | `src/embed.py` | 완료 |
| 2-1. 임베딩 점검 | `src/check_embeddings.py` | 완료 |
| 3. TF-IDF (공백 해석용) | `src/tfidf.py` | 완료 |
| 4. GTM 기술맵 (20·25·30 격자) | `src/gtm.py`, `src/gtm_map.py`, `src/plot_gtm_map.py` | 완료 |
| 4-1. 격자 크기 비교 | `src/grid_sweep.py` | 완료 (기본 격자 30×30으로 결정) |
| 5~8. 공백 탐색·가짜 공백 제거·역매핑·해석 | - | 예정 |

## 데이터

데이터는 이 저장소에 없습니다. 유료 특허 DB(Wisdomain)에서 받은 데이터라 공개하지 않습니다.

직접 받을 때의 형식:

- Wisdomain 다운로드 → "선택 항목 다운로드 (도면제외)" → CSV, 인코딩 UTF-8
- "고정포맷 다운로드" + Simple패밀리 체크
- 받은 파일을 `data/raw/`에 두고, `src/preprocess.py`의 `RAW` 경로를 파일 이름에 맞게 고칩니다

## 실행

```bash
pip install -r requirements.txt
python3 src/preprocess.py         # 정제 → data/interim/patents_clean.parquet
python3 src/embed.py              # 임베딩 → data/interim/emb_bge_m3.npy
python3 src/check_embeddings.py   # 점검 → data/interim/emb_check.json
python3 src/tfidf.py              # TF-IDF → data/interim/tfidf_matrix.npz
python3 src/gtm_map.py            # GTM 학습·점검 → data/interim/gtm/
python3 src/plot_gtm_map.py 30    # 점검 그림 → outputs/figures/ (숫자 = 격자 크기, 기본 30)
python3 src/grid_sweep.py         # 격자 크기 비교 → data/interim/gtm/grid_sweep.csv
```

- `embed.py`는 처음 실행할 때 bge-m3 가중치(2.27GB)를 내려받습니다.
- 모델을 한 번 받은 뒤에는 `HF_HUB_OFFLINE=1 python3 src/embed.py`로 실행하면 transformers가 다른 형식의 가중치를 추가로 내려받지 않습니다.
- Apple M3 Max의 내장 GPU(MPS)에서 10,900건 임베딩에 4.2분 걸렸습니다. GPU가 없으면 CPU로 돌아가며 더 오래 걸립니다.

## 처리 내용

**정제 (`preprocess.py`)**

- 요약이 같은 행(공백 차이 무시)은 가장 먼저 출원된 1건만 남깁니다. 지운 행은 `dedup_log.csv`에 기록합니다.
- 기업은 '현재소유자' 칼럼 기준이고, 현대자동차와 기아는 `현대차그룹`으로 묶습니다.
- 임베딩 입력은 `한국어 제목. 요약`이며, "본 발명의 일 실시예에 따른" 같은 고정 문구만 지웁니다.

**임베딩 (`embed.py`)**

- 모델은 [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)이고, 리비전을 `5617a9f`로 고정했습니다.
- 모델 공식 설정대로 CLS 토큰을 pooling한 뒤 L2 정규화합니다. 최대 길이는 8,192토큰입니다.

**점검 (`check_embeddings.py`)**

- 무결성: shape, NaN, 벡터 길이, 행 순서
- 의미가 가까운 이웃 10개끼리 CPC가 겹치는 비율을 무작위 쌍, 글자 단위 TF-IDF와 비교합니다.

**TF-IDF (`tfidf.py`)**

- GTM 학습에는 쓰지 않고, 학습이 끝난 뒤 공백 cell의 키워드를 읽는 데 씁니다.
- Kiwi로 일반명사·고유명사·영문만 남기고, 특허 상투어를 불용어로 뺍니다. 숫자는 "제1"·"제2" 같은 서수에서 나온 잡음이 대부분이라 남기지 않습니다.
- 1~2-gram, 5건 미만·70% 초과 단어 제외, 최대 30,000개, sublinear TF, L2 정규화를 씁니다.
- 행은 정제본과 같은 순서라서 임베딩·GTM 결과와 행끼리 바로 대응합니다.

**GTM 기술맵 (`gtm.py`, `gtm_map.py`)**

- GTM(Bishop, Svensén & Williams, 1998)을 numpy로 구현했습니다. 학습에는 bge-m3 벡터만 씁니다.
- 설정: RBF 10×10, 폭 = 중심 간격의 1.0배, 정규화 α = 0.001, PCA 초기화(무작위성 없음)
- 저장: 격자별 모델 파라미터, 특허별 cell·2D 좌표, cell별 원공간 벡터 y_k(빈 cell 포함)
- 점검: 빈 cell 비율, 원공간 최근접 이웃이 맵에서 같은·옆 cell에 놓인 비율, 같은 cell 특허끼리 CPC가 겹치는 비율

**격자 크기 비교 (`grid_sweep.py`)**

- 안쪽 cell마다 주변 8칸 평균 m으로 우연히 0건일 확률 e^(-m)을 구해(포아송 근사) 더합니다. 실제 빈 cell 수와 비교해 격자가 너무 촘촘한지 봅니다.
- 비교 결과 **기본 격자를 30×30**으로 정했습니다. 주변이 붐비는(평균 5건 이상) 빈 cell이 36개로 가장 많고, 우연히 빌 것으로 기대되는 cell은 3.3개입니다. 더 촘촘하면(35×35 이상) 우연한 빈 칸이 급격히 늘어납니다.
