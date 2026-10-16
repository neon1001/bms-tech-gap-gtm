# bms-tech-gap-gtm

전기차 BMS 특허로 GTM 기술맵을 만들고 기술공백을 찾는 프로젝트의 코드입니다. 진행 중입니다.

## 진행 상태

| 단계 | 스크립트 | 상태 |
| --- | --- | --- |
| 1. 데이터 정제 | `src/preprocess.py` | 완료 |
| 2. 문장 임베딩 (bge-m3) | `src/embed.py` | 완료 |
| 2-1. 임베딩 점검 | `src/check_embeddings.py` | 완료 |
| 3. TF-IDF, 4. GTM 기술맵, 5~8. 공백 탐색·역매핑·해석 | - | 예정 |

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
