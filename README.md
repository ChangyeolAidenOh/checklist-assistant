# Coverage Checklist Assistant

> 보험 약관 이해 보조 + 상담 시 확인 필요 영역 탐지 + 상담 전 체크리스트 생성 AI Assistant 프로토타입

Python · LangGraph · Anthropic Claude API · XGBoost · ChromaDB · Streamlit

---

## 1. 프로젝트 개요

**출발 질문:** "고객이 상담 전에 자신의 보장 상태를 미리 이해하고, 설계사에게 물어볼 질문을 준비할 수 있다면?"

보험 약관은 수십 페이지의 법률 문서로, 대부분의 고객은 자신의 보장 범위를 정확히 이해하지 못한다. 이 프로젝트는 고객(또는 설계사)이 보험 상품 요약서를 입력하면, AI가 보장 항목을 일상 언어로 변환하고, 생활·건강 관심 영역과 대조하여 상담 시 확인이 필요한 영역 후보를 탐지한 뒤, 상담 전 체크리스트를 자동 생성하는 사전 점검 도구의 프로토타입을 구축한다.

**경계:** 보험 상품 추천이나 가입 권유 도구가 아니다. 약관 이해를 돕고, 설계사 상담 전 확인 항목을 안내하는 사전 점검 보조 도구이다. 최종 판단은 반드시 설계사/전문 상담을 통해 이루어져야 한다.

---

## 2. 산업 맥락 & 포지셔닝

| 기존 플레이어 | 형태 | 한계 |
|---|---|---|
| AIG Korea "참 쉬운 보장 분석 서비스" | 설계사가 수동으로 보장 점검 | 설계사 의존, 확장 불가 |
| Milliman RAG 기반 GPT 시스템 | 내부 설계사용 약관 검색·해석 | 내부 전용, 고객 비접근 |
| 신한라이프 보험 문서 지식베이스 | 약관 2만종 디지털화·구조화 | 지식베이스 구축 단계, 서비스 미연결 |

기존이 "설계사 내부 도구" 또는 "지식베이스 구축"에 머무르는 반면, 이 프로젝트는 고객과 설계사 모두 사용할 수 있는 사전 점검 도구를 구축한다. 상품 추천/권유가 아니라, 약관 이해 보조와 상담 효율화라는 안전한 영역에서 생성형 AI의 가치를 검증하는 PoC이다.

---

## 3. 파이프라인 아키텍처

```
[사용자 입력]
├── 보험 상품 요약서 (MetLife 42종 PDF or 텍스트 붙여넣기)
└── 생활 프로파일 (연령, 직업군, 가족 구성, 건강 관심사)
        │
        ▼
┌─ Node 1: Coverage Parser ────────────────────────────
│  상품 요약서 PDF → 섹션 추출 → 구조화 JSON
│  보장 항목 / 면책 / 한도 / 대기기간
│  → 일상 언어 변환 + 원문 출처 표시
│  Model: Claude Haiku (경량 LLM)
│  Data: MetLife 전 상품 42종 상품요약서
└──────────────────────────────────────────────────────
        │
        ▼
┌─ Node 2: Health & Lifestyle Profiler ────────────────
│  프로파일 기반 관심 영역 벡터 생성
│  "30대 사무직 → 치과/외래·통원/피부과 상위"
│  Data: 건보심평원 상병 진료통계 27.7만 건 (baseline)
│        건보공단 건강검진정보 100만 명 (XGBoost)
│  Model: XGBoost (12-category multi-output)
└──────────────────────────────────────────────────────
        │
        ▼
┌─ Node 3: Gap Detector ──────────────────────────────
│  보장 범위 vs 관심 영역 대조
│  → 상담 시 확인할 영역 후보 + 우선순위
│  "통원 치료: 보장 미포함 — 상담 우선순위 8/10"
│  Model: Rule-based + ML hybrid
└──────────────────────────────────────────────────────
        │
        ▼
┌─ Node 4: Checklist Generator ────────────────────────
│  확인 항목을 상담 전 체크리스트로 변환
│  "상담 시 확인할 항목:"
│  1. 통원 보장이 포함되어 있지 않습니다...
│  2. 치과 보장 한도가 평균 대비 낮을 수 있습니다...
│  Model: Claude Sonnet (고성능 LLM)
└──────────────────────────────────────────────────────
        │
        ▼
┌─ Guardrail ─────────────────────────────────────────
│  "최종 판단은 설계사 상담 필요" 고정 삽입
│  RAG fallback: "자동 분석 범위 밖"
│  권유 표현 후처리 필터링 (regex + blocklist)
└──────────────────────────────────────────────────────

┌─ ChromaDB RAG ──────────────────────────────────────
│  MetLife 42종 상품요약서 → 2,115 chunks 임베딩
│  상품별 보장 상세, 면책사항 검색 지원
└──────────────────────────────────────────────────────
```

### 비용 거버넌스: Node별 모델 티어 분리

| Node | 모델 선택 | 근거 | 비용 |
|---|---|---|---|
| Node 1 (Parser) | Claude Haiku | 구조화 태스크, 비용 1/10 | ~$0.0096/건 |
| Node 2 (프로파일) | XGBoost | LLM 불필요 | $0 |
| Node 3 (탐지) | Rule-based + ML | LLM 불필요 | $0 |
| Node 4 (체크리스트) | Claude Sonnet | 고객 대면 텍스트, 품질 중요 | ~$0.0360/건 |
| **합계** | | | **$0.0456/건** |

전수 Sonnet 대비 **68% 비용 절감** ($0.1440 → $0.0456/건).

---

## 4. 데이터 & 모델

### 4.1 데이터 소스 (100% 한국 공공데이터)

| 데이터 | 소스 | 규모 | 용도 |
|---|---|---|---|
| MetLife 상품 요약서 | metlife.co.kr 보험상품공시 | 42종 PDF | Node 1 파싱 대상 + RAG 임베딩 |
| HIRA 상병별 진료 통계 | data.go.kr (건보심평원) | 277,046 rows | Node 2 baseline lookup |
| NHIS 건강검진정보 | data.go.kr (건보공단) | 1,000,000명 | Node 2 XGBoost 학습 |

### 4.2 XGBoost 모델 성능

- 학습 데이터: NHIS 건강검진정보 2024 (100만 명, 17 features)
- Train/Test split: 80만 / 20만
- 12-category multi-output regression
- MSE: 0.0007 ~ 0.0009 (전 카테고리)

| Feature | 설명 | 중요도 (입원 카테고리) |
|---|---|---|
| age_code | 5세 단위 연령대 | 0.966 |
| has_children | 자녀 유무 | 0.025 |
| bmi | 체질량지수 | - |
| sbp / dbp | 수축기/이완기 혈압 | - |
| fasting_glucose | 공복혈당 | - |
| smoking / drinking | 흡연/음주 상태 | - |

### 4.3 PDF 파싱 성능

MetLife 전 상품 42종 상품요약서 대상:

| 지표 | 결과 |
|---|---|
| 섹션 탐지율 | 155/168 (92%) |
| LLM 파싱 성공률 | 42/42 (100%) |
| confidence medium 이상 | 41/42 (97.6%) |
| confidence high | 5/42 (12%) |

### 4.4 ChromaDB RAG

- 42종 PDF -> 2,115 chunks 임베딩
- 임베딩 모델: all-MiniLM-L6-v2 (ChromaDB 기본)
- 유사도 검색: cosine similarity, threshold 0.65

---

## 5. 개발 여정 & 문제 해결

### 5.1 Phase 1: 스켈레톤 -> MVP (Day 1)

4-Node 파이프라인 스켈레톤을 먼저 구축하고, 각 Node를 순차적으로 테스트했다.

**마주한 문제: Streamlit에서 LLM API 호출 실패**

테스트 스크립트(`python scripts/test_node1_parser.py`)에서는 Haiku 파싱이 정상 동작했으나, Streamlit 대시보드에서는 모든 파싱이 실패 (confidence: low, items: 0). 원인은 `app.py`에서 `.env` 파일을 로드하지 않아 `ANTHROPIC_API_KEY`가 없었던 것.

```python
# 수정: app.py 최상단에 dotenv 로드 추가
from dotenv import load_dotenv
load_dotenv()
```

**마주한 문제: Sonnet 모델 ID 404 에러**

`claude-sonnet-4-20250514`가 deprecation 예정 모델이라 404 반환. `claude-sonnet-4-6`으로 교체하여 해결. 이 경험으로 LLM 백엔드 추상화의 가치를 실감 — 환경변수 하나로 모델 교체가 가능한 구조였기에 수정이 간단했다.

### 5.2 Phase 2: Node 2 고도화 - Synthetic에서 실제 데이터로

**초기 접근: Synthetic 데이터 + Kaggle Prudential**

처음에는 HIRA 통계 패턴을 보고 합성 데이터를 만들어 XGBoost를 학습시켰고, Kaggle Prudential Life Insurance 데이터(59,381건)를 보조 학습 데이터로 계획했다.

**전환 결정: Kaggle Prudential 제거**

Kaggle Prudential은 미국 보험 신청자 데이터(BMI, Ht, Wt 등)라서 한국 보험 시장과 맞지 않았다. 30대 한국 사무직의 의료이용 패턴을 미국 데이터로 학습하면 의미가 없다는 판단 하에 완전히 제거하고, 한국 공공데이터만 사용하기로 전략을 변경했다.

**실제 데이터 확보 및 적용:**

1. **건보심평원 상병별 진료 통계** (data.go.kr): KCD 4단 상병코드 × 성별 × 5세 연령군 → 환자수. 277,046 rows, 12,132개 질병코드를 12개 관심 영역에 매핑하여 baseline lookup table을 교체했다.

2. **건보공단 건강검진정보** (data.go.kr): 100만 명 개인 수준 데이터 (성별, 연령대, 신장, 체중, 혈압, 혈당, 콜레스테롤, 간수치, 흡연, 음주). XGBoost 학습 데이터로 사용하여 Synthetic 데이터를 완전 교체했다.

**결과: Synthetic → Real 전환 효과**

| 항목 | Synthetic | Real |
|---|---|---|
| Baseline | 직접 작성한 가짜 수치 | HIRA 27.7만 건 집계 |
| XGBoost 학습 | 10K synthetic samples | 100만 명 건강검진 데이터 |
| 20대 Top 3 | 치과, 외래/통원, 정신건강 | 피부과, 외래/통원, 치과 |
| 60+ Top 3 | 외래/통원, 건강검진, 안과 | 입원, 재활, 정신건강 |
| 면접 답변 | "합성 데이터입니다" | "한국 공공데이터 100만 명" |

20대에서 피부과가 1위로 올라온 것, 60+에서 입원·재활이 상위로 올라온 것은 실제 한국 의료이용 패턴과 일치한다.

### 5.3 Phase 3: 실제 MetLife PDF 파싱

**마주한 문제: macOS에서 `pdftotext` 미설치**

초기 PDF 추출기가 `pdftotext` (poppler-utils)에 의존했으나, macOS에 기본 설치되어 있지 않았다. `pypdf` 라이브러리로 fallback하는 구조로 수정하여 추가 설치 없이 동작하도록 했다.

**마주한 문제: Haiku JSON 출력 깨짐 (6/6 전부 실패)**

실제 MetLife PDF에서 추출한 한국어 텍스트를 Haiku에 보냈을 때, 반환된 JSON이 전부 "Unterminated string" 에러로 파싱 실패했다. 한국어 텍스트의 특수문자(따옴표, 줄바꿈)가 JSON string을 깨뜨린 것이 원인이었다.

3가지 수정으로 해결:
1. **JSON repair 로직**: 깨진 괄호/따옴표 자동 복구 + partial extraction
2. **입력 텍스트 축소**: 8K → 4K chars (Haiku 출력 길이 제한)
3. **프롬프트 강화**: "문자열 80자 이내, coverage_items 최대 10개" 제한

결과: 0/6 → 6/6 파싱 성공. 이후 42종 전체 테스트에서 100% 성공률 달성.

**마주한 문제: 한국어 파일명 Unicode NFD/NFC 불일치**

macOS는 파일명을 NFD (decomposed) 형태로 저장하는데, Python 문자열 리터럴은 NFC (composed) 형태. `os.path.exists()`가 False를 반환하여 파일을 찾지 못하는 현상이 발생했다. `unicodedata.normalize("NFC", filename)`으로 정규화하여 해결.

### 5.4 Phase 4: SHAP 호환성

**마주한 문제: SHAP + XGBoost 버전 충돌**

`shap.TreeExplainer(model)`에서 "could not convert string to float" 에러가 12개 카테고리 전부에서 발생. 최신 xgboost와 shap 사이의 알려진 호환성 문제였다.

해결: SHAP explainer를 캐싱하는 대신, XGBoost의 `feature_importances_`를 직접 사용하여 예측 시점에 설명을 생성하도록 변경. 이 접근이 실제로 더 안정적이고, 모델 직렬화 문제도 없다.

---

## 6. 대시보드 (Streamlit 5탭)

| 탭 | 콘텐츠 |
|---|---|
| **My Coverage** | 상품 선택 or 요약서 입력 -> 보장 항목 일상 언어 변환 (원문 출처 표시) |
| **My Profile** | 생활 프로파일 입력 -> 관심 영역 레이더 차트 + feature importance 설명 |
| **Check Items** | 보장 vs 관심 영역 대조 -> 확인 필요 항목 + 상담 우선순위 |
| **Consultation Checklist** | 상담 전 질문 체크리스트 + "설계사 상담 필요" 안내 |
| **Pipeline Trace** | 4-Node 입출력, 소요 시간, 모델 티어, 비용 거버넌스 |

---

## 7. 기술 스택

```
Python 3.10 | LangGraph | LangChain | ChromaDB
Anthropic Claude API (Haiku: 파싱 + Sonnet: 체크리스트)
XGBoost | scikit-learn
Streamlit | Plotly | pypdf
```

### LLM 백엔드 추상화

```python
# 환경변수만 바꿔서 프로바이더/모델 교체 가능
LLM_CONFIG = {
    "parser":    {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
    "generator": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
}
```

consumer-signal-agentic-platform 프로젝트에서 구축한 Ollama(로컬) / Gemini(무료) / Claude(프로덕션) 추상화 패턴을 재사용. 만약 Node 1의 Haiku 파싱이 불안정하면, `PARSER_MODEL=gpt-4o-mini`로 환경변수 하나만 바꿔서 교체 가능.

---

## 8. 프로젝트 구조

```
checklist_assistant/
|-- app.py                                    # Streamlit entry point
|-- config.py                                 # Project configuration
|-- requirements.txt
|-- .env.example
|
|-- core_pipeline/
|   |-- llm/
|   |   |-- backend.py                        # LLM backend abstraction (Haiku/Sonnet)
|   |-- parser/
|   |   |-- coverage_parser.py                # Node 1: Product summary parser
|   |   |-- pdf_extractor.py                  # PDF text extraction + section detection
|   |-- profiler/
|   |   |-- health_profiler.py                # Node 2: Health interest profiler
|   |   |-- real_data_processor.py            # HIRA/NHIS real data processing
|   |   |-- nhis_processor.py                 # NHIS synthetic data (deprecated)
|   |   |-- xgb_trainer.py                    # XGBoost training pipeline
|   |-- detector/
|   |   |-- gap_detector.py                   # Node 3: Coverage gap detector
|   |-- generator/
|   |   |-- checklist_generator.py            # Node 4: Checklist generator
|   |-- guardrail/
|   |   |-- filter.py                         # Output safety filter
|   |-- rag/
|   |   |-- __init__.py                       # ChromaDB RAG module
|   |-- pipeline.py                           # LangGraph StateGraph orchestrator
|
|-- dashboard/
|   |-- tabs/
|       |-- tab1_my_coverage.py
|       |-- tab2_my_profile.py
|       |-- tab3_check_items.py
|       |-- tab4_checklist.py
|       |-- tab5_trace.py
|
|-- data/
|   |-- product_summaries/                    # MetLife 42종 상품요약서 PDF
|   |-- nhis_stats/
|       |-- hira_disease_stats.csv            # HIRA 상병별 진료 통계 (277K rows)
|       |-- nhis_health_checkup.csv           # NHIS 건강검진정보 (1M rows)
|       |-- real_lookup_table.csv             # 가공된 baseline lookup
|
|-- models/
|   |-- xgb_profiler.pkl                      # Trained XGBoost model
|
|-- scripts/
|   |-- train_with_real_data.py               # Real data training
|   |-- setup_rag.py                          # ChromaDB ingestion
|   |-- test_node1_parser.py                  # Node 1 test
|   |-- test_node2_profiler.py                # Node 2 test
|   |-- test_node3_detector.py                # Node 3 test
|   |-- test_node4_generator.py               # Node 4 test
|   |-- test_pdf_parsing.py                   # PDF parsing batch test
|
|-- tests/
|-- figures/
|-- docs/
```

---

## 9. Setup

```bash
# Clone
git clone https://github.com/ChangyeolAidenOh/coverage-checklist-assistant.git
cd coverage-checklist-assistant

# Virtual environment
python -m venv .venv
source .venv/bin/activate

# Install
pip install -r requirements.txt

# Environment
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY

# Data (place downloaded files)
# data/nhis_stats/hira_disease_stats.csv   (from data.go.kr/data/15138971)
# data/nhis_stats/nhis_health_checkup.csv  (from data.go.kr/data/15007122)
# data/product_summaries/*.pdf             (from metlife.co.kr 보험상품공시)

# Train XGBoost with real data
python scripts/train_with_real_data.py

# Ingest PDFs into ChromaDB
python scripts/setup_rag.py --force

# Run dashboard
streamlit run app.py
```

---

## 10. 차별화 포인트

1. **업계 맥락을 이해한 포지셔닝:** AIG/Milliman/신한라이프 분석 위에서 "상담 전 사전 점검 도구"라는 안전한 포지션
2. **100% 한국 공공데이터:** 건보심평원 27.7만 건 + 건보공단 100만 명 - 해외 데이터(Kaggle) 의존 없음
3. **실제 MetLife 전 상품 검증:** 42종 상품요약서 파싱 성공률 100% (medium 이상 97.6%)
4. **비용 거버넌스:** Node별 모델 티어 분리 - 전수 Sonnet 대비 68% 절감
5. **한계를 아는 프로토타입:** 상품 요약서로 MVP, 약관 전문은 next step

---