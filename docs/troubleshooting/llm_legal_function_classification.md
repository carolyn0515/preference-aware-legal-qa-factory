# LLM Legal-Function Classification Troubleshooting

## 1. 문서 목적

이 문서는 한국어 법령 proposition을 12개 법적 기능으로 분류하는
파이프라인을 구현하면서 발생한 장애와 품질 문제를 기록한다.

대상 기능은 다음과 같다.

- `DEFINITION`
- `SCOPE`
- `OBLIGATION`
- `PROHIBITION`
- `PERMISSION`
- `CONDITION`
- `EXCEPTION`
- `PROCESS`
- `AUTHORITY`
- `SANCTION`
- `REFERENCE`
- `DELEGATION`
- `UNCLASSIFIED`: 위 12개 기능 중 어느 것에도 해당하지 않는 배타적 fallback

이 기록의 목적은 단순히 에러 메시지를 보존하는 것이 아니다. 장애가 발생한
데이터 단계, 원인 분석 과정, 잘못된 해결 방향, 최종 설계 결정, 재발 방지
통제를 함께 남겨 동일한 문제가 재발했을 때 복구 시간을 줄이는 데 있다.

## 2. 영향 범위

### 입력

- 하도급거래 공정화에 관한 법률 proposition
- 하도급거래 공정화에 관한 법률 시행령 proposition
- Silver 구조 노드와 Bronze PDF lineage

### 출력

- `legal_functions.parquet`
- `legal_function_manifest.yaml`
- SQLite LLM request cache

### 관련 코드

- `scripts/classify_legal_functions.py`
- `src/legal_qa_factory/llm/openai_client.py`
- `src/legal_qa_factory/llm/gemini_client.py`
- `src/legal_qa_factory/llm/cache.py`
- `src/legal_qa_factory/silver/semantics/legal_functions.py`
- `configs/models/legal_function.yaml`
- `configs/prompts/legal_function_classification.yaml`

## 3. 정상 처리 흐름

```text
Silver proposition
    ↓
Article·부모·조상 문맥 조립
    ↓
provider별 Structured Output 요청
    ↓
응답 schema 및 ID 순서 검증
    ↓
evidence phrase를 원문 span에 정렬
    ↓
intrinsic label 확정
    ↓
계층 기반 deterministic inheritance
    ↓
effective label 생성
    ↓
모든 문서가 성공한 경우에만 Parquet 발행
```

## 4. 장애 및 해결 타임라인

### 4.1 OpenAI API quota 부족

#### 증상

```text
openai.RateLimitError
code: insufficient_quota
```

#### 원인

API key 형식이나 요청 코드 문제가 아니라 해당 OpenAI API project에 사용할 수
있는 credit 또는 billing quota가 없었다. ChatGPT 구독과 OpenAI API billing은
별도이므로 ChatGPT를 사용 중이라는 사실만으로 API credit이 제공되지는 않는다.

#### 판단

- API 요청은 모델 생성 전에 거절됐다.
- 성공한 분류 결과는 생성되지 않았다.
- cache에도 성공 결과가 저장되지 않았다.
- 무료 실험을 위해 Gemini provider를 추가하기로 결정했다.

#### 해결

provider-neutral adapter를 도입했다.

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.5-flash
```

OpenAI adapter는 삭제하지 않고 비교 실험과 향후 유료 실행을 위해 유지했다.

### 4.2 OpenAI Structured Outputs schema 거절

#### 증상

```text
Invalid schema for response_format 'legal_function_batch'
'uniqueItems' is not permitted
```

#### 원인

Structured Outputs가 지원하는 JSON Schema 부분집합에는 `uniqueItems`가 포함되지
않았다. 일반 JSON Schema에서 유효하다는 사실과 특정 provider가 Structured
Outputs에서 지원한다는 사실을 동일하게 취급한 것이 원인이었다.

#### 잘못된 해결 방향

- Structured Outputs 전체를 끄고 자유 형식 JSON을 받는 방법
- label 중복을 허용하고 그대로 저장하는 방법

두 방법 모두 데이터 contract와 재현성을 약화하므로 채택하지 않았다.

#### 해결

- provider schema에서 `uniqueItems`를 제거했다.
- 응답 후 로컬 validator에서 label 중복을 차단했다.

```python
if len(labels) != len(set(labels)):
    raise ValueError("duplicate labels")
```

#### 검증

- API가 schema를 수락했다.
- duplicate label 단위 테스트를 추가했다.

### 4.3 Gemini evidence phrase 원문 불일치

#### 증상

첫 번째 문서는 발행됐으나 두 번째 문서 검증에서 중단됐다.

```text
ValueError: evidence phrases absent from source
```

#### 원인

Silver 원문에는 PDF 줄바꿈 때문에 비정상 공백이 포함되어 있었다.

```text
Silver 원문: 목적으 로 한다
Gemini 출력: 목적으로 한다
```

Gemini는 의미를 바꾸지 않고 표면 공백을 정상화했지만 기존 검증기는 완전한
문자열 일치만 허용했다.

#### 잘못된 해결 방향

- evidence 검증을 제거하는 방법
- LLM evidence를 원문 근거인 것처럼 그대로 저장하는 방법
- 단순 유사도 threshold만으로 evidence를 승인하는 방법

이 방법들은 hallucinated evidence가 통과할 가능성이 있어 채택하지 않았다.

#### 해결

다음 순서로 evidence를 검증한다.

```text
1. exact substring 일치
2. 실패하면 원문과 evidence의 공백만 제거하여 비교
3. compact 문자열이 일치하면 원문의 실제 시작·종료 offset 계산
4. 원문의 원래 공백을 보존한 exact span으로 evidence 교체
5. compact 비교도 실패하면 전체 batch 차단
```

예:

```text
LLM evidence: 목적으로 한다
저장 evidence: 목적으 로 한다
```

#### 검증

- PDF 줄바꿈 공백 정렬 테스트 통과
- 원문에 없는 evidence 차단 테스트 통과
- Gemini 실제 소량 호출 성공

### 4.4 목적 조항의 강제 오분류

#### 증상

하도급법 제1조 목적 조항이 `DEFINITION`으로 분류됐다.

```text
이 법은 ... 국민경제의 발전에 이바지함을 목적으로 한다.
→ DEFINITION
```

#### 원인

초기 taxonomy에는 12개 법적 기능만 존재했고 `PURPOSE` 또는 fallback이 없었다.
validator는 label을 최소 한 개 요구했기 때문에 모델이 가장 가까운 label을
강제로 선택했다.

#### 고려한 대안

1. `PURPOSE`를 13번째 법적 기능으로 추가
2. 빈 label 배열 허용
3. `UNCLASSIFIED` fallback 추가

#### 결정

12개 기능 체계를 유지하고 `UNCLASSIFIED`를 배타적 fallback으로 추가했다.

```text
["UNCLASSIFIED"]                  허용
["UNCLASSIFIED", "DEFINITION"]    거절
["UNCLASSIFIED", "SCOPE"]         거절
```

목적 조항이 항상 독립적인 검색 branch가 되어야 한다는 요구가 확인되면 이후
taxonomy major version에서 `PURPOSE` 추가를 다시 검토한다.

#### 추가 문제

`UNCLASSIFIED`를 추가한 것만으로는 모델이 기존 오분류를 바로 수정하지 않았다.
따라서 codebook에 다음 경계를 명시했다.

```text
순수 입법 목적은 정의나 적용 범위가 아니다.
다른 독립 법적 기능이 없으면 UNCLASSIFIED로 분류한다.
```

prompt와 cache를 함께 version-up하여 과거 응답이 재사용되지 않게 했다.

### 4.5 순차 표본의 정의 조항 편향

#### 증상

문서별 첫 20개를 분류한 결과 `DEFINITION`이 과도하게 많았다.

```text
40개 표본 중 DEFINITION label 출현: 30회
multi-label: 30/40
```

#### 원인

`--limit 20`은 무작위 또는 층화 추출이 아니라 문서 앞부분 20개를 선택한다.
두 문서 앞부분은 목적과 정의 조항에 집중되어 있다.

#### 판단

이 결과만으로 전체 데이터의 label 분포나 모델 편향을 추정할 수 없다.

#### 후속 조치

다음 실험부터 조항 제목, 구조 유형, 예상 법적 기능을 기준으로 층화표본을
생성해야 한다.

예상 strata:

- 목적
- 정의
- 적용 범위
- 계약·서면 의무
- 금지행위
- 지급
- 기술자료
- 조정 절차
- 공정거래위원회 권한
- 시정조치
- 과징금·벌칙
- 부칙·삭제 조문

### 4.6 목록 조각에 부모 의미가 혼합되는 문제

#### 증상

다음과 같은 Item이 자체적으로 `DEFINITION + SCOPE`로 분류됐다.

```text
물품의 제조
물품의 판매
물품의 수리
건설
```

이 텍스트에는 자체 주어와 서술어가 없다. 부모 문장과 결합되어야 정의의
구성요소라는 의미가 생긴다.

#### 원인 1: 문맥 누락

초기 payload는 하위 노드에 실제 Article 제목을 제공하지 않았다.

```python
article_title = node["title"] if node["node_type"] == "ARTICLE" else None
```

따라서 Paragraph, Item, Subitem은 대부분 `article_title=None`이었다.

#### 원인 2: 의미의 출처 미분리

초기 schema의 단일 `labels` 컬럼에는 다음이 섞여 있었다.

- 현재 텍스트가 자체적으로 수행하는 법적 기능
- 부모 lead-in에서 전달된 법적 기능
- 검색을 위해 최종적으로 사용할 기능

#### 해결

```text
intrinsic_labels
→ 현재 proposition 자체가 수행하는 기능

inherited_labels
→ 법령 계층과 enumeration 관계로부터 전달된 기능

labels
→ intrinsic ∪ inherited로 계산한 effective label
```

예:

```json
{
  "semantic_unit_type": "LIST_FRAGMENT",
  "intrinsic_labels": ["UNCLASSIFIED"],
  "inherited_labels": ["DEFINITION"],
  "labels": ["DEFINITION"]
}
```

#### 문맥 payload 개선

LLM 입력에 다음 항목을 추가했다.

- 실제 Article citation과 title
- 현재 node type
- ancestor path
- immediate parent lead-in
- semantic unit type

semantic unit type:

- `FULL_PROPOSITION`
- `ENUMERATION_LEAD`
- `LIST_FRAGMENT`
- `PURPOSE`
- `DELETED`

#### 상속 정책

초기 보수적 정책에서는 enumeration lead로부터 다음 label만 상속한다.

- `DEFINITION`
- `SCOPE`
- `CONDITION`
- `EXCEPTION`

다음 label은 자동 상속하지 않는다.

- `REFERENCE`
- `DELEGATION`
- `AUTHORITY`
- `PROCESS`
- `SANCTION`
- `UNCLASSIFIED`

`OBLIGATION`과 `PROHIBITION` 상속은 주체·행위·목적어 구조가 추가로 검증된
이후 도입한다.

#### v4 부분 검증

하도급법 20개에서 목록 조각 5개가 다음과 같이 처리됐다.

```text
intrinsic: UNCLASSIFIED
inherited: DEFINITION
effective: DEFINITION
```

이는 부모 의미와 현재 텍스트의 자체 의미를 분리하려는 설계가 실제 데이터에서
작동한 첫 검증 결과다.

### 4.7 LLM self-reported confidence 과신

#### 증상

초기 40개 결과의 confidence가 모두 `0.90`, `0.95`, `1.00`이었다. 목록 조각과
애매한 multi-label 결과도 높은 값을 받았다.

#### 원인

confidence는 모델이 스스로 출력한 값이며 실제 정확도로 calibration된 확률이
아니다.

#### 결정

현재 confidence는 참고용 원시 신호로만 저장한다. production quality gate에는
단독으로 사용하지 않는다.

향후 calibrated confidence 후보:

- 동일 입력 반복 분류의 label 일치율
- 다른 모델 또는 prompt 간 합의율
- deterministic rule과의 일치 여부
- 형제 노드 label 일관성
- evidence coverage
- human annotation 결과

### 4.8 Gemini free-tier quota 소진과 부분 발행

#### 증상

v4로 문서별 20개를 재분류하는 과정에서 다음 상태가 발생했다.

```text
하도급법: v4 발행 완료
시행령: quota 초과로 실패, 기존 v3 유지
```

오류:

```text
Gemini free-tier quota or rate limit was exceeded.
```

#### 영향

동일 경로 아래 두 문서가 서로 다른 prompt/schema 의미 버전을 가지게 됐다.

```text
KR_FSTA_ACT                  legal_function_semantic_frame_v4
KR_FSTA_ENFORCEMENT_DECREE  legal_function_semantic_frame_v3
```

두 파일을 같은 실험 결과로 비교하거나 합쳐서는 안 된다.

#### 근본 원인

기존 script는 한 문서의 API 호출이 완료되면 즉시 해당 Parquet을 발행했다.
전체 실행 단위가 아니라 문서 단위로 commit된 셈이다.

#### 해결

classification-gated two-phase publish를 도입했다.

```text
Phase 1
모든 문서 분류 및 검증
→ 성공 결과를 cache에 저장
→ 메모리에 publication 대기

Phase 2
모든 문서가 성공했을 때만
→ 모든 pending Parquet 작성
→ read-back 검증
→ final path로 replace
→ manifest 발행
```

중간에 quota 또는 validation 오류가 발생하면 Phase 2에 진입하지 않는다.

#### cache의 역할

quota가 초기화된 후 같은 명령을 재실행하면 성공한 batch는 cache에서 읽는다.
따라서 이미 성공한 요청에 대해 API token을 다시 사용하지 않는다.

단, cache key는 다음 값 전체에 종속된다.

- provider
- model
- prompt version
- cache namespace
- batch payload

`--limit`이나 batch 구성이 달라지면 동일 proposition이라도 cache key가 달라질
수 있다. 향후 proposition-level cache 또는 stable batch planner를 검토한다.

## 5. 현재 데이터 상태

이 문서 작성 시점의 상태:

```text
하도급법 legal_functions.parquet
→ prompt v4
→ 20 records

시행령 legal_functions.parquet
→ prompt v3
→ 20 records
```

따라서 현재 결과는 혼합 버전이며 통합 평가 대상으로 사용하면 안 된다.

확인 명령:

```bash
.venv/bin/python - <<'PY'
import glob
import pyarrow.parquet as pq

for path in glob.glob("data/silver/*/*/legal_functions.parquet"):
    rows = pq.read_table(path).to_pylist()
    print(rows[0]["source_id"], rows[0]["prompt_id"], len(rows))
PY
```

## 6. 복구 절차

Gemini quota가 초기화된 후 기존과 동일한 batch 조건으로 실행한다.

```bash
.venv/bin/python scripts/classify_legal_functions.py --limit 20
```

예상 동작:

1. 하도급법 v4의 성공 batch는 cache hit
2. 시행령의 미완료 batch만 Gemini API 호출
3. 두 문서 분류가 모두 완료된 경우에만 두 Parquet 발행
4. 두 manifest의 `prompt_id`가 v4인지 확인

복구 성공 기준:

```text
KR_FSTA_ACT                  legal_function_semantic_frame_v4 20
KR_FSTA_ENFORCEMENT_DECREE  legal_function_semantic_frame_v4 20
```

## 7. 전체 실행 전 품질 게이트

다음 조건을 모두 만족하기 전에는 전체 proposition 분류를 실행하지 않는다.

- [ ] 두 문서의 prompt/schema/cache version이 동일함
- [ ] 층화표본 50~100개가 준비됨
- [ ] `intrinsic_labels`와 `inherited_labels`를 사람이 구분 검수함
- [ ] 목록 조각에서 부모 label 상속이 일관됨
- [ ] `UNCLASSIFIED`가 다른 label과 함께 존재하지 않음
- [ ] evidence phrase가 모두 원문 span으로 정렬됨
- [ ] `OBLIGATION`과 `AUTHORITY`의 경계 사례를 검수함
- [ ] 단순 법령 인용과 규범적 `REFERENCE`를 구분함
- [ ] self-reported confidence를 단독 quality gate로 사용하지 않음
- [ ] quota 중단 시 final dataset이 부분 발행되지 않음

## 8. 다음 개선 작업

### 우선순위 1: 층화 샘플러

`--limit N` 대신 문서·Article·node type·길이·예상 기능별 표본을 생성한다.

### 우선순위 2: human audit artifact

```text
data/artifacts/legal_function_audits/
├── classification_preview.csv
├── label_distribution.yaml
├── unclassified_cases.csv
├── multi_label_cases.csv
├── inheritance_cases.csv
├── low_confidence_cases.csv
└── evidence_audit.jsonl
```

### 우선순위 3: reference edge 분리

문장에 법령 인용이 존재한다는 사실은 deterministic `CITES` edge로 저장한다.
규범적으로 다른 조항을 적용하거나 준용하는 의미만 `REFERENCE` label 후보로
사용한다.

### 우선순위 4: stable cache grain

현재 batch payload 기반 cache 외에 proposition 단위 semantic cache를 추가해
batch size나 sampling 방식이 바뀌어도 성공 결과를 재사용할 수 있게 한다.

### 우선순위 5: run-level snapshot

다음 metadata를 갖는 run manifest를 도입한다.

- classification run ID
- provider/model
- prompt/schema version
- source version set
- requested/completed/cached proposition 수
- token usage
- 시작·종료 시각
- status: `STARTED`, `FAILED`, `VALIDATED`, `PUBLISHED`

## 9. 핵심 교훈

1. JSON Schema가 표준상 유효해도 provider의 Structured Outputs에서 지원된다고
   가정하면 안 된다.
2. LLM evidence는 의미적으로 맞는 것만으로 부족하며 원본 span으로 다시
   정렬되어야 한다.
3. taxonomy에 fallback이 없으면 모델은 틀린 label을 강제로 선택한다.
4. 법령의 Item과 Subitem은 독립 문장이 아니라 부모 의미를 상속하는 조각일 수
   있다.
5. LLM의 self-reported confidence는 정확도가 아니다.
6. 순차 `limit` 표본은 문서 앞부분의 내용에 편향된다.
7. API cache와 dataset version은 prompt 변경 시 함께 갱신해야 한다.
8. 외부 API quota 오류는 예상 가능한 운영 조건이며 부분 발행을 막아야 한다.
9. 성공한 API 호출과 publish transaction은 별개의 단계로 관리해야 한다.
10. 전체 토큰을 사용하기 전에 작은 대표 표본과 human audit으로 설계를
    검증해야 한다.
