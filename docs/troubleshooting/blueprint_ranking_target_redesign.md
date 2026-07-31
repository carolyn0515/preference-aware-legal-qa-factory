# Blueprint Ranking Target Redesign Troubleshooting

## 1. 문서 목적

이 문서는 Gold QA에서 추론한 검색·답변 패턴을 새로운 질문에 추천하는
Blueprint 학습 파이프라인을 설계하면서 발생한 문제와 해결 과정을 기록한다.

단순히 최종 성능만 제시하지 않고 다음을 함께 보존한다.

- 최초 문제 정의와 데이터 구조
- 평가 누수 발견 및 제거 과정
- 실패한 실험과 기각 근거
- 타깃 재설계의 논리
- 데이터 품질 격리 기준
- 단계별 정량 결과
- 아직 남아 있는 한계와 다음 검증 조건

이 문서에서 사용하는 모든 성능은 `synthetic_v3` 기반의
`DIAGNOSTIC_ONLY` 결과다. 고객 Gold 및 휴먼 검수 데이터로 검증되기 전에는
production 성능으로 표현하지 않는다.

## 2. 대상 데이터와 평가 단위

### 2.1 데이터 규모

| 구분 | 수량 | 의미 |
|---|---:|---|
| 최초 독립 QA seed | 24 | 서로 다른 법률 주제 |
| seed당 형태 | 13 | 원문 1개와 문맥·표현 변형 12개 |
| 물리적 QA 행 | 312 | Parquet에 저장된 전체 행 |
| Answer claim | 624 | QA당 평균 2개 문장 |
| 법률 주제 | 24 | `written_contract`, `direct_payment` 등 |
| 최초 exact pattern | 17 | Answer flow와 retrieval action의 정확한 조합 |
| 독립 평가 그룹 | 24 | `parent_example_id` 기준 |

312행은 312개의 독립적인 법률 사례가 아니다.

```text
312 physical rows
└── 24 parent groups
    └── 그룹당 13개 표현·문맥 변형
```

따라서 train/test 분할은 반드시 `parent_example_id` 단위로 수행한다. 동일한
seed에서 파생된 질문이 학습과 평가에 동시에 포함되면 표현 변형을 기억하는
것만으로 정답을 맞힐 수 있어 성능이 과대평가된다.

### 2.2 최종 유효 행

최종 평가에서는 312행 중 24행을 삭제하지 않고 격리했다.

| 상태 | 행 수 | 처리 |
|---|---:|---|
| `ranking_training_eligible=true` | 288 | 학습 및 평가 사용 |
| `ranking_training_eligible=false` | 24 | lineage 보존, 랭킹 학습 제외 |

격리 사유는 다음과 같다.

```text
QUESTION_INTENT_CHANGED_WITHOUT_ANSWER_REGENERATION
```

`exception_check` 변형은 모든 질문 앞에 “예외나 추가 요건도 포함해서
확인하고 싶다”는 요구를 추가했다. 그러나 Answer와 retrieval lineage는 원본을
그대로 사용했다. 이는 단순 paraphrase가 아니라 질문 요구사항을 변경한
augmentation이므로 입력과 정답이 불일치한다.

## 3. 최초 문제 정의

최초 타깃은 다음 두 배열의 정확한 조합이었다.

```python
pattern_id = hash({
    "answer_flow": answer_flow,
    "retrieval_actions": retrieval_actions,
})
```

예:

```json
{
  "answer_flow": [
    "CONCLUSION",
    "CONDITION",
    "PRACTICAL_GUIDANCE"
  ],
  "retrieval_actions": [
    "SEARCH_ANCHOR",
    "EXPAND_CHILDREN"
  ]
}
```

배열의 원소나 순서가 하나만 달라도 별개의 `pattern_id`가 됐다. 모델의 최종
출력은 17개 pattern 중 하나를 고르는 단일 분류였다.

```text
Question
  ↓
유사 학습 질문 Top-K
  ↓
weighted vote
  ↓
17개 exact pattern 중 하나
```

### 문제

독립 사례 24개에 클래스가 17개였고, 대부분의 pattern support는 1개였다.

| 독립 support | 특징 |
|---:|---|
| 4 | 최대 support pattern |
| 2~3 | 일부 pattern |
| 1 | 대부분 pattern |

그룹 하나를 홀드아웃하면 singleton pattern은 학습 데이터에서 사라진다. 이
상태에서 해당 exact hash를 맞히는 것은 일반화 문제가 아니라 관찰하지 못한
클래스를 맞히는 문제가 된다.

## 4. 평가 누수 제거

### 4.1 초기 평가의 문제

초기 `leave-one-row-out`은 한 행만 평가에서 제외했다. 동일 parent의 나머지
12개 변형이 학습 데이터에 남아 있었다.

```text
잘못된 평가

평가: “대금은 언제 지급하나요?”
학습: “계약 검토 중입니다. 대금은 언제 지급하나요?”
학습: “내부 감사 관점에서 대금은 언제 지급하나요?”
```

### 4.2 변경

평가 전략을 다음과 같이 변경했다.

```text
LEAVE_ONE_PARENT_GROUP_OUT
```

평가 대상 parent의 13개 행을 모두 학습 데이터에서 제거했다.

### 4.3 결과

누수를 제거한 exact classification 결과는 다음과 같았다.

| 모델 | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| weighted kNN k=1 | 0.1154 | 0.0612 | 0.0961 |
| weighted kNN k=3 | 0.1186 | 0.0545 | 0.0866 |
| weighted kNN k=5 | 0.0897 | 0.0366 | 0.0534 |

성능은 낮았지만 평가 결과는 이전보다 신뢰할 수 있게 됐다. 이 단계에서 낮은
수치를 숨기지 않고 타깃 정의를 재검토했다.

## 5. Exact Classification에서 Graded Ranking으로 전환

### 5.1 변경 이유

기존 평가는 세 검색 행동 중 두 개를 맞혀도 `pattern_id`가 다르면 완전 오답으로
처리했다.

```text
Gold:
SEARCH_ANCHOR
+ EXPAND_CHILDREN
+ FOLLOW_DECREE_DELEGATION

Prediction:
SEARCH_ANCHOR
+ EXPAND_CHILDREN

기존 평가: 완전 오답
```

따라서 `question × candidate blueprint`별 graded relevance를 생성했다.

```text
312 queries × 17 candidates = 5,304 ranking pairs
```

초기 relevance는 다음 두 요소로 구성했다.

```text
0.55 × Answer flow LCS similarity
+ 0.45 × Retrieval action weighted Jaccard
```

계수는 설정 파일로 분리했으며 최종 확정값이 아니다.

### 5.2 결과

| 모델 | nDCG@3 | nDCG@5 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|---:|
| ranking k=1 | 0.5273 | 0.5692 | 0.4455 | 0.5000 | 0.3239 |
| ranking k=3 | 0.5422 | **0.5910** | 0.4679 | 0.5288 | **0.3367** |
| ranking k=5 | **0.5473** | 0.5903 | **0.4872** | **0.5385** | 0.3125 |

Exact accuracy만으로 볼 수 없었던 부분 정답의 순위 품질을 측정할 수 있게 됐다.

## 6. Semantic Compatibility Reranker

### 6.1 가설

유사 질문 투표 외에 질문 intent와 Blueprint 구성 간 호환성을 직접 평가하면
singleton pattern도 일부 추천할 수 있다고 판단했다.

예:

| 질문 intent | 기대 Answer role | 기대 Retrieval action |
|---|---|---|
| 조건 | `CONDITION` | `EXPAND_CHILDREN` |
| 예외 | `EXCEPTION_NOTICE` | `SEARCH_ROLE_SIBLINGS` |
| 절차 | `PROCEDURE` | `EXPAND_CHILDREN` |
| 기한 | `CONDITION` | `SEARCH_ROLE_SIBLINGS` |
| 제재 | `SANCTION_NOTICE` | `FOLLOW_DECREE_DELEGATION` |

최종 후보 점수:

```text
0.65 × Neighbor relevance
+ 0.35 × Semantic compatibility
```

### 6.2 결과

| 지표 | Ranking v1 k=3 | Semantic v2 k=3 | 변화 |
|---|---:|---:|---:|
| nDCG@3 | 0.5422 | 0.5765 | +0.0343 |
| nDCG@5 | 0.5910 | 0.6207 | +0.0297 |
| Recall@1 | 0.1154 | 0.1506 | +0.0352 |
| Recall@3 | 0.4679 | 0.5256 | +0.0577 |
| Recall@5 | 0.5288 | 0.5801 | +0.0513 |
| MRR | 0.3367 | 0.3846 | +0.0479 |
| 평균 exact rank | 6.04 | 5.17 | -0.87 |

개선은 확인했지만 17개 exact 조합을 직접 선택하는 구조적 난점은 해결되지
않았다.

## 7. Hierarchical Target 도입

### 7.1 타깃 분해

17개 exact pattern을 삭제하지 않고 세 단계로 분해했다.

```text
5개 Pattern Family ranking
        ↓
Retrieval action ranking
        ↓
Answer role ranking
        ↓
세부 Blueprint 조립
```

도입한 Family:

| Family | 의미 |
|---|---|
| `DIRECT_RULE` | 기본 의무·금지·허용 판단 |
| `CONDITION_EXCEPTION` | 요건·조건·예외 판단 |
| `DEADLINE_CALCULATION` | 기한·기준일 판단 |
| `PROCEDURE_DELEGATION` | 절차·신청·위임 탐색 |
| `SANCTION_REMEDY` | 제재·과징금·구제 |

### 7.2 최초 Hierarchical 결과

| 지표 | k=1 | k=3 | k=5 |
|---|---:|---:|---:|
| Family Recall@1 | 0.4263 | **0.4808** | 0.4776 |
| Family Recall@2 | 0.8301 | **0.8429** | 0.8397 |
| Family Recall@3 | 0.9167 | 0.9006 | **0.9167** |
| Family MRR | 0.6779 | **0.7058** | 0.7051 |
| Family nDCG@3 | 0.7772 | 0.7821 | **0.7894** |
| Retrieval Action MAP | 0.7809 | **0.9069** | 0.9043 |
| Answer Role MAP | 0.8452 | 0.8835 | **0.8868** |

Top-2 routing과 구성요소 랭킹은 개선됐지만 Family Top-1은 48.08%에 머물렀다.

## 8. Family Recall 저하 원인 Audit

### 8.1 Audit 설계

다음을 각각 기록하는 `family_confusion_audit.json`을 추가했다.

- 행 단위 confusion matrix
- parent group 단위 confusion matrix
- augmentation variant별 정확도
- 예상 Family와 질문 기반 hint
- 최종 예측 Family
- 오류 유형

오류를 두 종류로 분해했다.

| 오류 유형 | 정의 |
|---|---|
| `QUESTION_TARGET_MISMATCH` | 질문에서 관찰되는 intent와 Gold Family가 다름 |
| `NEIGHBOR_OVERRIDE` | 질문 hint는 맞았지만 이웃 투표가 이를 뒤집음 |

### 8.2 최초 Audit 결과

| 항목 | 수치 |
|---|---:|
| 전체 오답 행 | 162 |
| `QUESTION_TARGET_MISMATCH` | 140 |
| `NEIGHBOR_OVERRIDE` | 22 |
| 질문–타깃 불일치 비중 | 86.4% |

모델의 이웃 검색보다 타깃 정의가 더 큰 문제였다.

### 8.3 근본 원인

당시 `pattern_family`는 다음 정보를 혼합했다.

```text
Gold metadata intent
+ Gold Answer flow
+ Gold retrieval actions
```

그러나 추론 시점에 사용할 수 있는 입력은 질문뿐이다.

```text
Training target: 질문 + 미래의 답변 정보
Inference input: 질문만
```

예를 들어 질문은 조건 문의처럼 보이지만 Gold Answer에 시행령 탐색이나 절차
문장이 포함되었다는 이유로 `PROCEDURE_DELEGATION`이 정답이 될 수 있었다.
질문만 보고는 알 수 없는 latent 정보를 routing 정답에 포함한 것이다.

## 9. 기각된 Query Prefix Normalization

### 9.1 가설

합성 질문에 반복되는 다음 prefix가 법률 쟁점보다 강한 lexical signal이 된다고
판단했다.

```text
“내부 감사에서 문제 될 수 있는지 검토 중입니다.”
“계약서를 검토하고 있습니다.”
“실무 FAQ 형식으로 질문드립니다.”
```

prefix를 제거한 focused question으로 재평가했다.

### 9.2 결과

| 지표 | 변경 전 k=3 | Prefix 제거 k=3 | 변화 |
|---|---:|---:|---:|
| Family Recall@1 | 0.4808 | 0.4295 | -0.0513 |
| Family MRR | 0.7058 | 0.6801 | -0.0257 |
| Family nDCG@3 | 0.7821 | 0.7784 | -0.0037 |
| Retrieval Action MAP | 0.9069 | 0.8121 | -0.0948 |
| Answer Role MAP | 0.8835 | 0.8481 | -0.0354 |

### 9.3 결정

실험을 기각하고 코드를 복구했다.

공통 prefix는 노이즈였지만 작은 데이터에서는 유사한 문맥 변형을 연결하는
신호로도 사용되고 있었다. 성능 하락을 확인했으므로 직관만으로 전처리를
채택하지 않았다.

## 10. Routing Family와 Lineage Family 분리

### 10.1 변경

하나였던 Family를 두 필드로 분리했다.

```text
pattern_family
= 질문에서 관찰 가능한 routing intent

lineage_pattern_family
= Gold Answer flow와 retrieval lineage의 상세 구조
```

`pattern_family` 매핑:

| Reference intent | Routing Family |
|---|---|
| `prohibition`, `document_issuance` | `DIRECT_RULE` |
| `condition_lookup`, `exception_lookup` | `CONDITION_EXCEPTION` |
| `deadline` | `DEADLINE_CALCULATION` |
| `procedure`, `permission` | `PROCEDURE_DELEGATION` |
| `sanction` | `SANCTION_REMEDY` |

질문 intent가 없는 데이터에서는 기존 lineage 기반 Family를 fallback으로
사용하지만 label source를 함께 기록한다.

### 10.2 중간 결과

24개 inconsistent 행을 포함한 상태에서도 성능이 개선됐다.

| 지표 | 분리 전 | 분리 후·격리 전 |
|---|---:|---:|
| Family Recall@1 | 0.4808 | 0.6378 |
| Family MRR | 0.7058 | 0.7879 |
| Family nDCG@3 | 0.7894 | 0.8485 |

이는 이전 타깃에 추론 시 관찰할 수 없는 정보가 포함돼 있었다는 가설을
지지한다.

## 11. Label-Inconsistent Augmentation 격리

### 11.1 삭제 대신 격리한 이유

문제가 있는 행을 물리적으로 삭제하면 데이터가 왜 제외됐는지 추적하기
어렵다. 따라서 원본과 lineage는 유지하고 학습 eligibility를 별도 관리한다.

추가 필드:

```text
ranking_training_eligible
ranking_exclusion_reason
reference_intent
reference_topic
generation_variant_id
lineage_pattern_family
```

### 11.2 최종 평가 조건

```text
전체 312행
- exception_check 24행
= ranking-eligible 288행
```

평가 전략은 계속 `LEAVE_ONE_PARENT_GROUP_OUT`을 유지했다.

## 12. 최종 결과

### 12.1 Hierarchical Ranking

| 지표 | k=1 | k=3 | k=5 |
|---|---:|---:|---:|
| Answer Role MAP | 0.8461 | 0.8837 | **0.8847** |
| Family MRR | **0.8293** | 0.8229 | 0.8232 |
| Family nDCG@3 | **0.8728** | 0.8643 | 0.8690 |
| Family Recall@1 | **0.7083** | **0.7083** | **0.7083** |
| Family Recall@2 | **0.8715** | 0.8403 | 0.8368 |
| Family Recall@3 | **0.9583** | 0.9444 | 0.9549 |
| Retrieval Action MAP | 0.7809 | **0.9050** | 0.9010 |

### 12.2 주요 변화

| 지표 | 최초 Hierarchical | 최종 | 절대 변화 |
|---|---:|---:|---:|
| Family Recall@1 | 0.4808 | 0.7083 | +0.2275 |
| Family MRR | 0.7058 | 0.8293 | +0.1235 |
| Family nDCG@3 | 0.7894 | 0.8728 | +0.0834 |
| Family Recall@2 | 0.8429 | 0.8715 | +0.0286 |
| Family Recall@3 | 0.9167 | 0.9583 | +0.0416 |

### 12.3 최종 독립 그룹 Confusion Matrix

행은 Gold Family, 열은 예측 Family다.

| Gold \ Prediction | Condition | Deadline | Direct | Procedure | Sanction |
|---|---:|---:|---:|---:|---:|
| Condition | 1 | 0 | 3 | 0 | 0 |
| Deadline | 0 | 4 | 1 | 0 | 0 |
| Direct | 0 | 1 | 9 | 0 | 0 |
| Procedure | 0 | 0 | 2 | 2 | 0 |
| Sanction | 0 | 0 | 0 | 0 | 1 |

독립 그룹 기준:

```text
17 correct / 24 groups = 70.83%
```

남은 주요 혼동은 `CONDITION_EXCEPTION` 및 `PROCEDURE_DELEGATION`이
`DIRECT_RULE`로 예측되는 경우다.

## 13. 단계별 전체 결과 요약

서로 다른 문제 정의의 지표를 같은 의미로 비교하면 안 되므로 빈 칸은 해당
단계에서 측정하지 않은 값이다.

| 단계 | Target | 핵심 지표 | 결과 | 결정 |
|---|---|---|---:|---|
| Exact classification | 17 pattern ID | Accuracy k=1 | 0.1154 | 폐기 |
| Graded ranking v1 | 17 candidate relevance | nDCG@5 | 0.5910 | 개선 실험 유지 |
| Semantic ranking v2 | 17 candidate relevance | nDCG@5 | 0.6207 | 유지 |
| Hierarchical v1 | 5 Family + components | Recall@1 | 0.4808 | 원인 audit |
| Prefix normalization | 5 Family + components | Recall@1 | 0.4295 | 기각·복구 |
| Routing/lineage 분리 | 5 Family + components | Recall@1 | 0.6378 | 채택 |
| Inconsistent 행 격리 | 5 Family + components | Recall@1 | **0.7083** | 최종 채택 |

## 14. Silver Evidence 실행 검증

Top-2 Family를 실제 Silver 법률 트리에 실행하는 경로도 구현했다.

테스트 질문:

```text
발주자로부터 선급금을 받은 경우
수급사업자에게 언제까지 지급해야 하나요?
```

1위 근거:

| 항목 | 결과 |
|---|---|
| Source | `KR_FSTA_ACT` |
| Article | 제6조 |
| Paragraph | ① |
| 핵심 결론 | 선급금을 받은 날부터 15일 이내 |
| Question relevance | 0.610633 |
| Query coverage | 0.631579 |
| Hierarchy proximity | 1.000000 |
| Final evidence score | 0.694790 |

근거 점수:

```text
0.50 × Question relevance
+ 0.30 × Query coverage
+ 0.20 × Hierarchy proximity
```

이 계수 역시 고객·휴먼 relevance label을 확보한 후 튜닝해야 한다.

## 15. 재현 명령

### Policy 데이터 재컴파일

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/compile_blueprint.py \
  --dataset-dir \
  data/reference/processed/HYUNDAI_ENGINEERING/synthetic_v3/bd892b61d1a6e66f1ad21e72877d5f4809c52ed61fb7c856871d6522b1553976
```

### Hierarchical 성능 평가

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/evaluate_hierarchical_blueprint.py \
  --training-data \
  data/reference/processed/HYUNDAI_ENGINEERING/synthetic_v3/bd892b61d1a6e66f1ad21e72877d5f4809c52ed61fb7c856871d6522b1553976/policy/policy_training.parquet
```

### Family confusion audit

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/audit_family_confusion.py \
  --training-data \
  data/reference/processed/HYUNDAI_ENGINEERING/synthetic_v3/bd892b61d1a6e66f1ad21e72877d5f4809c52ed61fb7c856871d6522b1553976/policy/policy_training.parquet \
  --k 3
```

### Silver RAG 실행

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/execute_hierarchical_rag.py \
  --training-data \
  data/reference/processed/HYUNDAI_ENGINEERING/synthetic_v3/bd892b61d1a6e66f1ad21e72877d5f4809c52ed61fb7c856871d6522b1553976/policy/policy_training.parquet \
  --question "발주자로부터 선급금을 받은 경우 언제까지 지급해야 하나요?" \
  --output data/artifacts/blueprints/hierarchical_rag_demo.json
```

### 품질 검사

```bash
.venv/bin/ruff check \
  src/legal_qa_factory/blueprints \
  src/legal_qa_factory/retrieval \
  scripts \
  tests

PYTHONPATH=src .venv/bin/python -m pytest -q
```

최종 확인 시점의 결과:

```text
49 tests passed
Ruff passed
```

## 16. 채택한 설계 원칙

1. 물리적 행 수와 독립 표본 수를 구분한다.
2. 파생 질문은 `parent_example_id` 단위로 분할한다.
3. 추론 시 관찰할 수 없는 정보를 routing target에 포함하지 않는다.
4. routing intent와 latent lineage pattern을 별도 컬럼으로 보존한다.
5. 복합 Blueprint를 단일 class로 압축하지 않는다.
6. Family, retrieval action, answer role을 계층적으로 평가한다.
7. label-inconsistent 데이터는 삭제하지 않고 사유와 함께 격리한다.
8. 직관적으로 타당한 전처리도 성능 하락 시 기각한다.
9. 낮은 baseline과 실패 실험을 결과에서 제거하지 않는다.
10. 합성 데이터 지표를 production 성능으로 표현하지 않는다.

## 17. 한계와 해석상 주의

### 17.1 Production gate는 여전히 차단 상태

```text
NO_CUSTOMER_GOLD
INSUFFICIENT_HUMAN_REVIEWED_LABELS
INSUFFICIENT_PRODUCTION_ELIGIBLE_ROWS
```

최종 70.83%는 synthetic preference label에 대한 진단 성능이다.

### 17.2 독립 표본은 여전히 24개

물리적으로 288행을 평가하지만 독립 법률 사례는 24개다. 신뢰구간이 넓고 한
그룹의 성공 여부가 약 4.17%p를 변화시킨다.

### 17.3 격리 결정의 선택 편향 가능성

`exception_check`의 입력–정답 불일치는 의미론적으로 설명 가능하지만 평가
과정에서 발견됐다. 따라서 행 격리 후 성능은 별도의 untouched 고객 Gold에서
재검증해야 한다. 단순히 점수가 낮다는 이유로 제외한 것이 아니라는 근거로
`ranking_exclusion_reason`과 원본 lineage를 보존한다.

### 17.4 계수 튜닝 미완료

현재 Family 이웃·intent 가중치와 evidence reranking 계수는 휴먼 label에 대한
nested group validation으로 선택된 값이 아니다. 현 데이터에 반복적으로
맞추면 과적합이 발생한다.

### 17.5 Retrieval end-to-end benchmark 미완료

Blueprint routing, action ranking, answer-role ranking은 평가했지만 Silver
evidence executor의 전체 288개 질문에 대한 다음 지표는 아직 필요하다.

- Evidence Recall@1/3/5
- Article Recall@K
- MRR
- nDCG@K
- BM25 baseline 대비 개선율
- Top-1 branch 대비 Top-2 branch union 개선율

## 18. 다음 작업

1. 고객 Gold와 휴먼 검수 Family label 확보
2. 질문만 보고 판정 가능한 Family annotation guideline 확정
3. `CONDITION_EXCEPTION → DIRECT_RULE` 오류 사례 보강
4. `PROCEDURE_DELEGATION → DIRECT_RULE` 오류 사례 보강
5. Family·Action·Role에 서로 다른 k를 적용하는 운영 설정 분리
6. 전체 evidence retrieval benchmark 구현
7. untouched group validation으로 계수 재검증
8. 최종 Answer claim별 proposition citation 평가

