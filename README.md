# Preference-Aware Legal QA Factory

법률 원문과 고객 Reference QA를 분리해 관리하고, 고객의 선호와 잠재적인
검색·추론 패턴을 재현 가능한 Retrieval Blueprint로 컴파일하여 고품질 법률
QA 데이터셋을 생성하는 lineage-first 데이터 파이프라인입니다.

## Current executable scope

```bash
make setup
make ingest
make bronze
make check
make demo
```

현재 `ingest`와 `bronze`는 실행 가능합니다. 이후 단계의 파일과 계약은 전체
경계를 먼저 고정하기 위해 scaffold로 제공되며, 구현 전 실행하면 명시적으로
종료합니다.

## End-to-end stages

1. Source registration and immutable Raw ingestion
2. Traceable PDF-block Bronze Parquet
3. Source-type-aware Silver legal nodes
4. Customer Reference QA preference profiling
5. Gold-dataset lineage inference
6. Retrieval Blueprint compilation and caching
7. Budgeted question and grounded-answer generation
8. Deterministic, LLM-judge, and human evaluation
9. Immutable dataset publication and change-impact rebuild

설계 근거와 데이터 모델은 `docs/architecture.md`와 `docs/data_model.md`에
정리되어 있습니다. API 키는 `.env.example`을 복사한 로컬 `.env`에만 넣습니다.
