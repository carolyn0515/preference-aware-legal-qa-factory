# Preference-Aware Legal QA Factory

고객 Reference Q&A에서 답변 선호와 잠재적 검색 lineage를 추론하고,
최신 법률 근거에 연결된 버전형 QA 데이터셋을 생성하기 위한 데이터 플랫폼입니다.

## Current milestone

현재 구현 범위는 법률 PDF를 불변 Raw Object로 적재하는 단계입니다.

```text
Source YAML
→ input PDF validation
→ SHA-256 deduplication
→ immutable Raw Object
→ ingestion manifest
→ quarantine on invalid PDF
```

## Setup

```bash
make setup
make ingest
make test
```

실제 PDF와 `.env`는 저장소에 커밋하지 않습니다.

## Next milestone

Raw PDF를 page/block 단위의 Bronze Parquet로 변환하고, 모든 레코드를
원본 PDF의 페이지와 좌표까지 추적할 수 있게 만드는 것이 다음 목표입니다.

