# Data model

| Layer | Grain | Storage | Why |
|---|---|---|---|
| Raw | one immutable source object | PDF + YAML manifest | exact replay and audit |
| Bronze | one physical PDF text block | Parquet | typed batch validation and traceability |
| Silver | one versioned legal node | Parquet | canonical retrieval corpus |
| Reference | one customer QA pair | encrypted object/table | preference evidence |
| Artifact | one profile, lineage edge, or blueprint | Parquet/DB | reproducible derivation |
| Published | one evaluated QA row | Parquet + manifest | portable versioned delivery |

Parquet is used for immutable analytical snapshots. It is not used as a transactional
workflow database, vector index, or low-latency cache.
