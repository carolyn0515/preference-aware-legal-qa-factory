# Architecture

The factory separates legal truth, customer preference, and model judgment.

```text
Registered PDFs
  -> Raw immutable objects + manifests
  -> Bronze traceable physical blocks
  -> Silver canonical legal nodes + bitemporal versions
  -> Retrieval index

Customer Reference QA
  -> preference profile
  -> claim decomposition
  -> observed/inferred evidence lineage
  -> versioned Retrieval Blueprints + cache

Blueprint + current legal snapshot
  -> question generation
  -> grounded answer generation
  -> deterministic gates
  -> LLM judge
  -> risk-stratified human review
  -> immutable published QA snapshot
```

## Non-negotiable semantics

- Reference answers represent customer preference, not legal truth.
- `OBSERVED` lineage is never merged with `INFERRED` lineage.
- Every published legal claim resolves to evidence in a pinned source snapshot.
- Legal valid time and system transaction time are stored independently.
- Caches include source, profile, blueprint, parser, embedding, and prompt versions.
- A legal-source change creates an impact set; only affected downstream artifacts rebuild.
