# REXEL-Based Knowledge Graph Extraction Pipeline

## Method, Implementation, and Results

## Abstract
This report presents an end-to-end knowledge graph (KG) extraction pipeline that combines text denoising, phonetic correction, joint neural extraction (REXEL), optional LLM-based triple verification, and Neo4j graph storage. The core extraction component is a joint document information extraction model that predicts mentions, entity clusters, and relations from shared contextual representations. Experiments on the Luke Skywalker example with the best checkpoint (`rexel_joint_v8_best.pt`) show that extraction quality is strongly affected by judge configuration: strict judge mode yields fewer high-confidence edges, while disabling judge maximizes recall and graph density. The system supports both JSON export and live Neo4j graph materialization for visualization and querying.

## 1. Introduction
The objective is to convert noisy, long-form text into structured graph data:
- Entities as nodes
- Relations as typed edges

Unlike fully modular pipelines where NER, coreference, and relation extraction are separate disconnected stages, this system uses a joint REXEL-style approach for extraction, followed by optional verification.

## 2. Pipeline Architecture
Current runtime order:
1. `denoise_text`
2. `correct_phonetics`
3. `rexel_extract`
4. `judge_triples`
5. `store_to_neo4j`

Operational behavior:
- Accepts raw text or input file
- Produces draft triples via REXEL
- Optionally filters triples with GraphJudge (LLM)
- Writes final graph to `kg_output.json` and Neo4j

## 3. REXEL Approach (Core Method)
The implemented REXEL torch backend is a joint DocIE model with one shared transformer encoder and multiple task heads.

### 3.1 Shared Encoder
- RoBERTa-base provides contextual token embeddings over the document.

### 3.2 Joint Prediction Heads
- **Span head**: mention/span proposal score
- **Type head**: entity type classification
- **Relation head**: relation prediction for entity pairs
- **Coref head**: mention/entity coreference score
- **Link projector**: embedding projection for entity linking compatibility

### 3.3 Mention Proposal and Filtering
Candidate spans are enumerated up to a bounded width (capped to 4 at inference), scored, then filtered.

Filters remove:
- Cross-sentence spans
- Punctuation-heavy spans
- Clause-like spans with interior stopwords

Then overlap suppression keeps higher-confidence mentions and removes fragmentary overlaps.

### 3.4 Entity Clustering and Coreference Merge
Two-stage merge strategy:
1. String-normalization grouping (case/punctuation variants)
2. Neural coref merge with high threshold + shared-token guard

This reduces alias duplication and improves relation consistency.

### 3.5 Relation Extraction
For each ordered entity pair:
- Relation logits are predicted
- Softmax confidence is computed
- Non-NA relations above threshold are retained
- Duplicate triples are removed

### 3.6 Optional Entity Linking
A linking hook exists for Wikidata enrichment. If unavailable, extraction still proceeds (non-blocking fallback).

## 4. Judge Layer (GraphJudge)
The judge verifies candidate triples using an LLM prompt with strict JSON output schema.

Enhancements in current implementation:
- Wikidata property mapping (`P`-codes to readable relation labels)
- Evidence-focused context selection (relevant sentence windows, not full noisy passage)
- Alias-aware verification using entity cluster mentions and canonical names
- Backward-compatible parser for both `is_true` and `valid` JSON outputs
- Strictness control:
  - `JUDGE_STRICT=true`: reject uncertain triples
  - `JUDGE_STRICT=false`: keep undecidable triples and recover some false negatives via lexical co-occurrence fallback
- Full bypass mode:
  - `JUDGE_ENABLED=false`: skip verification and keep all REXEL triples

Current observation: judge is conservative and can significantly reduce recall.

## 5. Storage and Visualization
Final outputs are persisted in two forms:
- **JSON**: `kg_output.json` (always)
- **Neo4j**: Entity nodes + typed relationships (if connection succeeds)

Neo4j writer supports both triple schemas:
- `subject/predicate/object`
- `head/relation/tail`

This resolves prior cases where connection succeeded but no relationships appeared.

## 6. Experimental Setup
Input:
- `data/example_docs/luke_skywalker.txt`

Checkpoint:
- `checkpoints/rexel_joint_v8_best.pt` (best available)

Inference thresholds:
- Span threshold = `0.10`
- Relation threshold = `0.05`

Three modes were compared under identical settings.

## 7. Results

### 7.1 Strict Judge ON
Configuration:
- `JUDGE_ENABLED=true`
- `JUDGE_STRICT=true`

Observed:
- REXEL produced: **29 entities, 26 draft triples**
- Verified: **5 / 26**
- Final graph: **29 nodes, 5 edges**
- Neo4j: connected and written

### 7.2 Strict Judge OFF
Configuration:
- `JUDGE_ENABLED=true`
- `JUDGE_STRICT=false`

Observed:
- REXEL produced: **26 entities, 24 draft triples**
- Verified: **9 / 24**
- Final graph: **26 nodes, 9 edges**
- Neo4j: connected and written

Interpretation:
- Non-strict mode now performs better than before due to evidence-focused judging and lexical-support recovery.

### 7.3 Judge Disabled
Configuration:
- `JUDGE_ENABLED=false`

Observed:
- REXEL produced: **25 entities, 20 draft triples**
- Final graph: **25 nodes, 20 edges**
- Neo4j: connected and written

## 8. Analysis
### 8.1 Main Bottleneck
The judge stage is currently the largest edge bottleneck. Disabling judge gives substantially denser graphs.

### 8.2 Precision–Recall Tradeoff
- Strict judge: precision-oriented, low recall
- Non-strict judge: moderate recall increase
- Judge disabled: highest recall and density, potentially more noise

### 8.3 Run-to-Run Variance
Counts can vary across runs due to LLM-driven denoising and phonetic correction, which slightly change upstream text and therefore downstream extraction outcomes.

## 9. Error Patterns and Observations
Typical rejection causes:
- Canonicalization artifacts (name variants, punctuation splits)
- Conservative LLM verification despite plausible evidence
- Relation semantic mismatch between extracted relation and passage wording

Mitigations already in place:
- Better relation hinting via PID label expansion
- Judge context set to original text for fuller evidence
- Evidence-focused context extraction around head/tail aliases
- Alias-aware matching from entity clusters
- Non-strict lexical support fallback for conservative LLM rejections
- Configurable judge bypass for recall-first workflows

## 10. Conclusion
The pipeline is fully operational with Neo4j integration and graph visualization capability. REXEL extraction is productive and can generate dense graphs, while the LLM judge introduces a strong precision bias that suppresses many edges. For exploratory KG construction and downstream analysis, judge-disabled mode is currently most practical. For curated outputs, strict judge mode is suitable when lower recall is acceptable.

## 11. Recommended Operating Modes
- **Demo / Exploration**: `JUDGE_ENABLED=false`
- **Balanced Filtering**: `JUDGE_ENABLED=true`, `JUDGE_STRICT=false`
- **Precision-First Curation**: `JUDGE_ENABLED=true`, `JUDGE_STRICT=true`

Practical note:
- `JUDGE_STRICT` has effect only when `JUDGE_ENABLED=true`.

## Appendix A: Example Command
```bash
python -m kg_pipeline.main \
  --input-file data/example_docs/luke_skywalker.txt \
  --rexel-backend torch \
  --rexel-checkpoint checkpoints/rexel_joint_v8_best.pt \
  --rexel-span-threshold 0.10 \
  --rexel-relation-threshold 0.05
```
