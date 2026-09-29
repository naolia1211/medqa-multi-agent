# MedQA Multi-Agent LLM — AI Security & Adversarial Robustness Evaluation

MedQA-USMLE multi-agent RAG system with **GCG jailbreak attacks, SmoothLLM defense, semantic jailbreak evaluation, and adversarial robustness benchmarking**.

This repository contains the implementation of a Medical Multi-Agent QA system and the Phase 2 security evaluation conducted as part of the final project.

---

## AI Security Research Contribution

My contribution to this project focuses on the **security evaluation and adversarial robustness of the MedQA multi-agent LLM system**.

### Research Scope

- Reproduced optimization-based **GCG (Greedy Coordinate Gradient) jailbreak attacks** against aligned LLMs.
- Implemented and evaluated **SmoothLLM** as a perturbation-based jailbreak defense.
- Evaluated **semantic jailbreak attacks** and residual defense weaknesses.
- Built an **attack-defense evaluation workflow** covering jailbreak attack success rate (ASR), benign utility, query/latency overhead, and transferability.
- Analyzed the effectiveness and limitations of SmoothLLM across different adversarial attack strategies.

### Research Overview

- **Attack:** GCG (Greedy Coordinate Gradient)
- **Defense:** SmoothLLM
- **System:** MedQA Multi-Agent LLM
- **Evaluation:** ASR, benign utility, overhead, transferability, and residual failure modes

➡️ **Implementation and experiments:** [`phase2_pair2_gcg_smoothllm/`](./phase2_pair2_gcg_smoothllm/)

---

## Phase 2 — Attack & Defense Evaluation

Phase 2 evaluates the adversarial robustness of the MedQA multi-agent system using **GCG jailbreak attacks** and **SmoothLLM defense**.

The complete attack/defense implementation, evaluation harness, experimental data, and measured results are available in:

[`phase2_pair2_gcg_smoothllm/`](./phase2_pair2_gcg_smoothllm/)

### Key Experimental Results

Experiments on **Vicuna-7B** showed:

- GCG jailbreak ASR: **97%**
- GCG + SmoothLLM ASR: **22%**
- Semantic jailbreak baseline ASR: **97%**
- Semantic jailbreak + SmoothLLM ASR: **77–87%**
- GCG suffix transfer to the Gemma base model: **0% ASR**
- SmoothLLM introduced approximately **10× query overhead**

These results show that SmoothLLM substantially reduced the effectiveness of optimization-based GCG attacks in the evaluated setup, while **semantic jailbreak attacks retained a high residual ASR**, highlighting limitations of perturbation-based defenses.

See the complete evaluation:

[`phase2_pair2_gcg_smoothllm/README.md`](./phase2_pair2_gcg_smoothllm/README.md)

---

## System Architecture

The system ingests **18 medical textbooks**, chunks them using the MedCPT tokenizer, and builds two retrieval indexes over the same chunk collection and `chunk_id`.

### Dense Retrieval

- **Vector database:** Qdrant
- **Embedding:** MedCPT-Article
- **Vector dimension:** 768
- **Distance:** Dot product
- **Search:** Exact search with HNSW disabled

### Sparse Retrieval

- **Search engine:** Elasticsearch
- **Retrieval:** BM25
- **Analyzer:** `medical_en`

### Hybrid Retrieval

Dense and sparse retrieval results are combined using **Reciprocal Rank Fusion (RRF)** based on the shared `chunk_id`.

The fused candidates are then reranked using MedCPT before the top-ranked chunks are passed to the LLM.

### Online RAG Flow

```text
Question
   │
   ├── MedCPT Query Embedding ──> Qdrant Dense Retrieval
   │
   └── BM25 ────────────────────> Elasticsearch Sparse Retrieval
                                      │
                         Reciprocal Rank Fusion (RRF)
                                      │
                               MedCPT Reranking
                                      │
                                  Top-5 Chunks
                                      │
                                      LLM
                                      │
                                   Answer
```

Embeddings are provided through a **configurable remote MedCPT embedding API**.

---

## Requirements

- **RAM:** ≥ 8 GB
- Docker
- Docker Compose
- Python 3
- Approximately 2 GB of disk space for indexes
- Network access to the configured MedCPT embedding API
- Hugging Face access for the initial tokenizer download

---

## Installation

Create and activate the Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## Start Qdrant and Elasticsearch

```bash
docker compose up -d
```

### Elasticsearch Permission Issue

If Elasticsearch fails during startup with an error similar to:

```text
failed to obtain lock ... AccessDeniedException
```

ensure the bind-mounted `es_storage/` directory is writable by Elasticsearch:

```bash
mkdir -p es_storage
sudo chown -R 1000:0 es_storage

docker compose restart elasticsearch
```

### Health Check

Wait approximately 30–60 seconds for Elasticsearch to initialize:

```bash
curl -s localhost:6333/healthz
curl -s localhost:9200/_cluster/health | jq .status
```

Expected results:

```text
healthz check passed
green/yellow
```

---

## Build the Retrieval Pipeline

Run the pipeline in the following order:

```bash
source venv/bin/activate

# Create Qdrant collection and Elasticsearch index
python setup_indices.py

# Process 18 medical textbooks into chunks
python build_corpus.py

# Build sparse BM25 index
python build_index.py --only sparse

# Build dense Qdrant index
python build_index.py --only dense

# Generate index manifest
python make_manifest.py

# Validate index consistency
python verify_index.py

# Run one real hybrid retrieval query
python smoke_test.py

# Evaluate RAG readiness using development questions
python readiness_test.py
```

Both indexes can also be built together:

```bash
python build_index.py
```

### Dense Index Checkpointing

Dense indexing can take several hours depending on the configured embedding service.

The pipeline stores checkpoints every 256 chunks under:

```text
index/.checkpoint/
```

If indexing is interrupted, resume with:

```bash
python build_index.py --only dense
```

To rebuild the dense index from the beginning:

```bash
python build_index.py --restart-dense
```

---

## Online Hybrid RAG

The online retrieval pipeline is:

```text
question
   ↓
MedCPT query embedding + BM25
   ↓
Reciprocal Rank Fusion
   ↓
MedCPT reranking
   ↓
Top-5 chunks
   ↓
LLM
```

No HyDE is used.

### LLM Configuration

LLM inference uses OpenRouter.

Store API keys locally in:

```text
openrouter_keys.txt
```

The file is excluded from Git through `.gitignore`.

Example format:

```text
<OPENROUTER_API_KEY>
```

The client supports round-robin key rotation and failover for rate-limit or provider errors.

---

## Open-Ended Question Answering

Example:

```bash
python ask.py "What is the first-line treatment for myelofibrosis?"
```

The output includes:

- Generated answer
- Top retrieved sources
- `chunk_id`
- Source metadata
- Reranking score

---

## MedQA Evaluation

Run the default evaluation:

```bash
python eval.py
```

Run a larger evaluation:

```bash
python eval.py \
  --limit 200 \
  --split MedQA-USMLE/questions/US/test.jsonl \
  --concurrency 8
```

Results are written to:

```text
index/eval_results.json
```

Online parameters are configured through `config.yaml`, including:

- reranking candidates
- retrieval `top_k`
- LLM model
- temperature
- reasoning configuration
- key file
- evaluation split
- evaluation limit
- concurrency

The main reusable online components are:

```text
retrieval.retrieve_rerank()
rag.answer(question, options=None)
```

---

## Clean Rebuild

Recreate both indexes:

```bash
python setup_indices.py --recreate
```

Rebuild the dense index from chunk 0:

```bash
python build_index.py --restart-dense
```

---

## Repository Structure

| File / Directory | Purpose |
|---|---|
| `phase2_pair2_gcg_smoothllm/` | GCG attack, SmoothLLM defense, evaluation harness, data, and experimental results |
| `config.yaml` | Central configuration for pipeline parameters |
| `common.py` | Configuration loader, tokenizer, embedding client, Qdrant/Elasticsearch clients |
| `setup_indices.py` | Creates Qdrant collection and Elasticsearch index |
| `build_corpus.py` | Processes textbooks and creates tokenizer-aware chunks |
| `build_index.py` | Builds dense and sparse indexes with checkpoint/resume support |
| `retrieval.py` | Hybrid search, RRF fusion, and MedCPT reranking |
| `smoke_test.py` | Runs a real hybrid retrieval test |
| `verify_index.py` | Validates index consistency and invariants |
| `make_manifest.py` | Generates the index manifest and `chunks_hash` |
| `readiness_test.py` | Evaluates RAG pipeline readiness |
| `rag.py` | Online RAG question-answering interface |
| `ask.py` | CLI for open-ended questions |
| `eval.py` | MedQA multiple-choice accuracy evaluation |
| `demo/` | Demonstration resources |
| `docs/` | Supporting documentation |

---

## Important System Invariants

### MedCPT Encoder Separation

Document chunks must use the **Article Encoder**, while online questions must use the **Query Encoder**.

Using the wrong encoder may not produce an explicit error but can silently degrade retrieval quality.

### Shared `chunk_id`

`chunk_id` is the key used to combine dense and sparse retrieval results.

The following identifiers must remain consistent:

```text
Qdrant payload chunk_id
        ==
Elasticsearch _id / chunk_id
```

Elasticsearch maps `chunk_id` as a `keyword` to preserve exact matching.

### Exact Dense Search

HNSW is disabled (`m=0`) to provide deterministic exact search for the experimental setup.

### Corpus Consistency

All indexes are built from the same:

```text
chunks.json
```

The corpus is validated through:

```text
manifest.json → chunks_hash
```

Different corpus hashes across environments indicate incompatible indexes and invalidate direct ablation comparisons.

---

## Confirmed Retrieval Configuration

| Parameter | Configuration |
|---|---|
| Embedding API | Configurable remote MedCPT embedding endpoint |
| Article embedding batch size | 8 |
| Vector dimension | 768 |
| Vector normalization | `false` |
| Similarity | Dot product |
| Medical subword ratio | ~1.41–1.48 tokens/word |
| Maximum chunk length | 512 tokens |
| Dense search | Exact (HNSW disabled) |
| Sparse retrieval | Elasticsearch BM25 |
| Fusion | Reciprocal Rank Fusion |
| Reranking | MedCPT |
| Final retrieval context | Top-5 chunks |

---

## Security and Reproducibility Notes

- API credentials are **not committed** to the repository.
- LLM API keys are stored locally in a gitignored file.
- External service endpoints are configurable rather than hardcoded in the public documentation.
- Index consistency is verified through corpus hashes and automated validation.
- Attack and defense experiments are isolated under `phase2_pair2_gcg_smoothllm/`.
- Experimental results should be interpreted within the tested models, datasets, attack configurations, and defense parameters.

---

## Research Context

This project was developed as part of a university **Machine Learning Security** project.

The overall system combines:

- Medical multi-agent LLM architecture
- Hybrid RAG
- Dense and sparse retrieval
- MedCPT reranking
- LLM security evaluation
- Optimization-based jailbreak attacks
- Semantic jailbreak evaluation
- Perturbation-based defense
- Adversarial robustness benchmarking

The Phase 2 security research evaluates both **attack effectiveness and defense limitations**, rather than treating successful defense against a single attack configuration as evidence of general robustness.

---

## Attribution

This repository is a fork of the original team project:

**Upstream:** `tructt41/medqa-multi-agent`

The MedQA multi-agent system was developed collaboratively as a university project.

The **GCG/SmoothLLM security evaluation described under "AI Security Research Contribution" represents my contribution to the project**, including attack reproduction, defense evaluation, benchmarking, and analysis of residual weaknesses.
