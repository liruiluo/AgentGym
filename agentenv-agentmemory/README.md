# agentenv-agentmemory

AgentMemoryGym integrates the original MemoryArena WebShop environment with six
policy-facing memory tools. The formal action surface is:

```text
search[keywords]
click[current clickable value]
ADD {"key": "...", "value": "..."}
UPDATE {"memory_id": "mem_0000", "value": "..."}
DELETE {"memory_id": "mem_0000"}
RETRIEVE {"query": "...", "top_k": 3}
SUMMARY {"text": "...", "source_ids": ["S0", "C0"]}
FILTER {"keep_ids": ["C0"], "scope": "active"}
```

Shopping remains native WebShop: search results expose real ASINs, product and
option pages are navigated with `click[...]`, and only `click[Buy Now]` commits
a purchase. There is no formal synthetic `SEARCH`, `BUY`, `ANSWER`, or
`GROUND` action.

## Runtime contract

- One episode is one complete six-session bundled-shopping chain.
- A correct purchase advances immediately without requiring `ADD` or another
  memory action first.
- A wrong ASIN or cumulative budget overflow gives `-0.01` and terminates the
  chain without verifier feedback or retry.
- Each correct non-final purchase gives `+1.0`; the sixth also receives the
  final `+1.0` bundle bonus.
- Native browser state, S* trace, and active C* context clear at a session
  boundary. Policy-authored long-term memory persists but remains hidden until
  `RETRIEVE`.
- `ADD` and `UPDATE` preserve policy-authored text verbatim.
- Prices and purchases come from structured native WebShop state and are
  accumulated in integer cents. HTML parsing is not authoritative.

## Formal launch

The server refuses an implicit or SQLite fallback. All native paths are
required, and Uvicorn runs one worker so the 1.1M-product catalog and Lucene
searcher are loaded once and shared by isolated rollout sessions.

```bash
PYTHONPATH=AgentGym/agentenv-agentmemory:AgentGym/agentenv \
/path/to/workspace/runtime/venvs/webshop-py310/bin/python -m agentenv_agentmemory.launch \
  --surface memoryarena_webshop_native_v1 \
  --memoryarena-root /path/to/frozen/MemoryArena \
  --raw-data /path/to/bundled_shopping/data.jsonl \
  --items-file /path/to/items_shuffle.json \
  --attributes-file /path/to/items_ins_v2.json \
  --search-root /path/to/search_engine \
  --java-home /path/to/workspace/runtime/jre11 \
  --domain-data-path /path/to/domain_data.json \
  --lucene-index-manifest /path/to/original_lucene_index_files.sha256 \
  --annotation-audit-summary /path/to/summary.json \
  --annotation-audit-chains /path/to/chains.jsonl \
  --annotation-manual-evidence /path/to/manual_candidate_evidence.json \
  --memoryarena-base-commit 6cd9de14b71915e39ac742a20dc33785e14b6aab \
  --run-id <run-id> \
  --split train \
  --price-seed 233 \
  --annotation-gate-mode provisional \
  --annotation-gate-manifest /path/to/annotation_gate.json \
  --annotation-gate-manifest-sha256 <externally-pinned-sha256> \
  --port 8000
```

`--search-root` is the parent `search_engine/` directory; the original engine
appends `indexes-full` itself. Passing the index leaf is rejected.

Use `--annotation-gate-mode strict --annotation-gate-manifest ...` only after
the complete six-step chains used by that split are listed as passed. A
provisional run must bind the current audit manifest and cannot support a final
capability claim.

`--annotation-gate-mode trust_all` preserves every canonical audit verdict in
the manifest while allowing `pass`, `unknown`, `fail`, and
`semantic_ambiguity` chains. Use it only when the run contract explicitly
treats the upstream annotations as authoritative; it does not relabel those
audit verdicts as passed.

## Tests

Mac contract tests do not import the heavy MemoryArena runtime:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=AgentGym/agentenv-agentmemory:AgentGym/agentenv \
python3 -B -m unittest discover \
  -s AgentGym/agentenv-agentmemory/tests -v
```

The adapter and full native smoke require the 9N runtime with torch, PyLucene,
the product files, and the Lucene index. A fake backend may verify parser and
state-machine logic, but it cannot replace a native smoke or produce model,
reward, algorithm, curriculum, throughput, or capability evidence.

## Repository boundary

This package contains only the native MemoryArena WebShop runtime and its
contract tests. Converted datasets, frozen launch inputs, experiment manifests,
runtime evidence, and one-off analysis programs belong in the external
AgentMemoryGym workspace rather than this source repository.

The retired SQLite/FTS shopping surrogate is not shipped here. The native
server rejects `AGENTMEMORY_CATALOG_INDEX_PATH` and
`AGENTMEMORY_SEARCH_TIMEOUT_MS` so an old launcher cannot silently restore it.

Maintained operational entrypoints are grouped by responsibility under
`scripts/audits/`, `scripts/services/`, and `scripts/smoke/`.
