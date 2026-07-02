# agentenv-agentmemory

Minimal AgentMemoryGym environment package for the `feat/agentmemory-env-v0` branch.

Current scope:

- Implements a small **bundled web shopping** memory task.
- Loads the default smoke items from `agentenv_agentmemory/data/bundled_shopping_smoke.jsonl`.
- Keeps minimal split files under `agentenv_agentmemory/data/splits/{train,dev,test}.txt`.
- Supports dataset selection through `AGENTMEMORY_DATA_PATH`, `AGENTMEMORY_SPLIT`, and `AGENTMEMORY_SPLIT_DIR`.
- Exposes `/metadata` with `task_count`, `task_ids`, `splits`, and `source`.
- Exposes LTM tools: `ADD`, `UPDATE`, `DELETE`.
- Exposes STM tools: `RETRIEVE`, `SUMMARY`, `FILTER`.
- Exposes task actions: `BUY`, `ANSWER`, and optional product-catalog `SEARCH` when `AGENTMEMORY_CATALOG_INDEX_PATH` is configured.
- Automatically renders current-session STM as an action/tool-result trace and clears it when a successful `BUY` starts the next shopping session.
- Records `memory_state_diff`, `progress_score`, `compatibility_violations`, `tool_ops`, memory-only `memory_ops`, `session_trace`, and hidden purchase history in `info`.

This is still a skeleton/smoke environment. It now includes a MemoryArena bundled-shopping converter, catalog / ASIN resolver, strict candidate-metadata enrichment, product-catalog `SEARCH`, a scripted SEARCH baseline, and a failure-audit helper. Formal target freeze exists (`120/15/15`, `asin_catalog=900`, `ambiguous=0`), Qwen3-4B single-GPU smoke has run, and scripted SEARCH diagnostics now include memory ablations: no-memory `0/15`, full-context `6/15`, memory-tool strict no-retry `6/15`, retry5 semantic matcher fixed `13/15`, and soft-fallback verifier `15/15`. These are interface/solvability diagnostics only and still do **not** claim RL improvement.

The current AgentGym-RL vLLM rollout now has a fail-fast guard for `task_name=agentmemory`: formal rollout is blocked unless raw-history leakage is explicitly allowed for a diagnostic smoke run. In the normal path, `latest-observation` means the current environment observation, including current-session STM if the environment renders it; it does not mean previous-session raw history is preserved.

## Data schema

Default smoke tasks are JSONL records:

```json
{
  "task_id": "tv_bundle_75",
  "split": "train",
  "source": "memoryarena_webshop_style_handcrafted_v0",
  "difficulty": "smoke_dependency_distance_2",
  "memory_dependency": "tv_size_weight_vesa_reused_across_sessions",
  "title": "...",
  "subtasks": [
    {
      "instruction": "...",
      "target_product_id": "tv_b",
      "candidate_products": [
        {"product_id": "tv_b", "title": "Nebula 4K TV", "attributes": {"tv_size_in": 75}}
      ]
    }
  ]
}
```

This is the placeholder schema for later MemoryArena/WebShop-style conversion.

## Splits and validation

The smoke package currently has one item per split:

- `train`: `tv_bundle_75`
- `dev`: `laptop_bundle_14`
- `test`: `monitor_bundle_27`

Validate the local data package with:

```bash
PYTHONPATH=agentenv-agentmemory python3 agentenv-agentmemory/scripts/validate_agentmemory_data.py
```

Load one split explicitly:

```bash
AGENTMEMORY_SPLIT=dev agentmemory --host 0.0.0.0 --port 8000
```

The server metadata endpoint then reports only the selected split's task ids.

## Action format

Use ReAct with one action line, for example:

```text
Thought:
I should store the TV size for later compatibility checks.

Action:
ADD {"key": "tv_size", "value": "The purchased TV is 75 inches."}
```

Other examples:

```text
Action:
RETRIEVE {"query": "tv size", "top_k": 2}

Action:
BUY {"product_id": "mount_b"}
```

Observation memory sections:

- `Current session short-term history`: automatic current-session action/tool-result trace. It clears when a successful `BUY` advances to a new shopping session.
- `Active retrieved/summary context`: explicit context produced by `RETRIEVE`, `SUMMARY`, and `FILTER`. Long-term memory remains hidden until retrieved.

Visible context items are rendered with stable IDs inside the current
observation: `S0`, `S1`, ... for current-session STM trace entries and `C0`,
`C1`, ... for active retrieved/summary context. The clean RL path for
context-control tools is policy-authored:

- `SUMMARY {"text": "...", "source_ids": ["S0", "C0"]}` lets the current
  policy model write the summary tokens; the environment only validates optional
  visible source IDs and replaces active context with that summary.
- `FILTER {"keep_ids": ["C0"], "scope": "active"}` or
  `FILTER {"drop_ids": ["S0"], "scope": "session"}` lets the policy choose
  exactly which visible context IDs to keep/drop.

Deterministic scaffolds remain available for smoke tests and baselines:
`SUMMARY {"span": "session"}` and `FILTER {"query": "...", "scope": "active"}`.
They do not call an external LLM or hidden judge.

`SEARCH` is a product-catalog tool, not a memory tool: it appears in
`info["tool_ops"]` but not in `info["memory_ops"]`.

## Direct smoke

```bash
PYTHONPATH=agentenv-agentmemory python3 - <<'PY'
from agentenv_agentmemory.environment import AgentMemoryEnv

env = AgentMemoryEnv()
obs, info = env.reset(data_idx=0)
print(obs)
print(env.step('BUY {"product_id": "tv_b"}')[0])
print(env.step('ADD {"key": "tv_size", "value": "The TV is 75 inches."}')[0])
print(env.step('RETRIEVE {"query": "tv size", "top_k": 1}')[0])
print(env.step('BUY {"product_id": "mount_b"}')[0])
PY
```

Or run the packaged smoke helper:

```bash
PYTHONPATH=agentenv-agentmemory python3 agentenv-agentmemory/scripts/smoke_agentmemory.py
```

Loader smoke:

```bash
PYTHONPATH=agentenv-agentmemory python3 - <<'PY'
from agentenv_agentmemory.environment import default_smoke_data_path, load_tasks_from_jsonl
tasks = load_tasks_from_jsonl(default_smoke_data_path())
print("JSONL_LOADER_SMOKE_OK", len(tasks))
PY
```

## MemoryArena bundled-shopping converter

Convert public MemoryArena `bundled_shopping` JSONL into the AgentMemoryGym JSONL schema without committing the dataset into this repo:

```bash
PYTHONPATH=agentenv-agentmemory \
  python3 agentenv-agentmemory/scripts/convert_memoryarena_bundled_shopping.py \
  --input https://huggingface.co/datasets/ZexueHe/memoryarena/resolve/main/bundled_shopping/data.jsonl \
  --output /tmp/agentmemorygym-memoryarena/bundled_shopping.jsonl \
  --split-dir /tmp/agentmemorygym-memoryarena/splits \
  --report /tmp/agentmemorygym-memoryarena/target_match_report.jsonl
```

If a local MemoryArena product DB mirror is available, pass product catalog
shards or the product DB root to resolve target ASINs before falling back to
attribute-overlap matching:

```bash
PYTHONPATH=agentenv-agentmemory \
  python3 agentenv-agentmemory/scripts/convert_memoryarena_bundled_shopping.py \
  --input /path/to/bundled_shopping/data.jsonl \
  --output /tmp/agentmemorygym-memoryarena/bundled_shopping.jsonl \
  --split-dir /tmp/agentmemorygym-memoryarena/splits \
  --report /tmp/agentmemorygym-memoryarena/target_match_report.jsonl \
  --catalog-path /path/to/memoryarena-product-db/product_catalog/electronics_accessories_supplies.json \
  --catalog-path /path/to/memoryarena-product-db/product_catalog/electronics_television_video.json \
  --catalog-path /path/to/memoryarena-product-db/product_catalog/grocery_gourmet_food_pantry_staples.json \
  --catalog-path /path/to/memoryarena-product-db/product_catalog/grocery_gourmet_food_snacks_sweets.json
```

Current verified full mirror on the Jingyan shared disk:

```text
/media/cfs/ai-jingyan-train/luolirui.1/post-train/data/memoryarena-product-db/
135 files / 13,517,161,526 bytes; extra/missing/mismatch/part = 0
```

Keep the large product DB on the shared disk rather than copying it to the
devbox/local development disk. Shared-disk capacity is not the blocker here:
full mirroring and full indexing are allowed and intended on the shared disk.
Do not downscope this path to a "local minimal dependency" workaround merely
because the Mac/devbox is a 0-card development machine or lacks a local DB copy.
For a formal freeze, prefer the helper below: it first scans the full product
DB by target ASIN to select only relevant catalog shards, then calls the
converter and validator and writes a manifest.

```bash
PYTHONPATH=agentenv-agentmemory \
  python3 agentenv-agentmemory/scripts/freeze_memoryarena_bundled_shopping.py \
  --input /path/to/bundled_shopping/data.jsonl \
  --output-dir /path/to/freeze-run \
  --product-db-root /media/cfs/ai-jingyan-train/luolirui.1/post-train/data/memoryarena-product-db
```

Smoke the converter on the bundled synthetic fixture:

```bash
PYTHONPATH=agentenv-agentmemory python3 agentenv-agentmemory/scripts/smoke_memoryarena_converter.py
```

The converter writes a target-match audit report because MemoryArena answers expose target ASIN/attributes while prompts expose natural-language option descriptions. Without a catalog, full public bundled-shopping conversion has 12/900 tied heuristic matches. With the four currently relevant product-catalog shards above on Jingyan shared disk, the same public conversion validates with 0/900 ambiguous matches (`catalog=450`, `fallback=450`); this is still data-conversion evidence, not an RL result.


## Product-catalog SEARCH index

`SEARCH` is configured through `AGENTMEMORY_CATALOG_INDEX_PATH` or the
`AgentMemoryEnv(..., catalog_index_path=...)` constructor argument. It returns
public product metadata only: title, average rating, price, review count, and a
match score. It must not expose ASIN/source path/target labels in observation.

Build a full SQLite/FTS index from the MemoryArena product DB on the shared disk:

```bash
PYTHONPATH=agentenv-agentmemory \
  python3 agentenv-agentmemory/scripts/build_memoryarena_catalog_search_index.py \
  --product-db-root /media/cfs/ai-jingyan-train/luolirui.1/post-train/data/memoryarena-product-db \
  --output /media/cfs/ai-jingyan-train/luolirui.1/post-train/data/memoryarena-product-db/agentmemory_catalog_search.sqlite
```

Current Jingyan shared-disk index marker:

```text
AGENTMEMORY_CATALOG_SEARCH_INDEX_OK products=1031654
index size ~= 479M
```

Example action:

```text
SEARCH {"query":"A gluten-free carrot cake mix with easy-to-use instructions and vegan-friendly ingredients.","top_k":3}
```


## Scripted SEARCH baseline and failure audit

Strict baseline, retry diagnostic, and soft-fallback verifier diagnostic share
the same runner:

```bash
PYTHONPATH=agentenv-agentmemory \
  python3 agentenv-agentmemory/scripts/run_scripted_search_baseline.py \
  --data /path/to/memoryarena_agentmemory.jsonl \
  --split-dir /path/to/splits \
  --split dev \
  --catalog-index /media/cfs/ai-jingyan-train/luolirui.1/post-train/data/memoryarena-product-db/agentmemory_catalog_search.sqlite \
  --output-dir /path/to/evidence/run \
  --max-buy-attempts 1
```

Use `--include-target-audit` only for saved audit fields; the runner does not
use target ids for action selection. Use `--compatibility-fallback
ranked-all-after-compatible` only as an explicit verifier-feedback diagnostic:
it tries strict compatibility matches first and then other visible candidates
ranked by the same public SEARCH metadata.

The runner also supports policy-surface diagnostics:

```bash
--policy-mode scripted-search-memory  # default; uses ADD/RETRIEVE memory tools
--policy-mode search-full-context     # keeps prior accepted purchases in runner context, no memory tools
--policy-mode search-no-memory        # ignores prior purchases and uses no memory tools
```

Current semanticfix5 dev diagnostics:

```text
search-no-memory:        0/15, mean_progress=0.1889, add/retrieve=0/0
search-full-context:     6/15, mean_progress=0.5778, add/retrieve=0/0
scripted-search-memory:  6/15 strict no-retry; 13/15 retry5; 15/15 soft-fallback verifier diagnostic
```

Only `scripted-search-memory` exercises the explicit memory tool surface. The
full-context mode is a diagnostic upper/lower-bound comparison, not a learned
memory policy.

Analyze failed steps with:

```bash
python3 agentenv-agentmemory/scripts/analyze_scripted_search_failures.py \
  --run-dir /path/to/evidence/run \
  --output-dir /path/to/evidence/failure-audit
```

The analyzer prints `AGENTMEMORY_SCRIPTED_SEARCH_FAILURE_AUDIT_OK` and classifies
residual failures such as `compatibility_filter_excluded_target`.

## Server

After installing package dependencies:

```bash
agentmemory --host 0.0.0.0 --port 8000
```
