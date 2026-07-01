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
- Exposes task actions: `BUY`, `ANSWER`.
- Records `memory_state_diff`, `progress_score`, `compatibility_violations`, `memory_ops`, and hidden purchase history in `info`.

This is still a skeleton/smoke environment. It now includes a MemoryArena bundled-shopping converter entrypoint, but it does **not** claim frozen formal MemoryArena data or RL improvement yet. The next real steps are to resolve converter target-match ambiguities with a WebShop catalog / ASIN map and run a real small-model/API rollout.

The current AgentGym-RL vLLM rollout now has a fail-fast guard for `task_name=agentmemory`: formal rollout is blocked unless raw-history leakage is explicitly allowed for a diagnostic smoke run.

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

Smoke the converter on the bundled synthetic fixture:

```bash
PYTHONPATH=agentenv-agentmemory python3 agentenv-agentmemory/scripts/smoke_memoryarena_converter.py
```

The converter writes a target-match audit report because MemoryArena answers expose target ASIN/attributes while prompts expose natural-language option descriptions. Full public bundled-shopping conversion currently has a small number of tied heuristic matches; resolve those with a WebShop catalog / ASIN map before treating the converted file as a frozen formal train/eval dataset.

## Server

After installing package dependencies:

```bash
agentmemory --host 0.0.0.0 --port 8000
```
