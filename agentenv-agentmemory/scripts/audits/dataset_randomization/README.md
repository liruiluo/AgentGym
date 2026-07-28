# MemoryArena dataset randomization audit tools

These scripts audit the frozen upstream `bundled_shopping` rows before any
derived-data generator is implemented.

- `audit_dataset_structure.py`: schema, reuse graph, split leakage, semantic
  path concentration, and shortcut diagnostics.
- `audit_full_catalog_pools.py`: streaming scan of `items_shuffle.json` using
  canonical category-prefix matching and context-aware allow/deny labels.
- `summarize_catalog_pool_feasibility.py`: conservative path and
  counterfactual-pool summary from the two audit outputs.
- `verify_candidate_order_randomization.py`: proof-carrying validation for the
  first presentation-only variant. It verifies that every answer, candidate
  line, and non-candidate byte remains frozen while only option order changes.

Run the 5.2 GB catalog scan on `cpu9n`, not on a training pod. Inputs must be
pinned by SHA256. Generated reports and copied datasets belong under the
workspace `audits/` tree, not in this source directory.

These tools produce candidate-pool evidence only. They do not certify a new
task. A derived bundle still requires exact candidate-ASIN identity, native
Lucene/page/BUY replay, unique preference optimum, split isolation, and human
review under the project data contract.

The upstream 150-row dataset remains an immutable human-reviewed evaluation
anchor. Any generated output must use a separately versioned
`MemoryArena-derived` name.

## Candidate-order pilot

`candidate_order_v1` is an online presentation transform, disabled by default.
It is intended to measure and remove fixed-position shortcuts before any
catalog-derived task is trusted. It does not create new semantic paths, new
targets, or independent examples, so it cannot by itself establish resistance
to memorizing the 150 upstream bundles.

The runtime records the base seed, environment/episode identity, complete
permutation list, source/rendered question hashes, and the invariant
`frozen_upstream_target_asins_unchanged`. Run the full frozen-file audit before
a model pilot:

```bash
PYTHONPATH=AgentGym/agentenv-agentmemory \
python AgentGym/agentenv-agentmemory/scripts/audits/dataset_randomization/verify_candidate_order_randomization.py \
  --raw-data /path/to/raw_memoryarena_bundled_shopping_data.jsonl \
  --base-seed 20260728 --replicas 8 --output /path/to/audit.json
```
