# MemoryArena dataset randomization audit tools

These scripts audit the frozen upstream `bundled_shopping` rows before any
derived-data generator is implemented.

- `audit_dataset_structure.py`: schema, reuse graph, split leakage, semantic
  path concentration, and shortcut diagnostics.
- `audit_full_catalog_pools.py`: streaming scan of `items_shuffle.json` using
  canonical category-prefix matching and context-aware allow/deny labels.
- `summarize_catalog_pool_feasibility.py`: conservative path and
  counterfactual-pool summary from the two audit outputs.

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
