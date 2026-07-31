# agentenv-agentmemory

AgentMemoryGym integrates all four MemoryArena environment families behind one
HTTP/runtime contract and adds the same policy-facing memory tools without
exposing private answers. MemoryArena Formal Reasoning contains separate Math
and Physics data domains. Search, Travel, and both Formal data domains expose
separate paper-evaluation and fail-fast contracts. AMG keeps those nine frozen
MemoryArena surfaces and adds one non-paper procedural training surface:

| Source / contract | AMG surface |
| --- | --- |
| Web Shopping | `memoryarena_webshop_native_v1` |
| Programmatic memory training | `agentmemory_webshop_procedural_natural_chain_train_v1` |
| Travel Planner / fail-fast | `memoryarena_travel_planner_failfast_one_action_v3` |
| Travel Planner / paper eval | `memoryarena_travel_planner_paper_eval_one_action_v3` |
| Web Search / public221 paper eval | `memoryarena_progressive_search_paper_eval_public221_one_action_v3` |
| Web Search / public221 fail-fast | `memoryarena_progressive_search_failfast_public221_one_action_v3` |
| Formal Reasoning / Math fail-fast | `memoryarena_formal_reasoning_math_failfast_v3` |
| Formal Reasoning / Math paper eval | `memoryarena_formal_reasoning_math_paper_eval_one_action_v3` |
| Formal Reasoning / Physics fail-fast | `memoryarena_formal_reasoning_phys_failfast_v3` |
| Formal Reasoning / Physics paper eval | `memoryarena_formal_reasoning_phys_paper_eval_one_action_v3` |

Math and Physics are separate runnable/evaluation surfaces, not separate
upstream environment families. The paper reports them in separate columns and
forms its all-task average over Shopping, Travel, Search, Math, and Physics.
Only each domain's `paper_eval` contract contributes paper-style metrics;
fail-fast surfaces are named training variants.

The procedural surface isolates cross-session memory with a binary natural-
attribute chain. Each of six phases names an approved shortlist of exactly two
certified native products by their complete real catalog titles; the policy never
receives an ASIN in the task text. One product represents each of that phase's
two natural attribute values. The model searches the complete visible title,
opens the matching native result, and the rule verifier uses the hidden ASIN in
the purchase receipt. Certification scans the whole frozen catalog and keeps a
title only when Unicode-normalized, whitespace-normalized, case-folded matching
resolves to exactly one ASIN. Only the two displayed listings are eligible for
the current order. A same-colored or same-material product elsewhere in the
million-product catalog is not an approved substitute. Attribute uniqueness is
deliberately shortlist-scoped; only the complete visible title is catalog-wide
unique.

The first phase states a starting natural attribute. Each later phase provides
an independently generated two-row customer pairing table and requires the
attribute of the product actually bought in the immediately preceding session.
A paired task changes only the first request, keeps every later observation and
candidate order byte-identical, and flips all six correct ASINs. Any phase with
more or fewer than two approved candidates is rejected. The verifier enumerates
all `2^6=64` in-shortlist purchase paths per task, proves exactly one legal path,
and checks that the budget excludes none of them. Runtime rejects every
out-of-shortlist purchase, including another catalog item with the same natural
attribute, without revealing the answer. This surface requires neither human
review nor an LLM judge and is never paper-eligible.

The Web Shopping action surface is:

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

Shopping remains native WebShop: the task text shows natural product cards, the
model searches their complete titles, and native results expose clickable
listing identifiers as part of ordinary navigation. Product and option pages
are navigated with `click[...]`, and only `click[Buy Now]` commits a purchase.
The purchase receipt's ASIN is an internal verifier key, not a customer request.
There is no formal synthetic `SEARCH`, `BUY`, `ANSWER`, or `GROUND` action.

## Runtime contract

- Web Shopping keeps the original WebShop browser state machine. One episode is
  one complete six-session bundled-shopping chain.
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
- Travel delegates all six tools and the seven-slot online plan judge to the
  frozen upstream Travel Planner runtime. In the explicitly named `failfast`
  variant, a correct plan gives `+1` and advances once; a wrong plan gives `0`,
  does not advance, and terminates without answer or verifier-reason feedback.
  Earlier correct-plan rewards remain in the episode return. Exhausting a
  traveler's 30 native actions also gives `0`, does not advance, and terminates.
- The Travel `paper_eval` variant preserves upstream outer continuation: every
  submitted plan advances, including an incorrect plan, and a phase timeout
  records an empty incorrect plan before advancing. Correct plans give `+1`
  and incorrect plans give `0`. Once every traveler has a prediction, the
  terminal transition emits the official PS/SPS/SR contribution ledger from
  frozen upstream `eval.py`. Online reward checks seven slots including
  `current_city`; paper metrics check the official six-slot set without
  `current_city`. Fail-fast output is never eligible for the Travel paper column.
- Both Travel surfaces are explicitly `one_action_v3` adapter variants: one AMG
  policy turn executes exactly one native tool action or plan submission. The
  upstream agent may batch multiple tool calls in one of its 30 model turns,
  so neither adapter's 30-action traveler budget is native-agent turn-budget
  parity. Tool meaning and the online plan judge remain upstream behavior;
  phase advancement is selected by the explicit surface contract.
  Exact native batch parity would require a separate surface whose single
  policy turn carries an ordered list of tool calls, executes all calls before
  the next model sample, and counts that list as one of 30 model turns. The
  shared AMG rollout/evidence schema currently records one action per sample,
  so no batch-parity surface is registered or implied here.
- Progressive Search production surfaces accept only the frozen public
  `ZexueHe/memoryarena` `progressive_search` JSONL: 221 tasks and 1,641 phases.
  This public panel omits 35 tasks from the paper's 256-task panel and is never
  reported as a complete paper reproduction.
- `paper_eval_public221_one_action_v3` judges every phase, continues after incorrect
  intermediate answers, and emits zero online reward. Its terminal ledger
  supports the paper's task-macro PS, correctness-at-depth SR@k, and final-phase
  SR. `failfast_public221_one_action_v3` gives `+1` for a correct phase and
  advances once; a wrong phase gives `0` and terminates while preserving earlier
  rewards.
- Both Search surfaces are explicitly `one_action_v3`: one AMG policy turn
  executes one native action, submission, invalid attempt, or memory action.
  Upstream Progressive Search may execute multiple returned tool calls inside
  one model iteration, so AMG does not claim upstream batched model-turn parity.
  Each subquery independently allows 35 native or invalid action attempts and
  the final phase allows 30. Memory actions do not consume those per-phase
  native budgets.
  Separately, `max_total_actions` is enforced by the common AMG wrapper and
  counts native, memory, and invalid actions; legacy metadata `max_steps`
  reports the same total-action cap. A memory-heavy trajectory can therefore
  reach the AMG total cap before exhausting every phase's native allowance.
- Both Search contracts expose `search` and `get_document`. Search returns
  `k=5` snippets truncated to 512 tokens with the locally cached, revision-pinned
  `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`
  tokenizer. Before loading it offline, the runtime hashes the six resolved
  snapshot files and verifies bundle SHA256
  `62851e5e39395f893633e2283ace53d5b223896d0058c751fa086f81c7a4f187`.
  It refuses an unpinned tokenizer download. The adapter intentionally follows
  the paper protocol (`k=5` plus `get_document`); it does not inherit an
  inconsistent clean-upstream launcher default such as `k=10` or omission of
  `get_document`.
- Search embedding metadata records provider, model, a SHA256 of the normalized
  endpoint, a named route variant, and a public config SHA256. It never records
  the endpoint text or API key. Paper evaluation requires provider `openai`.
  The fail-fast surface may use the fixed OpenRouter route only under the
  explicit `failfast_openrouter_nonpaper_embedding_v1` metadata variant.
- `memoryarena_progressive_search_bm25_integration_smoke_public221_one_action_v3`
  is a separate integration-only surface. It reuses MemoryArena's upstream
  Pyserini `BM25Searcher` with an explicit local Lucene index and keeps the same
  one-action fail-fast phase state machine. Metadata fixes
  `search_backend.id=bm25_lucene_integration`, `integration_only=true`, and
  `paper_eligible=false`. This surface cannot produce the MemoryArena paper
  Search column and does not disable, relax, or substitute for the dense
  `text-embedding-3-small` provenance guard on the two canonical Search
  surfaces.
- Formal Math/Physics preserve the upstream questions, background, ordering,
  and equivalence judge. Their explicitly named AMG `failfast_v3` variant gives
  `+1` for each correct subtask, advances once, and terminates with reward `0`
  on the first wrong subtask while retaining earlier rewards. This termination
  rule is an AMG training variant, not the original continuing runner.
- Formal Math/Physics `paper_eval_one_action_v3` preserves the original runner's
  continuation semantics: every judged answer advances, including an incorrect
  answer. Each correct answer gives `+1`, each incorrect answer gives `0`, and
  terminal evidence records the complete ordered verdict sequence. Task PS is
  the fraction of correct questions; task success and the paper SR contribution
  are determined only by the final question, matching frozen upstream
  `formal_reasoning_env/eval.py`. Paper-eval refuses reward overlays.
- V3 adapters are reward-neutral by default for memory/invalid-action overlays.
  Any optional overlay must be explicit in the launch manifest and cannot be
  reported as the unmodified upstream reward contract.

## Web Shopping launch

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

## Programmatic memory training

First certify a balanced product pool against the exact native catalog,
attribute file, price table, and every Lucene index byte:

Certification keeps the policy-facing identity natural: each candidate is
shown by its complete native product title, while its ASIN remains hidden until
the normal WebShop `click[ASIN]` result action. Native search uses a deterministic
three-or-more-word contiguous phrase copied from that visible title, retains the
certified natural attribute (for example, `leather` or `vanilla`), and excludes
characters unsafe for `search[...]`. This supports long or bracketed catalog
titles without inventing a synthetic product ID. Candidates must pass native
first-page search, product-page open, and exact purchase-receipt checks before
they are assigned evenly and ASIN-disjointly to train, dev, and test.

```bash
python AgentGym/agentenv-agentmemory/scripts/audits/certify_procedural_memory_product_pool.py \
  --memoryarena-root /path/to/frozen/MemoryArena \
  --items-file /path/to/items_shuffle.json \
  --attributes-file /path/to/items_ins_v2.json \
  --search-root /path/to/search_engine \
  --java-home /path/to/java-home \
  --lucene-index-manifest /path/to/original_lucene_index_files.sha256 \
  --expected-items-sha256 <sha256> \
  --expected-attributes-sha256 <sha256> \
  --expected-lucene-manifest-sha256 <sha256> \
  --expected-price-table-sha256 <sha256> \
  --output-pool /path/to/certified_pool_v3.json \
  --output-audit /path/to/certified_pool_v3.audit.json
```

Verify any finite task window before training. This example generates 10,000
tasks and machine-checks 640,000 complete purchase paths:

```bash
python AgentGym/agentenv-agentmemory/scripts/audits/verify_procedural_memory_dataset.py \
  --product-pool /path/to/certified_pool_v3.json \
  --product-pool-sha256 <pool-file-sha256> \
  --split train \
  --generator-seed 233 \
  --task-count 10000 \
  --output-manifest /path/to/train_10000.audit.json
```

The training server generates the verified stream on demand and refuses an
odd task count, an unpinned pool, or `split=all`. A training seed enumerates
the generator's complete collision-free semantic period before the stream
derives another seed. With the certified 4-product-per-cell pool, that gives
343,597,383,680 nonrepeating tasks before the first reseed; the deterministic
stream remains unbounded after that point, but does not claim semantic
uniqueness across complete seed epochs:

```bash
PYTHONPATH=AgentGym/agentenv-agentmemory:AgentGym/agentenv \
/path/to/python -m agentenv_agentmemory.launch \
  --surface agentmemory_webshop_procedural_natural_chain_train_v1 \
  --memoryarena-root /path/to/frozen/MemoryArena \
  --memoryarena-base-commit 6cd9de14b71915e39ac742a20dc33785e14b6aab \
  --items-file /path/to/items_shuffle.json \
  --attributes-file /path/to/items_ins_v2.json \
  --search-root /path/to/search_engine \
  --java-home /path/to/java-home \
  --lucene-index-manifest /path/to/original_lucene_index_files.sha256 \
  --procedural-product-pool /path/to/certified_pool_v3.json \
  --procedural-product-pool-sha256 <pool-file-sha256> \
  --procedural-task-count 10000 \
  --procedural-generator-seed 233 \
  --split train \
  --run-id <run-id> \
  --port 8000
```

Run `scripts/smoke/smoke_procedural_memory_webshop_native.py` with the same
pinned native inputs to exercise all six real search, product-page, purchase,
`ADD`, and later-session `RETRIEVE` transitions.

## Other environment launches

Every v3 launch also requires the frozen MemoryArena checkout and its exact
base commit:

```bash
COMMON="--memoryarena-root /path/to/frozen/MemoryArena \
  --memoryarena-base-commit 6cd9de14b71915e39ac742a20dc33785e14b6aab \
  --run-id <run-id>"
```

The Web Shopping runtime additionally fails closed unless the imported
MemoryArena WebShop source is pristine at that commit. Keep external task
instructions and the AMG reward ledger in `agentenv-agentmemory`; do not patch
the upstream `SimServer` constructor, reward function, Lucene import, or goal
module in the frozen checkout. Generate each new annotation gate against the
same pristine checkout. Run-specific launchers and gates from older experiments
remain historical evidence and are not reusable with a different source tree.

Travel Planner fail-fast training and paper evaluation:

```bash
python -m agentenv_agentmemory.launch $COMMON \
  --surface memoryarena_travel_planner_failfast_one_action_v3 \
  --travel-tasks-path /path/to/group_travel_planner.jsonl \
  --travel-database-path /path/to/assembled/travel/database \
  --port 8001

python -m agentenv_agentmemory.launch $COMMON \
  --surface memoryarena_travel_planner_paper_eval_one_action_v3 \
  --travel-tasks-path /path/to/group_travel_planner.jsonl \
  --travel-database-path /path/to/assembled/travel/database \
  --port 8002
```

The server binds the Travel JSONL byte-for-byte to HF config
`group_travel_planner` at the frozen revision and requires exactly 270 groups
and 1,869 traveler phases. Reset indices are 0-based dataset positions
`0..269`; upstream source IDs `1..270` are retained separately in evidence.
The database root must contain the frozen canonical files for flights,
restaurants, accommodations, attractions, distance matrix, and cities. All six
classes and their SHA256 values are attested before the tool executor starts.

Formal Reasoning uses the same adapter for both data domains, but each launch
must point at the separately frozen `formal_reasoning_math` or
`formal_reasoning_phys` dataset export. Run Math and Physics separately, and do
not mix a fail-fast diagnostic with the corresponding paper-eval result:

```bash
python -m agentenv_agentmemory.launch $COMMON \
  --surface memoryarena_formal_reasoning_math_failfast_v3 \
  --formal-reasoning-tasks-path /path/to/formal_reasoning_math.jsonl \
  --formal-reasoning-judge-model <judge-model> \
  --formal-reasoning-judge-base-url http://127.0.0.1:8100/v1 \
  --port 8002

python -m agentenv_agentmemory.launch $COMMON \
  --surface memoryarena_formal_reasoning_math_paper_eval_one_action_v3 \
  --formal-reasoning-tasks-path /path/to/formal_reasoning_math.jsonl \
  --formal-reasoning-judge-model <judge-model> \
  --formal-reasoning-judge-base-url http://127.0.0.1:8100/v1 \
  --port 8003

python -m agentenv_agentmemory.launch $COMMON \
  --surface memoryarena_formal_reasoning_phys_failfast_v3 \
  --formal-reasoning-tasks-path /path/to/formal_reasoning_phys.jsonl \
  --formal-reasoning-judge-model <judge-model> \
  --formal-reasoning-judge-base-url http://127.0.0.1:8100/v1 \
  --port 8004

python -m agentenv_agentmemory.launch $COMMON \
  --surface memoryarena_formal_reasoning_phys_paper_eval_one_action_v3 \
  --formal-reasoning-tasks-path /path/to/formal_reasoning_phys.jsonl \
  --formal-reasoning-judge-model <judge-model> \
  --formal-reasoning-judge-base-url http://127.0.0.1:8100/v1 \
  --port 8005
```

Progressive Search requires the frozen public221 dataset, four exact FAISS
indexes and ID maps, the seven exact parquet source shards, the deterministic
100,195-row materialized corpus, and an LLM judge. The corpus JSONL canonical
SHA256 is `6b306573f6194367d5e2a7daaae12d9cb4242409413f261ea6d81a19d7cf4b26`;
the runtime hashes the actual parquet and JSONL bytes rather than trusting the
manifest's claimed values. Manifest paths are relative to its own directory so
the same shared assets remain verifiable under `/home/...` and `/media/cfs/...`
mounts. `OPENAI_API_KEY` is read from the process environment
and is never written to metadata:

```bash
python -m agentenv_agentmemory.launch $COMMON \
  --surface memoryarena_progressive_search_paper_eval_public221_one_action_v3 \
  --browsecomp-tasks-path /path/to/progressive_search/data.jsonl \
  --browsecomp-index-path '/path/to/index/shard*.index' \
  --browsecomp-corpus-path /path/to/corpus.jsonl \
  --browsecomp-corpus-manifest /path/to/corpus-manifest.json \
  --browsecomp-embedding-provider openai \
  --browsecomp-embedding-model text-embedding-3-small \
  --browsecomp-judge-model <judge-model> \
  --browsecomp-judge-max-tokens 8000 \
  --browsecomp-api-base-url <openai-compatible-base-url> \
  --port 8004
```

Use `memoryarena_progressive_search_failfast_public221_one_action_v3` only for
the explicit fail-fast training variant. Paper evaluation refuses nonzero
memory or invalid-action reward overlays and refuses the OpenRouter embedding
provider. Judge and embedding metadata record public provider/model/config
fields and endpoint SHA256 values; credentials and endpoint text are not
emitted. Search, embedding, judge-transport, or judge-parse failures terminate
immediately as excluded infrastructure samples without adding a phase verdict.

For adapter/PPO integration QA when an exact dense embedding route is
unavailable, launch the separately named BM25 surface. Pyserini and Java are
runtime dependencies; the Lucene index belongs in the workspace data/runtime
layer, not this source repository:

```bash
python -m agentenv_agentmemory.launch $COMMON \
  --surface memoryarena_progressive_search_bm25_integration_smoke_public221_one_action_v3 \
  --browsecomp-tasks-path /path/to/progressive_search/data.jsonl \
  --browsecomp-bm25-index-path /path/to/browsecomp/lucene_index \
  --browsecomp-judge-model <judge-model> \
  --browsecomp-judge-max-tokens 8000 \
  --browsecomp-api-base-url <openai-compatible-base-url> \
  --port 8005
```

## Tests

Mac contract tests do not import the heavy MemoryArena runtime:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=AgentGym/agentenv-agentmemory:AgentGym/agentenv \
python3 -B -m unittest discover \
  -s AgentGym/agentenv-agentmemory/tests -v
```

The adapter and real environment smokes require the frozen upstream checkout
and each domain's native assets. Web Shopping additionally needs PyLucene and
the full product/index files; Progressive Search needs its matching embedding service;
Travel needs the assembled six-file database; Formal Reasoning needs its judge
endpoint. Fake dependencies verify parser and
state-machine logic only and cannot replace a native smoke or produce model,
reward, algorithm, curriculum, throughput, or capability evidence.

## Repository boundary

This package contains reusable adapters for all four MemoryArena environment
families and their contract tests. Converted datasets, frozen launch inputs,
indexes, experiment manifests, runtime evidence, and one-off analysis programs
belong in the external AgentMemoryGym workspace rather than this source
repository.

The retired SQLite/FTS shopping surrogate is not shipped here. The native
server rejects `AGENTMEMORY_CATALOG_INDEX_PATH` and
`AGENTMEMORY_SEARCH_TIMEOUT_MS` so an old launcher cannot silently restore it.

Maintained operational entrypoints are grouped by responsibility under
`scripts/audits/`, `scripts/services/`, and `scripts/smoke/`.
