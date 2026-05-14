# Eval Instructions (detached CPU eval) — CanItEdit

This repo contains **generation outputs** from the same-size dirty-LoRA
experiment, committed by the k8s GPU pod. The judge / eval phase runs on
a separate CPU machine using these committed artifacts — generation is
not re-run.

## Critical: experiment isolation

The same model_name slugs (e.g. `qwen2.5-coder-7b-lora-dirty`) are
**shared** with the prior different-size experiment. The two are
distinguished ONLY by `summary.json:"config_path"`:

- **Same-size** (this experiment, what we want to eval):
  `config_path` starts with `/workspace/Master-Benchmarking-Orchestrator/`
- **Different-size** (older experiment):
  `config_path` starts with `/shared_workspace_mfs/.../Master_VLLM/`

Before running judges, verify every run dir. Note: `summary.json` is
gitignored in this repo so it will be absent on a fresh clone — the eval
worker recreates it if missing. If present:

```bash
for f in runs/*-canitedit-*/summary.json; do
  [ -f "$f" ] || continue
  cfg=$(python3 -c "import json; print(json.load(open('$f')).get('config_path',''))")
  [[ "$cfg" != /workspace/Master-Benchmarking-Orchestrator/* ]] && echo "WRONG-EXPERIMENT: $f -> $cfg"
done
```

Expect: empty output. If anything prints, you have a stale checkout or
mixed history — STOP and ask before evaluating.

## Which dirs are ready

Only dirs where `summary.json:"status" == "generation_complete"`. Anything
else (`running_generation`, `generation_failed`, `generation_pending`,
`eval_running`, `eval_failed`) is either partial or already mid-judge —
skip it. Cross-check by file count: a complete run has exactly 210
`.json.gz` files (105 problems × 2 instruction kinds).

## 3B-tier models

Regenerated cleanly on 2026-05-14 and now included. Eligible 3B slugs:
`qwen2.5-3b-lora-dirty`, `qwen2.5-coder-3b-instruct-lora-dirty`,
`qwen3-4b-base-lora-dirty`, `starcoder2-3b-lora-dirty`. (The earlier
contaminated dirs were removed in commit `2662666` before the rerun.)

`llama-3.2-3b-lora-dirty` has no generation dir on the GPU pod and is
not included; it may land later.

## Active runs

None — the previously-active `qwen3-14b-base-lora-dirty` CanItEdit dir
finished cleanly and is included as of this commit.

## Eval entrypoint: CanItEdit

Generation artifacts under `runs/<model>-canitedit-<UTC-ts>-<slot>-<rand>/`:

- `<problem-id>_<name>_instruction_<descriptive|lazy>.json.gz` — 210 files
  per complete run (105 problems × 2 instruction kinds)
- `summary.json` — written by the orchestrator at generation time
  (gitignored — recreated by the eval worker if missing on the CPU host)
- `summary.json.lock` — zero-byte filelock sentinel, ignored by eval

### Prerequisites on the eval machine

- Docker daemon running (pulls `ghcr.io/nuprl/canitedit`)
- The CanItEdit repo at `/workspace/CanItEdit` (paths in any pre-existing
  `summary.json` assume this)

### Run eval on one model

```bash
cd /workspace/CanItEdit
python3 scripts/detached_eval_worker.py \
    --run-dir runs/<run-name> \
    --summary-path runs/<run-name>/summary.json
```

This:
1. Stages `*.json.gz` files into `runs/<run-name>/_eval_input/` via
   symlinks (avoids the upstream container crashing on `summary.json`).
2. Runs `docker run --rm -v <run-dir>:/results:rw ghcr.io/nuprl/canitedit
   --dir /results/_eval_input --output-dir /results`.
3. Runs `benchmark/pass_k.py runs/<run-name>` to aggregate pass@1/10/100
   and ExcessCode.
4. Writes `*.results.json.gz` per problem and updates `summary.json` to
   `status=eval_complete`.

### Run dirs ready (as of this commit)

```
codellama-13b-hf-lora-dirty-canitedit-20260513T031925Z-slot0-537292142
codellama-7b-hf-lora-dirty-canitedit-20260511T032139Z-slot0-034511708
deepseek-coder-6.7b-base-lora-dirty-canitedit-20260512T124936Z-slot0-332578265
qwen2.5-coder-14b-instruct-lora-dirty-canitedit-20260513T083455Z-slot0-761658179
qwen2.5-coder-14b-lora-dirty-canitedit-20260513T031928Z-slot1-026071171
qwen2.5-coder-7b-instruct-lora-dirty-canitedit-20260511T235631Z-slot1-896008860
qwen2.5-coder-7b-lora-dirty-canitedit-20260511T235624Z-slot0-733111600
qwen3-8b-base-lora-dirty-canitedit-20260512T085306Z-slot1-364488892
starcoder2-15b-lora-dirty-canitedit-20260513T101137Z-slot0-217466480
starcoder2-7b-lora-dirty-canitedit-20260512T223528Z-slot1-821087404
qwen3-14b-base-lora-dirty-canitedit-20260513T150232Z-slot0-187889840
qwen2.5-3b-lora-dirty-canitedit-20260514T020244Z-slot1-247496836
qwen2.5-coder-3b-instruct-lora-dirty-canitedit-20260514T020218Z-slot0-116823438
qwen3-4b-base-lora-dirty-canitedit-20260514T022128Z-slot0-761479267
starcoder2-3b-lora-dirty-canitedit-20260514T031600Z-slot1-913929142
```
