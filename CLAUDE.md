# CanItEdit Benchmark - Agent Instructions

This project runs the [CanItEdit](https://github.com/nuprl/CanItEdit) code-editing benchmark against models served by vLLM.

## Quick Reference

- **Configs**: `/shared_workspace_mfs/aadi/Projects/EditBench_fork/configs/*.yaml`
- **vLLM serve script**: `/shared_workspace_mfs/aadi/Projects/EditBench_fork/serve.sh`
- **Benchmark runner**: `./run_from_config.sh <config.yaml> [OPTIONS]`
- **Results**: `./runs/<model_name>-canitedit-<date>/`
- **Conda envs**: `canitedit` (preferred), `SFT_env` (fallback)
- **Telegram notifications**: `python /shared_workspace_mfs/aadi/Projects/notify_telegram.py "<message>"`

## Running a Benchmark (Agent Procedure)

When the user asks you to run a CanItEdit benchmark, follow this exact procedure:

### Prerequisites

The user will provide:
1. A vLLM server that is already running (port, model name, config path)
2. The config YAML path (in EditBench_fork/configs/)

You do NOT need to start the vLLM server yourself -- the user or another agent handles that.

### Step 1: Verify Server Health

```bash
curl -s http://localhost:<PORT>/health
```

If this returns empty or 200, the server is ready. If it fails, tell the user.

### Step 2: Read the Config

Read the YAML config to understand:
- `port` -- must match the running server
- `enable_lora` -- if true, the model name for API calls is `editbench_adapter` (handled automatically by `run_from_config.sh`)
- `model_name` -- used for naming the output directory

### Step 3: Launch the Benchmark (Background)

```bash
nohup bash run_from_config.sh <config_path> --batch-size 300 \
  > /shared_workspace_mfs/aadi/Projects/CanItEdit/logs/<model_name>-canitedit.log 2>&1 &
echo "PID: $!"
```

**Critical**: Always use `--batch-size 300` for LoRA models. The default (100) is too slow. For non-LoRA base models, 300 also works fine -- vLLM handles the concurrency.

Record the PID for monitoring.

### Step 4: Verify It Started

```bash
sleep 5 && tail -30 /shared_workspace_mfs/aadi/Projects/CanItEdit/logs/<model_name>-canitedit.log
```

Confirm you see the config summary and progress bar starting.

### Step 5: Set Up Monitoring (CronCreate)

Create a cron job that fires every 10 minutes to check progress:

```
Check the CanItEdit benchmark for <model_name>.
Run: tail -5 <log_path>
Check PID: ps -p <PID> -o pid,etime,args --no-headers
Count completions: find <output_dir> -name '*.json.gz' ! -name '*.results.json.gz' | wc -l
Report status briefly.
If process is done and log shows "Benchmark complete", send Telegram notification and delete this cron job.
```

The cron prompt should include the Telegram command:
```bash
python /shared_workspace_mfs/aadi/Projects/notify_telegram.py "<message>"
```

### Step 6: When Benchmark Completes

The pipeline runs 3 steps automatically:
1. **Generate completions** -- 20 completions x 210 items (105 problems x 2 instruction types)
2. **Docker eval** -- runs hidden tests + coverage via `ghcr.io/nuprl/canitedit`
3. **pass_k.py** -- computes pass@1, ExcessCode, MeanMedianCoverage

When your monitoring cron detects completion:
1. Read the final log output to get the scores
2. Send Telegram with results summary
3. Delete the cron job
4. Report to user

## Killing / Restarting a Run

If you need to kill and restart a benchmark (e.g., to change batch size):

**CRITICAL: Kill the entire process tree, not just the parent shell.**

```bash
# Find the child python process
pgrep -P <PARENT_PID>
# Kill parent AND children
kill <PARENT_PID> <CHILD_PID>
# Or kill the process group
kill -- -<PARENT_PID>
```

Killing only the parent bash shell leaves the child `generate_completions.py` running as an orphan with hundreds of open connections to vLLM. This wastes server resources and blocks the port.

**Completions resume automatically.** Existing `.json.gz` files are skipped on re-run, so restarting is safe and picks up where it left off.

## Comparing Results

After multiple runs are complete:
```bash
python benchmark/compare_runs.py
```

This runs `pass_k.py` across all completed run directories and outputs a comparison table.

## Gotchas

1. **LoRA model name**: When `enable_lora: true` in config, `run_from_config.sh` automatically uses `openai/editbench_adapter` as the model ID. You don't need to override this.

2. **Port conflicts**: If another vLLM server is on the same port, the benchmark silently talks to the wrong model. Always verify the port matches the intended server.

3. **Batch size matters for speed**: LoRA models at batch-size 100 run ~35s/item. At batch-size 300, they run ~1-3s/item. Always use 300.

4. **Docker required**: Step 2 (eval) needs Docker. If Docker is unavailable, use `--generate-only` and run eval later.

5. **Orphan processes**: See "Killing / Restarting a Run" above. This is the #1 mistake agents make.

6. **Generation time estimates**:
   - 3B models: ~1-2s/item (~7 min total)
   - 7B base: ~20s/item (~70 min total)
   - 7B LoRA (batch 300): ~45s/item (~2.5h total)
   - 14B base (batch 300): ~12 min total generation
   - 14B LoRA (batch 300): ~3-5 min generation, ~1.5h total with Docker eval
