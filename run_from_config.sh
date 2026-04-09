#!/usr/bin/env bash
# Reads an EditBench YAML config and runs the CanItEdit benchmark.
# Usage: ./run_from_config.sh configs/deepseek-r1-qwen3-8b-lora.yaml [--generate-only]
#
# The config is the single source of truth for model_path, port, temperature,
# top_p, max_tokens, batch_size, and LoRA settings.  CanItEdit-specific
# overrides (model_type, completion_limit) can live under a "canitedit:" key
# in the same YAML, or be passed as extra CLI flags which take precedence.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="/shared_workspace_mfs/aadi/Projects/EditBench_fork/configs"
NOTIFY_SCRIPT="/shared_workspace_mfs/aadi/Projects/notify_telegram.py"

if [[ $# -lt 1 ]]; then
    cat <<EOF
Usage: $(basename "$0") <config.yaml> [OPTIONS]

Required:
  <config.yaml>           EditBench YAML config (name or path)

Optional:
  --generate-only         Only generate completions, skip Docker eval
  --completion-limit N    Override completions per prompt
  --batch-size N          Override max concurrent requests
  --model-type TYPE       Override model type (editcoder, editcoder-1shot, agentpack, agentpack-1shot, chat)
  --run-name NAME         Override run directory name
  -h, --help              Show this help

Available configs:
EOF
    ls -1 "$CONFIG_DIR"/*.yaml 2>/dev/null | xargs -I{} basename {}
    exit 1
fi

CONFIG_FILE="$1"; shift

# Resolve config: allow bare name (e.g. "deepseek-r1-qwen3-8b-lora.yaml")
if [[ ! -f "$CONFIG_FILE" ]]; then
    if [[ -f "$CONFIG_DIR/$CONFIG_FILE" ]]; then
        CONFIG_FILE="$CONFIG_DIR/$CONFIG_FILE"
    elif [[ -f "$CONFIG_DIR/${CONFIG_FILE}.yaml" ]]; then
        CONFIG_FILE="$CONFIG_DIR/${CONFIG_FILE}.yaml"
    else
        echo "Error: Config not found: $CONFIG_FILE"
        exit 1
    fi
fi

# Activate conda for pyyaml. Hard-fail if neither env is available so we do
# not silently run under whatever interpreter happens to be on PATH.
eval "$(/shared_workspace_mfs/aadi/miniconda3/bin/conda shell.bash hook)"
if ! conda activate canitedit 2>/dev/null; then
    if ! conda activate SFT_env 2>/dev/null; then
        echo "Error: neither 'canitedit' nor 'SFT_env' conda env is available" >&2
        exit 1
    fi
fi

# Parse config into shell variables
eval "$(python3 -c "
import yaml, shlex, sys

with open('$CONFIG_FILE') as f:
    c = yaml.safe_load(f)

cie = c.get('canitedit', {}) or {}

# Model: LoRA adapter name vs model_path
enable_lora = c.get('enable_lora', False)
if enable_lora:
    # vLLM serves LoRA adapters under the alias from --lora-modules
    model_id = 'editbench_adapter'
else:
    model_id = c.get('model_name', c.get('model_path', c.get('model', '')))

port = c.get('port', 8000)
model_name = c.get('model_name', 'unknown')

# CanItEdit-specific (from canitedit: sub-key, with fallbacks)
model_type   = cie.get('model_type', 'chat')
comp_limit   = cie.get('completion_limit', 20)
batch_size   = cie.get('batch_size', 100)
temperature  = cie.get('temperature', c.get('temperature', 0.2))
top_p        = cie.get('top_p', c.get('top_p', 0.95))
max_tokens   = cie.get('max_tokens', c.get('max_tokens', 3072))
run_name     = cie.get('run_name', '')

print(f'CFG_MODEL_ID={shlex.quote(str(model_id))}')
print(f'CFG_PORT={port}')
print(f'CFG_MODEL_NAME={shlex.quote(str(model_name))}')
print(f'CFG_MODEL_TYPE={shlex.quote(str(model_type))}')
print(f'CFG_COMPLETION_LIMIT={comp_limit}')
print(f'CFG_BATCH_SIZE={batch_size}')
print(f'CFG_TEMPERATURE={temperature}')
print(f'CFG_TOP_P={top_p}')
print(f'CFG_MAX_TOKENS={max_tokens}')
print(f'CFG_RUN_NAME={shlex.quote(str(run_name))}')
")"

# CLI overrides take precedence over config
GENERATE_ONLY=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --generate-only) GENERATE_ONLY=true; shift ;;
        --completion-limit) CFG_COMPLETION_LIMIT="$2"; shift 2 ;;
        --batch-size) CFG_BATCH_SIZE="$2"; shift 2 ;;
        --model-type) CFG_MODEL_TYPE="$2"; shift 2 ;;
        --run-name) CFG_RUN_NAME="$2"; shift 2 ;;
        -h|--help) exec "$0" ;; # re-run with no args to show usage
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Build run name: config model_name + date
if [[ -z "$CFG_RUN_NAME" ]]; then
    CFG_RUN_NAME="${CFG_MODEL_NAME}-canitedit-$(date +%Y-%m-%d)"
fi

MODEL="openai/${CFG_MODEL_ID}"
BASE_URL="http://localhost:${CFG_PORT}/v1"
OUTPUT_DIR="${SCRIPT_DIR}/runs/${CFG_RUN_NAME}"
mkdir -p "$OUTPUT_DIR"
SUMMARY_PATH="${OUTPUT_DIR}/summary.json"
DETACHED_LOG="${OUTPUT_DIR}/detached_eval.log"

echo "=== CanItEdit Benchmark ==="
echo "Config:      $CONFIG_FILE"
echo "Model:       $MODEL"
echo "Model type:  $CFG_MODEL_TYPE"
echo "Run name:    $CFG_RUN_NAME"
echo "Output:      $OUTPUT_DIR"
echo "Port:        $CFG_PORT"
echo "Max tokens:  $CFG_MAX_TOKENS"
echo "Temperature: $CFG_TEMPERATURE"
echo "Top-p:       $CFG_TOP_P"
echo "Completions: $CFG_COMPLETION_LIMIT per prompt"
echo "Batch size:  $CFG_BATCH_SIZE"
echo ""

# Set base URL for vLLM
export OPENAI_API_BASE="$BASE_URL"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
echo "API base:    $BASE_URL"

# -------------------------------------------------------------------------
# Summary.json lifecycle helper.
#
# We write summary.json BEFORE generation starts (status=running_generation)
# so the master scheduler can always distinguish a currently-running job
# from a crashed one. An EXIT trap promotes it to generation_failed on any
# nonzero exit before the detached worker is launched.
#
# Writes are atomic via tmp+rename so master's refresh_result_indexes never
# observes a partially-written summary.
# -------------------------------------------------------------------------
SUMMARY_FINALIZED=0
write_summary_status() {
    local status="$1"
    local extra_args="${2:-}"
    python - "$SUMMARY_PATH" "$status" "$CFG_RUN_NAME" "$OUTPUT_DIR" "$CONFIG_FILE" "$CFG_MODEL_NAME" "$MODEL" "$BASE_URL" "$DETACHED_LOG" "$extra_args" <<'PY'
import errno, fcntl, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

(
    summary_path,
    status,
    run_name,
    run_dir,
    config_path,
    model_name,
    api_model,
    api_base,
    detached_log,
    extra_json,
) = sys.argv[1:]

summary_path = Path(summary_path)
summary_path.parent.mkdir(parents=True, exist_ok=True)
lock_path = summary_path.with_name(summary_path.name + ".lock")

with lock_path.open("a+", encoding="utf-8") as lock_handle:
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    except OSError as exc:
        if exc.errno not in (errno.ENOLCK, errno.ENOSYS):
            raise

    # Load existing summary so repeated writes merge instead of clobber.
    data = {}
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError:
            data = {}

    data.update({
        "benchmark": "CanItEdit",
        "run_name": run_name,
        "run_dir": run_dir,
        "config_path": config_path,
        "model_name": model_name,
        "api_model": api_model,
        "api_base": api_base,
        "status": status,
        "eval_log_path": detached_log,
    })
    now = datetime.now(timezone.utc).isoformat()
    if status == "running_generation":
        data.setdefault("generation_started_at_utc", now)
    elif status == "generation_complete":
        data["generation_completed_at_utc"] = now
    elif status == "generation_failed":
        data["generation_failed_at_utc"] = now

    if extra_json:
        try:
            data.update(json.loads(extra_json))
        except json.JSONDecodeError:
            pass

    tmp = summary_path.with_name(
        f".{summary_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(data, indent=2) + "\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    os.replace(tmp, summary_path)
PY
}

cleanup_on_exit() {
    local rc=$?
    if [[ $rc -ne 0 && $SUMMARY_FINALIZED -eq 0 ]]; then
        write_summary_status "generation_failed" "{\"error\": \"runner exited with code ${rc} before detached eval was launched\"}" || true
    fi
}
trap cleanup_on_exit EXIT

# Seed summary.json with the in-progress status so partial failures can be
# distinguished from missing runs.
write_summary_status "running_generation"

# Step 1: Generate completions
echo ">>> Step 1: Generating completions..."
python "${SCRIPT_DIR}/benchmark/generate_completions.py" \
    --model "$MODEL" \
    --model-type "$CFG_MODEL_TYPE" \
    --output-dir "$OUTPUT_DIR" \
    --completion-limit "$CFG_COMPLETION_LIMIT" \
    --batch-size "$CFG_BATCH_SIZE" \
    --temperature "$CFG_TEMPERATURE" \
    --top-p "$CFG_TOP_P" \
    --max-tokens "$CFG_MAX_TOKENS"

COMPLETION_COUNT=$(find "$OUTPUT_DIR" -name '*.json.gz' ! -name '*.results.json.gz' | wc -l)
echo ">>> Generated $COMPLETION_COUNT completion files"

write_summary_status "generation_complete" "{\"raw_file_count\": ${COMPLETION_COUNT}}"

if [[ "$GENERATE_ONLY" == true ]]; then
    echo ""
    echo "=== Generation complete (--generate-only). ==="
    echo "To run eval:  docker run --rm -v ${OUTPUT_DIR}:/results:rw ghcr.io/nuprl/canitedit --dir /results --output-dir /results"
    echo "To score:     python ${SCRIPT_DIR}/benchmark/pass_k.py ${OUTPUT_DIR}"
    SUMMARY_FINALIZED=1
    exit 0
fi

echo ""
echo ">>> Launching detached evaluation..."
# ``setsid`` moves the worker into its own session so it is never torn down
# together with the runner shell's process group. ``nohup`` is still used
# so SIGHUPs from any surviving parent are ignored.
setsid nohup python "${SCRIPT_DIR}/scripts/detached_eval_worker.py" \
    --run-dir "$OUTPUT_DIR" \
    --summary-path "$SUMMARY_PATH" \
    --notify-script "$NOTIFY_SCRIPT" \
    >"$DETACHED_LOG" 2>&1 </dev/null &
DETACHED_PID=$!
disown "$DETACHED_PID" 2>/dev/null || true

# Record the worker pid so master / ops tooling can check liveness later.
write_summary_status "generation_complete" "{\"raw_file_count\": ${COMPLETION_COUNT}, \"detached_worker_pid\": ${DETACHED_PID}}"
SUMMARY_FINALIZED=1

echo "=== Generation complete; detached eval running ==="
echo "Run dir:      $OUTPUT_DIR"
echo "Summary:      $SUMMARY_PATH"
echo "Eval log:     $DETACHED_LOG"
echo "Worker pid:   $DETACHED_PID"
