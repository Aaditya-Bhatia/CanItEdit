#!/usr/bin/env python3

import argparse
import errno
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

def _find_master_root() -> Path:
    projects = Path(__file__).resolve().parent.parent.parent
    for name in ("Master_VLLM", "Master-Benchmarking-Orchestrator"):
        candidate = projects / name
        if candidate.is_dir():
            return candidate
    return projects / "Master_VLLM"
MASTER_ROOT = Path(os.environ.get("MASTER_ROOT", str(_find_master_root())))
if str(MASTER_ROOT) not in sys.path:
    sys.path.append(str(MASTER_ROOT))

from benchmark_results import summarize_canitedit_run


def parse_args():
    parser = argparse.ArgumentParser(description="Detached CanItEdit eval worker.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--notify-script", default="")
    return parser.parse_args()


def update_summary(summary_path: Path, **updates):
    """Merge-update ``summary_path`` atomically under an advisory file lock.

    Writes go through a uniquely-named tmp file in the same directory so that
    concurrent readers (e.g. the master's ``refresh_result_indexes``) never
    observe a truncated summary.
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = summary_path.with_name(summary_path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            if exc.errno not in (errno.ENOLCK, errno.ENOSYS):
                raise
        data = {}
        if summary_path.exists():
            try:
                raw = summary_path.read_text(encoding="utf-8").strip()
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                data = {}
        data.update(updates)
        tmp_path = summary_path.with_name(
            f".{summary_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        with tmp_path.open("w", encoding="utf-8") as tmp_handle:
            json.dump(data, tmp_handle, indent=2)
            tmp_handle.write("\n")
            tmp_handle.flush()
            try:
                os.fsync(tmp_handle.fileno())
            except OSError:
                pass
        os.replace(tmp_path, summary_path)


def send_telegram(notify_script: str, message: str):
    if not notify_script:
        return
    path = Path(notify_script)
    if not path.exists():
        return
    subprocess.run([sys.executable, str(path), message], check=False)


def format_percent(value: object) -> str | None:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}%"
    return None


def benchmark_message(model_name: str, state: str, score: str | None = None) -> str:
    parts = ["CanItEdit", model_name, state]
    if score:
        parts.append(score)
    return " | ".join(parts)


def run(cmd: list[str], cwd: Path):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def prepare_eval_input_dir(run_dir: Path) -> Path:
    """Expose only raw generation files to the Docker evaluator.

    The upstream CanItEdit container walks every JSON-ish artifact under
    ``--dir``. Once we started writing ``summary.json`` into ``run_dir``, the
    container began trying to evaluate that file too and crashed with
    ``KeyError: 'completions'``. Stage only the raw ``*.json.gz`` inputs in a
    dedicated subdirectory and point Docker there.
    """
    eval_input_dir = run_dir / "_eval_input"
    if eval_input_dir.exists():
        shutil.rmtree(eval_input_dir)
    eval_input_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(
        path
        for path in run_dir.glob("*.json.gz")
        if not path.name.endswith(".results.json.gz")
    )
    if not raw_files:
        raise FileNotFoundError(f"no raw CanItEdit generation files found in {run_dir}")

    for source in raw_files:
        target = eval_input_dir / source.name
        # Relative symlinks keep the staging step cheap even for hundreds of
        # files and still work because the symlink target stays inside the
        # mounted run directory.
        target.symlink_to(Path("..") / source.name)
    return eval_input_dir


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    summary_path = Path(args.summary_path).resolve()
    repo_root = Path(__file__).resolve().parents[1]

    try:
        model_name = run_dir.name.split("-canitedit-", 1)[0]
        update_summary(
            summary_path,
            status="eval_running",
            eval_started_at_utc=datetime.now(timezone.utc).isoformat(),
            detached_worker_pid=os.getpid(),
            error=None,
            eval_failed_at_utc=None,
        )
        eval_input_dir = prepare_eval_input_dir(run_dir)
        run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{run_dir}:/results:rw",
                "ghcr.io/nuprl/canitedit",
                "--dir",
                f"/results/{eval_input_dir.name}",
                "--output-dir",
                "/results",
            ],
            cwd=repo_root,
        )
        run([sys.executable, str(repo_root / "benchmark" / "pass_k.py"), str(run_dir)], cwd=repo_root)
        result_count = sum(1 for _ in run_dir.glob("*.results.json.gz"))
        update_summary(
            summary_path,
            status="eval_complete",
            result_file_count=result_count,
            eval_completed_at_utc=datetime.now(timezone.utc).isoformat(),
            error=None,
        )
        summary = summarize_canitedit_run(run_dir)
        send_telegram(
            args.notify_script,
            benchmark_message(model_name, "done", format_percent((summary or {}).get("pass_at_1"))),
        )
    except Exception as exc:
        update_summary(
            summary_path,
            status="eval_failed",
            error=str(exc),
            eval_failed_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        send_telegram(args.notify_script, benchmark_message(model_name, "failed"))
        raise


if __name__ == "__main__":
    main()
