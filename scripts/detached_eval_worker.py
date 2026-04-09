#!/usr/bin/env python3

import argparse
import errno
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


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


def run(cmd: list[str], cwd: Path):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    summary_path = Path(args.summary_path).resolve()
    repo_root = Path(__file__).resolve().parents[1]

    try:
        update_summary(
            summary_path,
            status="eval_running",
            eval_started_at_utc=datetime.now(timezone.utc).isoformat(),
            detached_worker_pid=os.getpid(),
        )
        run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{run_dir}:/results:rw",
                "ghcr.io/nuprl/canitedit",
                "--dir",
                "/results",
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
        )
        send_telegram(args.notify_script, f"CanItEdit eval done. run_dir={run_dir}")
    except Exception as exc:
        update_summary(
            summary_path,
            status="eval_failed",
            error=str(exc),
            eval_failed_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        send_telegram(args.notify_script, f"CanItEdit eval failed. run_dir={run_dir} error={exc}")
        raise


if __name__ == "__main__":
    main()
