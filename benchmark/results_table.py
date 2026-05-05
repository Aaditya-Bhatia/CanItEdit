#!/usr/bin/env python3
"""Present CanItEdit benchmark results in a tabular format grouped by model and ablation.

Shows Pass@1 across ablations.
Uses the most recent run when duplicates exist for the same (model, ablation).
"""

import csv
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR.parent / "runs"

ABLATIONS = ["baseline", "clean", "dirty", "unclean74k"]
ABLATION_LABELS = ["Baseline", "LLM-Cleaning", "Static-Cleaning", "Unfiltered"]

MODEL_TYPE_MAP = {
    "DeepSeek-Coder-6.7B": "base",
    "Llama-3.2-3B": "base",
    "Qwen2.5-3B": "base",
    "Qwen2.5-Coder-3B-Instruct": "instruct",
    "Qwen2.5-Coder-7B": "base",
    "Qwen2.5-Coder-7B-Instruct": "instruct",
    "Qwen2.5-Coder-14B-Instruct": "instruct",
    "Qwen3-4B": "base",
    "Qwen3-8B": "base",
    "Qwen3-14B": "base",
    "StarCoder2-3B": "base",
    "StarCoder2-7B": "base",
    "StarCoder2-15B": "base",
}

MODEL_RULES = [
    (r"(?i)qwen3-14b-base", "Qwen3-14B"),
    (r"(?i)qwen3-8b-base", "Qwen3-8B"),
    (r"(?i)qwen3-4b-base", "Qwen3-4B"),
    (r"(?i)qwen2\.5-coder-14b-instruct", "Qwen2.5-Coder-14B-Instruct"),
    (r"(?i)qwen2\.5-coder-7b-instruct", "Qwen2.5-Coder-7B-Instruct"),
    (r"(?i)qwen2\.5-coder-7b", "Qwen2.5-Coder-7B"),
    (r"(?i)qwen2\.5-coder-3b-instruct", "Qwen2.5-Coder-3B-Instruct"),
    (r"(?i)qwen2\.5-3b", "Qwen2.5-3B"),
    (r"(?i)deepseek-coder-6\.7b", "DeepSeek-Coder-6.7B"),
    (r"(?i)llama-3\.2-3b", "Llama-3.2-3B"),
    (r"(?i)starcoder2-15b", "StarCoder2-15B"),
    (r"(?i)starcoder2-7b", "StarCoder2-7B"),
    (r"(?i)starcoder2-3b", "StarCoder2-3B"),
]

ABLATION_PATTERNS = [
    (r"lora-unclean74k|lora-unclean", "unclean74k"),
    (r"lora-clean|clean-edit|Clean_Edit", "clean"),
    (r"lora-dirty|dirty-edit|Dirty_Edit", "dirty"),
    (r"baseline|base-canitedit", "baseline"),
]

TOTAL_ITEMS = 210  # 105 problems x 2 instruction types


def classify_dir(dirname: str):
    """Return (model_name, ablation) or None if unrecognized."""
    model = None
    for pattern, name in MODEL_RULES:
        if re.search(pattern, dirname):
            model = name
            break
    if model is None:
        return None

    ablation = None
    for pattern, abl in ABLATION_PATTERNS:
        if re.search(pattern, dirname):
            ablation = abl
            break

    if ablation is None:
        if "lora" not in dirname.lower():
            ablation = "baseline"

    if ablation is None:
        return None
    return model, ablation


def get_dir_date(dirname: str) -> str:
    m = re.search(r"(\d{8}T\d{6}Z)", dirname)
    if m:
        return m.group(1)
    m = re.search(r"(\d{4}-\d{2}-\d{2})", dirname)
    if m:
        return m.group(1).replace("-", "") + "T000000Z"
    return "00000000T000000Z"


def count_files(dirpath: Path) -> tuple:
    gen = len([f for f in dirpath.glob("*.json.gz") if ".results." not in f.name])
    evl = len(list(dirpath.glob("*.results.json.gz")))
    return gen, evl


def status_str(gen: int, evl: int) -> tuple:
    if gen == 0:
        return "No", "No"
    gen_pct = f"{gen * 100 // TOTAL_ITEMS}%"
    if gen >= TOTAL_ITEMS:
        gen_pct = "100%"
    if evl == 0:
        return gen_pct, "No"
    evl_pct = f"{evl * 100 // TOTAL_ITEMS}%"
    if evl >= TOTAL_ITEMS:
        evl_pct = "Yes"
    return gen_pct, evl_pct


def scan_runs():
    """Scan all run directories and return classified dict and file counts."""
    all_dirs = sorted(
        d for d in RUNS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_") and not d.name.startswith("test-")
    )

    classified = {}  # (model, ablation) -> Path
    for d in all_dirs:
        result = classify_dir(d.name)
        if result is None:
            continue
        model, ablation = result
        key = (model, ablation)
        if key in classified:
            if get_dir_date(d.name) > get_dir_date(classified[key].name):
                classified[key] = d
        else:
            classified[key] = d

    dir_counts = {}
    for key, dirpath in classified.items():
        dir_counts[key] = count_files(dirpath)

    return classified, dir_counts


def run_pass_k(dirs):
    """Run pass_k.py on the given directories, return {dirname: pass1_score}."""
    if not dirs:
        return {}
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "pass_k.py")] + [str(d) for d in dirs],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("pass_k.py failed:", result.stderr, file=sys.stderr)
        return {}

    metrics = {}
    lines = result.stdout.strip().split("\n")
    header = lines[0].split(",")
    name_idx = header.index("Name")
    estimate_idx = header.index("Estimate")

    for line in lines[1:]:
        cols = line.split(",")
        metrics[cols[name_idx]] = float(cols[estimate_idx])
    return metrics


def load_results(classified, dir_counts, metrics):
    """Load main experiment results.

    Returns (results, gen_status, eval_status) dicts.
    """
    results = defaultdict(dict)
    gen_status = defaultdict(dict)
    eval_status = defaultdict(dict)
    ablation_set = set(ABLATIONS)
    for (model, ablation), dirpath in classified.items():
        if ablation not in ablation_set:
            continue
        gen, evl = dir_counts.get((model, ablation), (0, 0))
        gs, es = status_str(gen, evl)
        gen_status[model][ablation] = gs
        eval_status[model][ablation] = es
        dirname = dirpath.name
        if dirname in metrics:
            results[model][ablation] = metrics[dirname]
    return results, gen_status, eval_status


def get_model_type(model: str) -> str:
    return MODEL_TYPE_MAP.get(model, "base")


def print_table(results, gen_status, eval_status, ablations, ablation_labels,
                title=None, model_filter=None):
    all_models = sorted(
        m for m in (set(results) | set(gen_status) | set(eval_status))
        if model_filter is None or m in model_filter
    )
    if not all_models:
        return

    model_w = max(max((len(m) for m in all_models), default=20), len("Model"))
    type_w = 10
    col_w = max(12, max((len(l) for l in ablation_labels), default=12) + 2)
    extra_w = 35

    if title:
        print(f"\n{'=' * 40}")
        print(f"  {title}")
        print(f"{'=' * 40}")

    header = (
        f"{'Model':<{model_w}}  {'Type':<{type_w}}  "
        + "  ".join(f"{a:>{col_w}}" for a in ablation_labels)
        + f"  {'Not Generated':<{extra_w}}  {'Gen. Not Evald':<{extra_w}}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)

    for model in all_models:
        mtype = get_model_type(model)
        row = f"{model:<{model_w}}  {mtype:<{type_w}}  "
        cells = []
        for abl in ablations:
            score = results.get(model, {}).get(abl)
            if score is not None:
                cells.append(f"{score:>{col_w}.1f}%")
            else:
                cells.append(f"{'-':>{col_w}}")
        row += "  ".join(cells)

        not_gen = []
        gen_not_eval = []
        for abl, abl_label in zip(ablations, ablation_labels):
            if abl in results.get(model, {}):
                continue
            gs = gen_status.get(model, {}).get(abl, "No")
            es = eval_status.get(model, {}).get(abl, "No")
            if gs == "No":
                not_gen.append(abl_label)
            else:
                if es != "Yes":
                    gen_not_eval.append(f"{abl_label}({gs})" if gs != "100%" else abl_label)

        row += f"  {'; '.join(not_gen) if not_gen else '':<{extra_w}}"
        row += f"  {'; '.join(gen_not_eval) if gen_not_eval else '':<{extra_w}}"
        print(row)

    print(sep)
    print(f"\nTotal models: {len(all_models)}")
    total_runs = sum(len(v) for v in results.values())
    print(f"Total runs:   {total_runs}")


def save_csv(results, gen_status, eval_status, ablations, ablation_labels, path,
             model_filter=None):
    all_models = sorted(
        m for m in (set(results) | set(gen_status) | set(eval_status))
        if model_filter is None or m in model_filter
    )
    if not all_models:
        return

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Model", "Type"] + ablation_labels + ["Not Generated", "Gen. Not Eval'd"]
        )
        for model in all_models:
            mtype = get_model_type(model)
            row = [model, mtype]
            for abl in ablations:
                score = results.get(model, {}).get(abl)
                row.append(f"{score:.1f}" if score is not None else "-")

            not_gen = []
            gen_not_eval = []
            for abl, abl_label in zip(ablations, ablation_labels):
                if abl in results.get(model, {}):
                    continue
                gs = gen_status.get(model, {}).get(abl, "No")
                es = eval_status.get(model, {}).get(abl, "No")
                if gs == "No":
                    not_gen.append(abl_label)
                else:
                    if es != "Yes":
                        gen_not_eval.append(
                            f"{abl_label}({gs})" if gs != "100%" else abl_label
                        )

            row.append("; ".join(not_gen) if not_gen else "")
            row.append("; ".join(gen_not_eval) if gen_not_eval else "")
            writer.writerow(row)

    print(f"\nCSV saved to: {path}")


def save_markdown(results, gen_status, eval_status, ablations, ablation_labels, path,
                  title="Results", model_filter=None):
    all_models = sorted(
        m for m in (set(results) | set(gen_status) | set(eval_status))
        if model_filter is None or m in model_filter
    )
    if not all_models:
        return

    lines = [f"# {title}\n"]

    header = "| Model | Type |"
    sep = "| --- | --- |"
    for label in ablation_labels:
        header += f" {label} |"
        sep += " --- |"
    header += " Not Generated | Gen. Not Eval'd |"
    sep += " --- | --- |"
    lines.append(header)
    lines.append(sep)

    for model in all_models:
        mtype = get_model_type(model)
        row = f"| {model} | {mtype} |"
        for abl in ablations:
            score = results.get(model, {}).get(abl)
            val = f"{score * 100:.1f}" if score is not None else "-"
            row += f" {val} |"

        not_gen = []
        gen_not_eval = []
        for abl, abl_label in zip(ablations, ablation_labels):
            if abl in results.get(model, {}):
                continue
            gs = gen_status.get(model, {}).get(abl, "No")
            es = eval_status.get(model, {}).get(abl, "No")
            if gs == "No":
                not_gen.append(abl_label)
            else:
                if es != "Yes":
                    gen_not_eval.append(f"{abl_label}({gs})" if gs != "100%" else abl_label)

        row += f" {'; '.join(not_gen) if not_gen else ''} |"
        row += f" {'; '.join(gen_not_eval) if gen_not_eval else ''} |"
        lines.append(row)

    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Markdown saved to: {path}")


def main():
    classified, dir_counts = scan_runs()
    if not classified:
        print("No recognized runs.")
        sys.exit(1)

    evaled_dirs = [
        d for d in set(classified.values())
        if list(d.glob("*.results.json.gz"))
    ]
    metrics = run_pass_k(evaled_dirs)

    main_res, main_gen, main_eval = load_results(classified, dir_counts, metrics)

    print_table(main_res, main_gen, main_eval, ABLATIONS, ABLATION_LABELS,
                title="MAIN RESULTS")

    base_dir = SCRIPT_DIR.parent
    save_csv(main_res, main_gen, main_eval, ABLATIONS, ABLATION_LABELS,
             base_dir / "results.csv")
    save_markdown(main_res, main_gen, main_eval, ABLATIONS, ABLATION_LABELS,
                  base_dir / "results.md", title="CanItEdit — Results")


if __name__ == "__main__":
    main()
