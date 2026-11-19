"""Stage 7: counterfactual and natural-order evaluation of one reader on InfoSeek.

Same questions, images, passages, prompt, greedy decoding (max 32 tokens) and official
scorer as the P17 cells; only the passage list changes. Three conditions per question:

  none           no passages at all (the authors' prompt with an empty Context block).
                 Measures what the reader answers from the image + its own knowledge.
  gold_removed   the 9 P17 distractors in their k10_first order, gold left out (k = 9).
                 If accuracy stays near the k=10 level, the reader is not using the gold
                 passage; if it drops to the "none" level, it is.
  natural        the retriever's real top 10 (retrieval_p17.jsonl), nothing inserted or
                 moved: the end-to-end setting. Gold is present only when PreFLMR ranked
                 it in the top 10; its rank is recorded per question.

Adapter: set $LAE_ADAPTER (the same [lae patch] hook as every other evaluation); unset =
the frozen reader.  Resumable: per (question, condition) results are cached in
<out>/cache.jsonl.  --compare aggregates every finished model (seed means, paired
bootstrap B=10,000, PCG64 seed 42, percentile 95% CI) into COUNTERFACTUAL.md.

  python lae/counterfactual.py --authors-dir A --frozen runs/p17_repro --out runs/counterfactual/NAME
  python lae/counterfactual.py --compare --frozen runs/p17_repro --root runs/counterfactual --out results
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path

CONDS = ("none", "gold_removed", "natural")
K = 10
MAX_NEW = 32           # READER_MAX_NEW in the authors' P17 runner
B, SEED = 10_000, 42
COND_LABEL = {"frozen": "Frozen Qwen", "standard": "Standard LoRA SFT", "balanced": "Balanced LoRA SFT",
              "goldfirst": "Gold-first LoRA SFT (control)"}


def log(msg: str) -> None:
    print(f"[cf {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wilson(h: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p, z2 = h / n, z * z
    c = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / (1 + z2 / n)
    return c - half, c + half


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def load_frozen(frozen: Path) -> tuple[dict, dict]:
    """k10_first rows (gold at index 0, then the 9 distractors) and the natural top-50 lists."""
    k10 = {r["qid"]: r for r in read_jsonl(frozen / "predictions_k10_first.jsonl")}
    top = {r["qid"]: r["top_ids"] for r in read_jsonl(frozen / "retrieval_p17.jsonl")}
    missing = [q for q in k10 if q not in top]
    if missing:
        raise SystemExit(f"{len(missing)} questions have no retrieval record, e.g. {missing[0]}")
    return k10, top


def plan_for(q: str, k10: dict, top: dict) -> dict[str, list[str]]:
    ordered = k10[q]["passages_ordered"]
    gold = k10[q]["gold_id"]
    assert ordered[0] == gold and len(ordered) == K, f"{q}: unexpected k10_first layout"
    return {"none": [], "gold_removed": ordered[1:], "natural": top[q][:K]}


# ------------------------------------------------------------------ run
def cmd_run(args) -> int:
    import torch
    sys.path.insert(0, str(args.authors_dir / "src"))
    import pyarrow.parquet as pq
    from tts_kbvqa.data.infoseek import download_passages_parquet, download_val_parquet, load_passages
    from tts_kbvqa.eval import infoseek_official
    from tts_kbvqa.reader.qwen import QwenReader

    torch.backends.cuda.enable_cudnn_sdp(False)   # as in the authors' runners
    torch.manual_seed(SEED)
    args.out.mkdir(parents=True, exist_ok=True)
    adapter = os.environ.get("LAE_ADAPTER", "")

    k10, top = load_frozen(args.frozen)
    qids = sorted(k10)
    plans = {q: plan_for(q, k10, top) for q in qids}
    log(f"{len(qids)} questions x {len(CONDS)} conditions; adapter={adapter or '<none, frozen>'}")

    cols = ["question_id", "question", "answers", "answer_eval", "img_path", "question_type"]
    rows = {r["question_id"]: r for r in pq.read_table(download_val_parquet(), columns=cols).to_pylist()}
    id_to_passage = {p.passage_id: p for p in load_passages(download_passages_parquet())}
    image_dir = args.authors_dir / "data/infoseek/images/val"

    cache_path = args.out / "cache.jsonl"
    cache: dict[tuple[str, str], dict] = {}
    if cache_path.exists():
        for r in read_jsonl(cache_path):
            cache[(r["qid"], r["cond"])] = r
    todo = [(q, c) for q in qids for c in CONDS if (q, c) not in cache]
    log(f"cached {len(cache)}, to run {len(todo)}")

    t0, peak = time.time(), None
    if todo:
        reader = QwenReader(dtype=torch.bfloat16, model_id="Qwen/Qwen2.5-VL-7B-Instruct")
        torch.cuda.reset_peak_memory_stats()
        with cache_path.open("a") as fh:
            for i, (q, c) in enumerate(todo, 1):
                r = rows[q]
                pids = plans[q][c]
                ts = time.time()
                out = reader.answer(image=image_dir / r["img_path"], question=r["question"],
                                    passages=[id_to_passage[p] for p in pids], max_new_tokens=MAX_NEW)
                rec = {"qid": q, "cond": c, "passages_ordered": pids, "pred": out.answer,
                       "input_tokens": out.input_token_len, "gen_tokens": out.generated_token_len,
                       "reader_latency_s": time.time() - ts}
                cache[(q, c)] = rec
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if i % 200 == 0 or i == len(todo):
                    rate = (time.time() - t0) / i
                    log(f"  {i}/{len(todo)}  {rate:.2f}s/call  eta {rate * (len(todo) - i) / 60:.1f} min  "
                        f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        peak = torch.cuda.max_memory_allocated() / 1e9

    # score with the official metric, exactly as the P17 runner does
    summary = {"name": args.out.name, "lae_adapter": adapter, "n": len(qids), "cells": {},
               "peak_alloc_gb": peak, "wall_s_this_session": time.time() - t0}
    for c in CONDS:
        preds = []
        for q in qids:
            r, rec = rows[q], cache[(q, c)]
            ok = bool(infoseek_official.is_correct(rec["pred"], r["question_type"], list(r["answers"] or []),
                                                   gold_answer_eval=list(r["answer_eval"] or [])))
            gold = k10[q]["gold_id"]
            p = {**rec, "gold_id": gold, "gold_via_fallback": k10[q]["gold_via_fallback"],
                 "correct_official": ok, "gold_in_context": gold in rec["passages_ordered"],
                 "gold_rank": rec["passages_ordered"].index(gold) if gold in rec["passages_ordered"] else None}
            preds.append(p)
        (args.out / f"predictions_{c}.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds))
        hits = sum(p["correct_official"] for p in preds)
        lo, hi = wilson(hits, len(preds))
        cell = {"n": len(preds), "hits": hits, "acc": hits / len(preds), "ci_low": lo, "ci_high": hi,
                "mean_input_tokens": sum(p["input_tokens"] for p in preds) / len(preds)}
        if c == "natural":
            with_g = [p for p in preds if p["gold_in_context"]]
            without = [p for p in preds if not p["gold_in_context"]]
            cell["n_gold_in_top10"] = len(with_g)
            cell["acc_gold_in_top10"] = sum(p["correct_official"] for p in with_g) / len(with_g) if with_g else None
            cell["acc_gold_absent"] = sum(p["correct_official"] for p in without) / len(without) if without else None
        summary["cells"][c] = cell
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    log("accuracy: " + "  ".join(f"{c}={100 * summary['cells'][c]['acc']:.2f}" for c in CONDS)
        + f"  (natural: gold in top-10 for {summary['cells']['natural']['n_gold_in_top10']} questions)")
    return 0


# ------------------------------------------------------------------ compare
def cmd_compare(args) -> int:
    import numpy as np
    k10, top = load_frozen(args.frozen)
    qids = sorted(k10)
    n = len(qids)
    frozen_k10 = {p: {r["qid"]: float(r["correct_official"]) for r in read_jsonl(args.frozen / f"predictions_k10_{p}.jsonl")}
                  for p in ("first", "last")}

    def load(d: Path) -> dict:
        out = {c: {r["qid"]: float(r["correct_official"]) for r in read_jsonl(d / f"predictions_{c}.jsonl")} for c in CONDS}
        for c in CONDS:
            if set(out[c]) != set(qids):
                raise SystemExit(f"{d.name}/{c}: question set differs from the frozen run")
        return out

    runs: dict[str, list[tuple[str, dict]]] = {c: [] for c in COND_LABEL}
    for d in sorted(args.root.iterdir()):
        if not (d / "summary.json").exists():
            continue
        if d.name == "frozen":
            runs["frozen"].append((d.name, load(d))); continue
        m = re.fullmatch(r"(standard|balanced_s\d+|goldfirst)_seed(\d+)", d.name)
        if m:
            runs["balanced" if m.group(1).startswith("balanced") else m.group(1)].append((d.name, load(d)))
    if not runs["frozen"]:
        raise SystemExit("no frozen counterfactual run found (runs/counterfactual/frozen)")

    vec = lambda cells, c: np.array([cells[c][q] for q in qids])
    models, status, per_seed = {}, {}, []
    for cond, label in COND_LABEL.items():
        if not runs[cond]:
            continue
        models[label] = {c: np.mean([vec(r, c) for _, r in runs[cond]], axis=0) for c in CONDS}
        status[label] = ("measured" if cond == "frozen" else
                         f"mean of {len(runs[cond])} seeds" if len(runs[cond]) >= args.planned_seeds
                         else f"PRELIMINARY: {len(runs[cond])} of {args.planned_seeds} seeds")
        for name, r in runs[cond]:
            per_seed.append({"run": name, **{c: round(100 * vec(r, c).mean(), 2) for c in CONDS}})

    # gold present in the natural top 10: same set for every model (same retrieval)
    present = np.array([k10[q]["gold_id"] in top[q][:K] for q in qids])
    final = []
    for label, V in models.items():
        row = {"model": label, "status": status[label], **{c: round(100 * V[c].mean(), 2) for c in CONDS},
               "natural_gold_in_top10": round(100 * V["natural"][present].mean(), 2),
               "natural_gold_absent": round(100 * V["natural"][~present].mean(), 2)}
        final.append(row)

    rng = np.random.Generator(np.random.PCG64(SEED))
    idx = rng.integers(0, n, size=(B, n))
    contrasts = []

    def add(label, fn):
        pt = fn(np.arange(n))
        s = np.array([fn(i) for i in idx])
        lo, hi = np.percentile(s, [2.5, 97.5])
        contrasts.append({"contrast": label, "estimate_pp": round(100 * pt, 2), "ci95_low_pp": round(100 * lo, 2),
                          "ci95_high_pp": round(100 * hi, 2), "excludes_zero": bool(lo > 0 or hi < 0), "n_questions": n})

    F = models["Frozen Qwen"]
    for label, V in models.items():
        add(f"{label}: gold_removed - none (evidence from distractors alone)",
            lambda i, V=V: V["gold_removed"][i].mean() - V["none"][i].mean())
        if label != "Frozen Qwen":
            for c in CONDS:
                add(f"{label} vs Frozen: {c}", lambda i, V=V, c=c: V[c][i].mean() - F[c][i].mean())
    S, Bm = COND_LABEL["standard"], COND_LABEL["balanced"]
    if S in models and Bm in models:
        for c in CONDS:
            add(f"{Bm} vs {S}: {c}", lambda i, c=c: models[Bm][c][i].mean() - models[S][c][i].mean())

    args.out.mkdir(parents=True, exist_ok=True)
    for fname, rows_ in (("counterfactual_final.csv", final), ("counterfactual_per_seed.csv", per_seed),
                         ("counterfactual_bootstrap_contrasts.csv", contrasts)):
        with (args.out / fname).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_[0]))
            w.writeheader()
            w.writerows(rows_)
    n_present = int(present.sum())
    md = [f"# Counterfactual and natural-order evaluation (generated {time.strftime('%Y-%m-%d %H:%M')})", "",
          f"InfoSeek, same {n} questions and official scorer as the P17 cells. Accuracy in %.",
          "none = no passages; gold_removed = the 9 P17 distractors only (k = 9); natural = the retriever's real "
          f"top 10, nothing inserted (gold is in it for {n_present} of {n} questions).",
          f"For reference, frozen k = 10 gold first / last: {100 * np.mean(list(frozen_k10['first'].values())):.2f} / "
          f"{100 * np.mean(list(frozen_k10['last'].values())):.2f}.", "",
          "| Model | none | gold removed | natural top-10 | natural, gold in top-10 | natural, gold absent | Status |",
          "|---|---|---|---|---|---|---|"]
    md += [f"| {r['model']} | {r['none']:.2f} | {r['gold_removed']:.2f} | {r['natural']:.2f} | "
           f"{r['natural_gold_in_top10']:.2f} | {r['natural_gold_absent']:.2f} | {r['status']} |" for r in final]
    md += ["", "## Paired bootstrap (B = 10,000, seed 42), percentage points", "",
           "| Contrast | Estimate | 95% CI | CI excludes 0 |", "|---|---|---|---|"]
    md += [f"| {c['contrast']} | {c['estimate_pp']:+.2f} | [{c['ci95_low_pp']:+.2f}, {c['ci95_high_pp']:+.2f}] | "
           f"{'yes' if c['excludes_zero'] else 'no'} |" for c in contrasts]
    (args.out / "COUNTERFACTUAL.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", type=Path, required=True, help="runs/p17_repro")
    ap.add_argument("--authors-dir", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="run: runs/counterfactual/NAME; compare: results dir")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--root", type=Path, help="runs/counterfactual (for --compare)")
    ap.add_argument("--planned-seeds", type=int, default=3)
    args = ap.parse_args()
    if args.compare:
        if not args.root:
            ap.error("--compare needs --root")
        return cmd_compare(args)
    if not args.authors_dir:
        ap.error("--authors-dir is required to run")
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
