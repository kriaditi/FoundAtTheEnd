"""Stage 8, step 0: inspect the M2KR Encyclopedic-VQA TRAIN files before building the
mixed (InfoSeek + E-VQA) training data.  Text only, no images, no GPU, read-only.

Prints: the E-VQA files in the M2KR repos (names, sizes), train-question columns and
counts, answer and positive-passage statistics, the train KB size and its overlap with
the KB the authors use at test time, and leakage against the 500 E-VQA evaluation
questions (same image file, same entity if an entity column exists, same gold passage).

Output: printed and saved to runs/inspect_evqa_train/report.txt.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

M2KR = "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR"
IMAGES = "BByrneLab/M2KR_Images"

lines: list[str] = []


def out(s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)


def list_files(repo: str, needle: str) -> list[tuple[str, int]]:
    info = HfApi().dataset_info(repo, files_metadata=True)
    return sorted((s.rfilename, s.size or 0) for s in info.siblings if needle.lower() in s.rfilename.lower())


def load(files: list[str], cols=None) -> list[dict]:
    rows = []
    for f in files:
        t = pq.read_table(hf_hub_download(M2KR, f, repo_type="dataset"))
        if cols:
            t = t.select([c for c in cols if c in t.column_names])
        rows += t.to_pylist()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    authors = Path(args.authors_dir)

    out("== M2KR files with 'evqa' in the path ==")
    files = list_files(M2KR, "evqa")
    for name, size in files:
        out(f"  {name:70s} {size / 1e6:9.1f} MB")
    out("== M2KR_Images files with 'evqa' in the path ==")
    for name, size in list_files(IMAGES, "evqa"):
        out(f"  {name:70s} {size / 1e9:9.2f} GB")

    # Only the small "EVQA_*" release (the KB the authors evaluate on: ~51k passages). The
    # "EVQAFull_*" passages are ~8 GB across 29 shards and are not used anywhere in this study.
    train_q = [n for n, _ in files if n.startswith("EVQA_data/train")]
    train_kb = [n for n, _ in files if n.startswith("EVQA_passages/train")]
    test_kb = [n for n, _ in files if n.startswith("EVQA_passages/test")]
    out(f"\n  train question files: {train_q}\n  train KB files: {train_kb}\n  test KB files: {test_kb}")
    if not train_q:
        out("ERROR: no E-VQA train question file found; see the listing above")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(lines) + "\n")
        return 1

    out("\n== E-VQA train questions ==")
    qs = load(train_q)
    cols = sorted(qs[0].keys())
    out(f"  rows: {len(qs)}   columns: {cols}")
    out(f"  sample row: {json.dumps({k: str(v)[:150] for k, v in qs[0].items()}, indent=1)}")
    for c in ("question_type", "dataset_name", "dataset_type"):
        if c in cols:
            out(f"  {c}: {Counter(r.get(c) for r in qs).most_common(8)}")
    npos = Counter(min(len(r.get("pos_item_ids") or []), 5) for r in qs)
    out(f"  #positives per question (5 = 5+): {sorted(npos.items())}")
    nans = Counter(min(len(r.get("answers") or []), 5) for r in qs)
    out(f"  #answers per question (5 = 5+): {sorted(nans.items())}")
    ans_words = [len(str((r.get("answers") or [""])[0]).split()) for r in qs]
    out(f"  words in first answer: mean {sum(ans_words) / len(ans_words):.2f}, "
        f"share with 1 word {sum(w == 1 for w in ans_words) / len(ans_words):.2f}, "
        f"share with >3 words {sum(w > 3 for w in ans_words) / len(ans_words):.2f}")
    img_col = "img_path" if "img_path" in cols else next((c for c in cols if "img" in c or "image" in c), None)
    ent_col = "entity_id" if "entity_id" in cols else next((c for c in cols if "entity" in c), None)
    out(f"  image column: {img_col}   entity column: {ent_col}")
    if img_col:
        out(f"  unique images: {len({r.get(img_col) for r in qs})}")
    if ent_col:
        out(f"  unique entities: {len({r.get(ent_col) for r in qs})}")

    out("\n== E-VQA KB ==")
    if train_kb:
        kb_ids = {r["passage_id"] for r in load(train_kb, ["passage_id"])}
        out(f"  train KB passages: {len(kb_ids)}")
        has_gold = sum(1 for r in qs if set(r.get("pos_item_ids") or []) & kb_ids)
        out(f"  train questions with >=1 positive in the train KB: {has_gold} / {len(qs)}")
    else:
        kb_ids = set()
        out("  no train KB file found")
    if test_kb:
        test_ids = {r["passage_id"] for r in load(test_kb, ["passage_id"])}
        out(f"  test KB passages: {len(test_ids)}   shared ids with train KB: {len(kb_ids & test_ids)}")
        has_gold_test = sum(1 for r in qs if set(r.get("pos_item_ids") or []) & test_ids)
        out(f"  train questions with >=1 positive in the TEST KB (the one used at evaluation): {has_gold_test} / {len(qs)}")

    out("\n== overlap with the 500 E-VQA evaluation questions (authors' subset) ==")
    try:
        sys.path.insert(0, str(authors / "src"))
        spec = importlib.util.spec_from_file_location("p20", authors / "scripts/run_p20_evqa_replication.py")
        p20 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(p20)
        rows = [json.loads(l) for l in (authors / "results/p20/subset500_evqa_v1.jsonl").open() if l.strip()]
        sub = {r["qid"] for r in rows if "qid" in r}   # the file has a non-question header record
        samples = [s for s in p20.evqa_data.load_test_samples(p20.IMAGE_DIR) if s.question_id in sub]
        out(f"  eval samples found: {len(samples)} / {len(sub)}")
        attrs = [a for a in dir(samples[0]) if not a.startswith("_")]
        out(f"  eval sample attributes: {attrs}")
        ev_img = {Path(s.image_filename).name for s in samples if getattr(s, "image_filename", None)}
        ev_gold = {p for s in samples for p in (getattr(s, "pos_item_ids", None) or [])}
        if img_col:
            out(f"  train questions sharing an image file with eval: "
                f"{sum(1 for r in qs if Path(str(r.get(img_col))).name in ev_img)}")
        out(f"  train questions sharing a gold passage id with eval: "
            f"{sum(1 for r in qs if set(r.get('pos_item_ids') or []) & ev_gold)}")
        ev_ent = {getattr(s, "entity_id", None) for s in samples} - {None}
        if ent_col and ev_ent:
            out(f"  train questions sharing an entity with eval: {sum(1 for r in qs if r.get(ent_col) in ev_ent)}")
        else:
            out("  entity overlap: not computable (no entity field on one side); gold-passage overlap is the proxy")
        out(f"  train question ids equal to eval ids: {len(sub & {r.get('question_id') for r in qs})}")
    except Exception as e:  # the inspection must still produce a report
        out(f"  overlap check failed: {type(e).__name__}: {e}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    out(f"\nsaved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
