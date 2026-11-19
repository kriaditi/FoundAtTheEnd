"""Stage 2b: add training conditions derived from the finished standard data.

Reads  <data>/final/standard/{train,dev}.jsonl  (built by build_train_data.py) and writes:

  * goldfirst/      the gold-always-first control: the same 10 passages per question,
                    gold moved to slot 0; the 9 distractors keep their relative order.
  * balanced_s<S>/  extra balanced seeds (same method and same RNG stream as the build:
                    balanced_slots(n, seed, stream) from build_train_data.py), for seeds
                    whose files do not exist yet.

Every new file is verified with the same hard checks as the original build (same
questions in the same order, same passage set, gold at its recorded slot, exactly one
positive, distractor order unchanged) and added to final/manifest.json (sha256).
Never overwrites an existing condition.  CPU only, seconds.

  python lae/extend_train_data.py --data-dir data/train --authors-dir third_party/lost_at_the_end_code \
      --balanced-seeds 2
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_train_data import K, balanced_slots, log, place, read_jsonl, sha256, write_jsonl  # noqa: E402

SPLITS = ("train", "dev")


def derive(base_rows: list[dict], cname: str, slots: list[int]) -> list[dict]:
    recs = []
    for r, slot in zip(base_rows, slots):
        # standard rows hold gold at natural_slot; place() moves it to `slot` and keeps the
        # distractor order, exactly as the original build does for the balanced conditions.
        pids = place(r["passage_ids"], r["natural_slot"], int(slot))
        recs.append({**r, "passage_ids": pids, "gold_slot": int(slot), "condition": cname})
    return recs


def check(cname: str, split: str, rows: list[dict], base: list[dict], eval_qids: set[str],
          uniform_n: int | None) -> list[str]:
    errors = []
    if [r["qid"] for r in rows] != [r["qid"] for r in base]:
        return [f"{cname}/{split}: question order differs from standard"]
    if {r["qid"] for r in rows} & eval_qids:
        errors.append(f"{cname}/{split}: contains evaluation qids")
    for r, b in zip(rows, base):
        pids = r["passage_ids"]
        if len(pids) != K or sorted(pids) != sorted(b["passage_ids"]):
            errors.append(f"{cname}/{split} {r['qid']}: passage set differs"); break
        if pids[r["gold_slot"]] != r["gold_pid"] or r["gold_pid"] != b["gold_pid"]:
            errors.append(f"{cname}/{split} {r['qid']}: gold not at gold_slot"); break
        if sum(p in set(r["pos_item_ids"]) for p in pids) != 1:
            errors.append(f"{cname}/{split} {r['qid']}: not exactly one positive among the 10"); break
        if [p for p in pids if p != r["gold_pid"]] != [p for p in b["passage_ids"] if p != b["gold_pid"]]:
            errors.append(f"{cname}/{split} {r['qid']}: distractor order changed"); break
    h = Counter(r["gold_slot"] for r in rows)
    if cname == "goldfirst":
        if set(h) != {0}:
            errors.append(f"goldfirst/{split}: gold not always at slot 0: {dict(h)}")
    elif uniform_n is not None and (set(h.values()) != {uniform_n // K} or len(h) != K):
        errors.append(f"{cname}/{split}: slot histogram not uniform {dict(h)}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True, help="the build's --out (holds final/)")
    ap.add_argument("--authors-dir", type=Path, required=True)
    ap.add_argument("--balanced-seeds", type=int, nargs="*", default=[],
                    help="extra balanced seeds to create (existing ones are left untouched)")
    ap.add_argument("--no-goldfirst", action="store_true")
    args = ap.parse_args()

    final = args.data_dir / "final"
    stats_path, manifest_path = final / "build_stats.json", final / "manifest.json"
    for p in (final / "standard/train.jsonl", final / "standard/dev.jsonl", stats_path, manifest_path):
        if not p.is_file():
            log(f"ERROR: missing {p}; run the build first"); return 1
    stats = json.loads(stats_path.read_text())
    if stats.get("hard_check_errors"):
        log(f"ERROR: the original build has check errors: {stats['hard_check_errors']}"); return 1
    manifest = json.loads(manifest_path.read_text())
    eval_qids = set(json.loads((args.authors_dir / "results/p11_7b_subset/subset500_v2_qids.json")
                               .read_text())["qids"])

    base = {s: read_jsonl(final / "standard" / f"{s}.jsonl") for s in SPLITS}
    for s in SPLITS:  # the derived files must start from the exact files the manifest describes
        if sha256(final / "standard" / f"{s}.jsonl") != manifest[f"standard/{s}.jsonl"]:
            log(f"ERROR: standard/{s}.jsonl does not match manifest.json"); return 1
    n = {s: len(base[s]) for s in SPLITS}
    log(f"standard data: {n['train']} train, {n['dev']} dev (verified against manifest)")

    conds: dict[str, int | None] = {}
    if not args.no_goldfirst:
        conds["goldfirst"] = None
    for seed in args.balanced_seeds:
        conds[f"balanced_s{seed}"] = seed

    hist = stats.setdefault("gold_slot_histograms", {})
    errors: list[str] = []
    for cname, seed in conds.items():
        if (final / cname).exists():
            log(f"skip {cname}: already exists (delete {final / cname} to regenerate)"); continue
        derived, errs = {}, []
        for split in SPLITS:  # derive and check both splits before writing anything
            rows = base[split]
            slots = ([0] * len(rows) if seed is None
                     else balanced_slots(len(rows), seed, 0 if split == "train" else 1).tolist())
            derived[split] = derive(rows, cname, slots)
            errs += check(cname, split, derived[split], rows, eval_qids, None if seed is None else n[split])
        if errs:
            errors += errs
            log(f"FAILED {cname}: {errs[0]}  (nothing written for this condition)"); continue
        for split, recs in derived.items():
            path = final / cname / f"{split}.jsonl"
            write_jsonl(path, recs)
            manifest[f"{cname}/{split}.jsonl"] = sha256(path)
            hist[f"{cname}/{split}"] = dict(sorted(Counter(r["gold_slot"] for r in recs).items()))
            log(f"wrote {path.relative_to(final)}: {len(recs)} rows, gold-slot histogram {hist[f'{cname}/{split}']}")

    stats["extended_conditions"] = sorted(set(stats.get("extended_conditions", [])) | set(conds))
    stats["extend_check_errors"] = errors
    stats_path.write_text(json.dumps(stats, indent=2))
    manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=2))
    if errors:
        log(f"ERRORS: {errors}"); return 1
    log("all checks passed; manifest.json and build_stats.json updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
