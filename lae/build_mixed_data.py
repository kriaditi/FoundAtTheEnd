"""Stage 8: build the MIXED training data (InfoSeek + Encyclopedic-VQA), compute-matched to
the InfoSeek-only runs: the same 20,000 train + 500 dev rows, half from each source.

  * InfoSeek half: the first 10,000 train / 250 dev rows of the finished standard data
    (same questions, passages and natural gold slots as the InfoSeek-only runs).
  * E-VQA half: built the same way as the InfoSeek data (lae/build_train_data.py):
      - M2KR EVQA_data/train, leakage filter against the authors' 500 evaluation questions
        (same image file, same gold passage, same entity = the passage-id stem
        "WikiWeb_<entity>"); E-VQA has no entity column, so the stem is the entity key
      - random pool (seed 42), images extracted from the two E-VQA archives
      - frozen PreFLMR, top 20, over the SAME index the evaluation uses (data/index/preflmr_evqa,
        51,462 passages; every train positive is in it) and the authors' query form
      - kept: exactly one positive in the top 10 -> natural gold slot
  * Conditions: mixed_standard (natural slots) and mixed_balanced_s<S> (gold moved to a slot
    drawn so each of the 10 slots gets exactly 2,000 train / 50 dev rows, same RNG scheme as
    balanced_s<S>). The 9 distractors keep their order.

Rows carry "source" and "image_root" (relative to the authors' repo) so lae/train_lora.py can
resolve images and passages from both datasets. Hard checks as in the original build, plus
manifest.json entries.  Resumable: candidates, images and retrieval are cached under --out.

  python lae/build_mixed_data.py --authors-dir A --infoseek-final data/train/final --out data/train_mixed \
      --pool 40000 --per-source-train 10000 --per-source-dev 250 --balanced-seeds 0 1 2
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import tarfile
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_train_data import K, TOP, balanced_slots, place, qualify, read_jsonl, sha256, write_jsonl  # noqa: E402

M2KR = "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR"
EVQA_TRAIN = "EVQA_data/train-00000-of-00001.parquet"
COLS = ["question_id", "image_id", "question", "answers", "gold_answer", "img_path",
        "question_type", "pos_item_ids", "instruction"]
INFOSEEK_IMAGE_ROOT = "data/infoseek/images/train"
EVQA_IMAGE_ROOT = "data/evqa/images"


def log(msg: str) -> None:
    print(f"[mixed {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def entity_of(pid: str) -> str:
    """E-VQA passage ids look like 'WikiWeb_<entity name>_<n>': the stem is the entity."""
    return pid.rsplit("_", 1)[0]


# ---------------------------------------------------------------- phase 1
def phase_candidates(args, p20, out: Path, stats: dict) -> list[dict]:
    path = out / "evqa_candidates.jsonl"
    if path.exists():
        cands = read_jsonl(path)
        stats.update(json.loads((out / "evqa_candidates_stats.json").read_text()))
        log(f"phase 1: reusing {len(cands)} E-VQA candidates")
        return cands

    log("phase 1: load the 500 E-VQA evaluation questions (leakage filter)")
    rows_sub = [json.loads(l) for l in (args.authors_dir / "results/p20/subset500_evqa_v1.jsonl").open() if l.strip()]
    sub = {r["qid"] for r in rows_sub if "qid" in r}
    ev = [s for s in p20.evqa_data.load_test_samples(p20.IMAGE_DIR) if s.question_id in sub]
    assert len(ev) == len(sub) == 500, f"found {len(ev)} of {len(sub)} eval samples"
    ev_img = {Path(s.image_filename).name for s in ev}
    ev_gold = {p for s in ev for p in s.pos_item_ids}
    ev_ent = {entity_of(p) for p in ev_gold}

    log("phase 1: load E-VQA train questions")
    rows = pq.read_table(hf_hub_download(M2KR, EVQA_TRAIN, repo_type="dataset"), columns=COLS).to_pylist()
    kb_ids = {p.passage_id for p in p20.evqa_data.load_passages(p20.evqa_data.download_passages_parquet())}
    drop, keep = Counter(), []
    for r in rows:
        pos = list(r.get("pos_item_ids") or [])
        if Path(r["img_path"]).name in ev_img:
            drop["same_image_as_eval"] += 1
        elif set(pos) & ev_gold:
            drop["shares_gold_passage_with_eval"] += 1
        elif {entity_of(p) for p in pos} & ev_ent:
            drop["same_entity_as_eval"] += 1
        elif not (r.get("answers") or []):
            drop["no_answer"] += 1
        elif not (set(pos) & kb_ids):
            drop["no_positive_in_kb"] += 1
        else:
            keep.append(r)
    del rows
    gc.collect()
    rng = np.random.Generator(np.random.PCG64(args.seed))
    order = rng.permutation(len(keep))[: args.pool]
    cands = [keep[int(i)] for i in order]
    for r in cands:
        r["pos_item_ids"] = list(r.get("pos_item_ids") or [])
        r["answers"] = [str(a) for a in r["answers"]]
        r["target"] = str(r.get("gold_answer") or "").strip() or r["answers"][0]   # same rule as InfoSeek
    stats["evqa_train_rows"] = len(keep) + sum(drop.values())
    stats["evqa_dropped"] = dict(drop)
    stats["evqa_eligible_after_filter"] = len(keep)
    stats["evqa_pool"] = len(cands)
    write_jsonl(path, cands)
    (out / "evqa_candidates_stats.json").write_text(json.dumps(stats, indent=2))
    log(f"phase 1: dropped {dict(drop)}; eligible {len(keep)}; pool {len(cands)}")
    return cands


# ---------------------------------------------------------------- phase 2
def phase_images(p20, cands: list[dict], image_dir: Path, stats: dict) -> None:
    wanted = {}
    for r in cands:
        if not (image_dir / r["img_path"]).is_file():
            wanted.setdefault(Path(r["img_path"]).name, r["img_path"])
    log(f"phase 2: {len({r['img_path'] for r in cands})} unique images for the pool, {len(wanted)} to extract")
    ev = p20.evqa_data
    for kind, fname in (("zip", ev.INAT_ZIP), ("tar", ev.GOOGLE_LANDMARK_TAR)):
        if not wanted:
            break
        path = Path(hf_hub_download(ev.IMAGES_REPO, fname, repo_type="dataset"))
        log(f"  scanning {fname} ({path.stat().st_size / 1e9:.1f} GB), {len(wanted)} wanted")
        t0 = time.time()
        if kind == "zip":
            with zipfile.ZipFile(path) as zf:
                for info in zf.infolist():
                    rel = wanted.get(Path(info.filename).name)
                    if rel and not info.is_dir():
                        dst = image_dir / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_bytes(zf.read(info))
                        del wanted[Path(info.filename).name]
                        if not wanted:
                            break
        else:
            with tarfile.open(path, "r|*") as tf:
                for m in tf:
                    if not m.isfile():
                        continue
                    rel = wanted.get(Path(m.name).name)
                    if rel:
                        dst = image_dir / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_bytes(tf.extractfile(m).read())
                        del wanted[Path(m.name).name]
                        if not wanted:
                            break
        log(f"  done in {time.time() - t0:.0f}s, {len(wanted)} still wanted")
    stats["evqa_pool_images_missing"] = len(wanted)


# ---------------------------------------------------------------- phase 3
def phase_retrieve(args, p20, cands: list[dict], image_dir: Path, out: Path, stats: dict) -> dict[str, list[str]]:
    import torch
    from tts_kbvqa.retrieval.preflmr import PreFLMRRetriever
    torch.backends.cuda.enable_cudnn_sdp(False)
    target = args.per_source_train + args.per_source_dev
    path = out / f"evqa_retrieval_top{TOP}.jsonl"
    done: dict[str, list[str]] = {r["qid"]: r["top"] for r in read_jsonl(path)} if path.exists() else {}
    usable = [r for r in cands if (image_dir / r["img_path"]).is_file()]
    kept = sum(1 for r in usable if r["question_id"] in done and qualify(done[r["question_id"]], r["pos_item_ids"]) is not None)
    log(f"phase 3: {len(done)} retrieved before, {kept} qualify, target {target}")
    if kept >= target:
        return done

    index_dir = p20.INDEX_DIR
    man = index_dir / "manifest.json"
    if not (man.exists() and "build_wall_s" in json.loads(man.read_text())):
        raise SystemExit(f"E-VQA index {index_dir} not found or incomplete: run the frozen E-VQA evaluation first")
    log(f"  using the evaluation index {index_dir}")
    retriever = PreFLMRRetriever(dtype=torch.bfloat16)
    index = retriever.load_index(str(index_dir))
    passages = p20.evqa_data.load_passages(p20.evqa_data.download_passages_parquet())
    id_to_passage = {p.passage_id: p for p in passages}
    stats["evqa_kb_passages"] = len(id_to_passage)

    todo = [r for r in usable if r["question_id"] not in done]
    t0, n_new = time.time(), 0
    with path.open("a") as fh:
        for i in range(0, len(todo), args.batch):
            batch = todo[i:i + args.batch]
            samples = [p20.evqa_data.EVQASample(
                question_id=r["question_id"], image_id=str(r.get("image_id") or ""), question=r["question"],
                answers=r["answers"], gold_answer=str(r.get("gold_answer") or ""), image_filename=r["img_path"],
                question_type=str(r.get("question_type") or ""), pos_item_ids=r["pos_item_ids"],
                instruction=str(r.get("instruction") or ""), image_dir=image_dir) for r in batch]
            tops = retriever.retrieve_batch(samples, index, id_to_passage, k=TOP)
            for r, rps in zip(batch, tops):
                top = [rp.passage_id for rp in rps]
                done[r["question_id"]] = top
                fh.write(json.dumps({"qid": r["question_id"], "top": top}) + "\n")
                if qualify(top, r["pos_item_ids"]) is not None:
                    kept += 1
            fh.flush()
            n_new += len(batch)
            if (i // args.batch) % 50 == 0:
                rate = (time.time() - t0) / n_new
                log(f"  retrieved {n_new} new ({len(done)} total), kept {kept}/{target}, {rate:.2f}s/q")
            if kept >= target:
                break
    stats["evqa_retrieval_s_per_question"] = round((time.time() - t0) / max(n_new, 1), 3)
    return done


# ---------------------------------------------------------------- phase 4
def evqa_items(args, cands, done, image_dir: Path) -> list[dict]:
    items = []
    for r in cands:
        top = done.get(r["question_id"])
        if top is None or not (image_dir / r["img_path"]).is_file():
            continue
        nat = qualify(top, r["pos_item_ids"])
        if nat is None:
            continue
        items.append({"qid": r["question_id"], "question": r["question"], "answers": r["answers"],
                      "target": r["target"], "question_type": r.get("question_type"), "img_path": r["img_path"],
                      "passage_ids": top[:K], "gold_pid": top[nat], "gold_slot": nat, "natural_slot": nat,
                      "pos_item_ids": r["pos_item_ids"], "source": "evqa", "image_root": EVQA_IMAGE_ROOT})
        if len(items) == args.per_source_train + args.per_source_dev:
            break
    return items


def infoseek_rows(final: Path, split: str, n: int) -> list[dict]:
    rows = read_jsonl(final / "standard" / f"{split}.jsonl")[:n]
    return [{**r, "source": "infoseek", "image_root": INFOSEEK_IMAGE_ROOT} for r in rows]


def phase_build(args, ev_items: list[dict], out: Path, stats: dict) -> dict[str, dict[str, list[dict]]]:
    n_tr, n_dev = args.per_source_train, args.per_source_dev
    if len(ev_items) < n_tr + n_dev:
        raise SystemExit(f"only {len(ev_items)} qualifying E-VQA items, need {n_tr + n_dev}")
    splits = {"train": infoseek_rows(args.infoseek_final, "train", n_tr) + ev_items[:n_tr],
              "dev": infoseek_rows(args.infoseek_final, "dev", n_dev) + ev_items[n_tr:n_tr + n_dev]}
    for split, rows in splits.items():
        assert len({r["qid"] for r in rows}) == len(rows), f"{split}: duplicate qids across sources"
    final = out / "final"
    conds = {"mixed_standard": None, **{f"mixed_balanced_s{s}": s for s in args.balanced_seeds}}
    hist = {}
    for cname, seed in conds.items():
        for split, rows in splits.items():
            slots = ([r["natural_slot"] for r in rows] if seed is None
                     else balanced_slots(len(rows), seed, 0 if split == "train" else 1).tolist())
            recs = []
            for r, slot in zip(rows, slots):
                pids = place(r["passage_ids"], r["natural_slot"], int(slot))
                recs.append({**{k: r[k] for k in ("qid", "question", "answers", "target", "question_type", "img_path",
                                                   "gold_pid", "natural_slot", "pos_item_ids", "source", "image_root")},
                             "passage_ids": pids, "gold_slot": int(slot), "condition": cname})
            write_jsonl(final / cname / f"{split}.jsonl", recs)
            hist[f"{cname}/{split}"] = dict(sorted(Counter(slots).items()))
    stats["kept_train"], stats["kept_dev"] = len(splits["train"]), len(splits["dev"])
    stats["gold_slot_histograms"] = hist
    stats["source_counts"] = {s: Counter(r["source"] for r in rows) for s, rows in splits.items()}
    return splits


# ---------------------------------------------------------------- phase 5
def phase_check(args, out: Path, stats: dict) -> list[str]:
    errors = []
    final = out / "final"
    conds = ["mixed_standard"] + [f"mixed_balanced_s{s}" for s in args.balanced_seeds]
    is_eval = set(json.loads((args.authors_dir / "results/p11_7b_subset/subset500_v2_qids.json").read_text())["qids"])
    # E-VQA train and test ids share the 'EVQA_<n>' naming, so leakage is checked on images and
    # entities (phase 1), not on ids; InfoSeek ids are globally unique.
    for split, n in (("train", 2 * args.per_source_train), ("dev", 2 * args.per_source_dev)):
        data = {c: read_jsonl(final / c / f"{split}.jsonl") for c in conds}
        base = data["mixed_standard"]
        if len(base) != n:
            errors.append(f"{split}: {len(base)} rows, expected {n}")
        if {r["qid"] for r in base if r["source"] == "infoseek"} & is_eval:
            errors.append(f"{split}: contains InfoSeek evaluation qids")
        src = Counter(r["source"] for r in base)
        if src != {"infoseek": n // 2, "evqa": n // 2}:
            errors.append(f"{split}: source counts {dict(src)}")
        for c, rows in data.items():
            if [r["qid"] for r in rows] != [r["qid"] for r in base]:
                errors.append(f"{c}/{split}: question order differs"); continue
            for r, b in zip(rows, base):
                pids = r["passage_ids"]
                if len(pids) != K or sorted(pids) != sorted(b["passage_ids"]):
                    errors.append(f"{c}/{split} {r['qid']}: passage set differs"); break
                if pids[r["gold_slot"]] != r["gold_pid"]:
                    errors.append(f"{c}/{split} {r['qid']}: gold not at gold_slot"); break
                if sum(p in set(r["pos_item_ids"]) for p in pids) != 1:
                    errors.append(f"{c}/{split} {r['qid']}: not exactly one positive"); break
                if [p for p in pids if p != r["gold_pid"]] != [p for p in b["passage_ids"] if p != b["gold_pid"]]:
                    errors.append(f"{c}/{split} {r['qid']}: distractor order changed"); break
                if not (args.authors_dir / r["image_root"] / r["img_path"]).is_file():
                    errors.append(f"{c}/{split} {r['qid']}: image missing"); break
            if c == "mixed_standard":
                if any(r["gold_slot"] != r["natural_slot"] for r in rows):
                    errors.append(f"mixed_standard/{split}: gold not at natural slot")
            else:
                h = Counter(r["gold_slot"] for r in rows)
                if set(h.values()) != {n // K} or len(h) != K:
                    errors.append(f"{c}/{split}: slot histogram not uniform {dict(h)}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors-dir", type=Path, required=True)
    ap.add_argument("--infoseek-final", type=Path, required=True, help="data/train/final of the InfoSeek build")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pool", type=int, default=40000)
    ap.add_argument("--per-source-train", type=int, default=10000)
    ap.add_argument("--per-source-dev", type=int, default=250)
    ap.add_argument("--balanced-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    man_is = json.loads((args.infoseek_final / "manifest.json").read_text())
    for s in ("train", "dev"):
        if sha256(args.infoseek_final / "standard" / f"{s}.jsonl") != man_is[f"standard/{s}.jsonl"]:
            raise SystemExit(f"InfoSeek standard/{s}.jsonl does not match its manifest")

    sys.path.insert(0, str(args.authors_dir / "src"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("p20", args.authors_dir / "scripts/run_p20_evqa_replication.py")
    p20 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p20)
    image_dir = args.authors_dir / EVQA_IMAGE_ROOT
    assert image_dir == p20.IMAGE_DIR, (image_dir, p20.IMAGE_DIR)

    stats: dict = {"design": "see lae/build_mixed_data.py docstring", "seed": args.seed, "k": K, "retrieve_top": TOP,
                   "per_source_train": args.per_source_train, "per_source_dev": args.per_source_dev,
                   "infoseek_source_manifest": {k: v for k, v in man_is.items() if k.startswith("standard/")}}
    cands = phase_candidates(args, p20, args.out, stats)
    phase_images(p20, cands, image_dir, stats)
    done = phase_retrieve(args, p20, cands, image_dir, args.out, stats)
    items = evqa_items(args, cands, done, image_dir)
    phase_build(args, items, args.out, stats)
    errors = phase_check(args, args.out, stats)

    stats["wall_clock_min"] = round((time.time() - t_start) / 60, 1)
    stats["hard_check_errors"] = errors
    final = args.out / "final"
    (final / "build_stats.json").write_text(json.dumps(stats, indent=2, default=dict))
    manifest = {str(p.relative_to(final)): sha256(p) for p in sorted(final.rglob("*.jsonl"))}
    (final / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(json.dumps({k: v for k, v in stats.items() if k not in ("gold_slot_histograms", "infoseek_source_manifest")},
                   indent=1, default=dict))
    for name, h in stats["gold_slot_histograms"].items():
        log(f"  {name}: {h}")
    if errors:
        log(f"HARD CHECK ERRORS: {errors}")
        return 1
    log("all hard checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
