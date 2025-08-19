#!/usr/bin/env python3
"""
Split a CSM-style manifest.jsonl into train/val/test with SPEAKER-LEVEL separation,
and write sesame-finetune compatible metadata as JSON arrays (not JSONL).

INPUT (each line in manifest.jsonl):
{
  "audio_path": "/abs/or/rel/path.wav",
  "text": "…",
  "speaker_id": "abc123",
  "lang": "de",
  "duration_secs": 3.472,
  "sample_rate": 24000
}

OUTPUT (per split, JSON array):
[
  {
    "path": "/absolute/path.wav",
    "text": "…",
    "speaker": 42,      # integer, via speaker_map.json
    "start": 0.0,       # optional; we set to 0.0
    "end": 3.472        # optional; we set to duration if present
  },
  ...
]

Also writes:
- speaker_map.json
- speakers_train.txt / speakers_val.txt / speakers_test.txt
- split_stats.json

Usage:
  python split_manifest_by_speaker.py \
    --manifest ./data_out/de/manifest.jsonl \
    --out-dir ./data_out/de/splits_sesame \
    --train 0.98 --val 0.01 --test 0.01 \
    --seed 42 \
    --skip-missing-files \
    --min-clip-secs 0.0 \
    --absolute-paths
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()
EPS = 1e-6


# -------- streaming JSON array writer (memory friendly) --------
class JSONStreamArray:
    def __init__(self, path: Path):
        self.f = path.open("w", encoding="utf-8")
        self.first = True
        self.f.write("[\n")

    def write(self, obj: dict):
        if not self.first:
            self.f.write(",\n")
        else:
            self.first = False
        json.dump(obj, self.f, ensure_ascii=False)

    def close(self):
        self.f.write("\n]\n")
        self.f.close()


def parse_args():
    ap = argparse.ArgumentParser(description="Speaker-level split -> sesame-finetune JSON metadata.")
    ap.add_argument("--manifest", type=Path, required=True, help="Path to manifest.jsonl")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory to write split files into")
    ap.add_argument("--train", type=float, default=0.98, help="Train ratio (default 0.98)")
    ap.add_argument("--val", type=float, default=0.01, help="Val ratio (default 0.01)")
    ap.add_argument("--test", type=float, default=0.01, help="Test ratio (default 0.01)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--skip-missing-files", action="store_true",
                    help="Skip entries whose audio_path is missing on disk when writing splits")
    ap.add_argument("--min-clip-secs", type=float, default=0.0,
                    help="Ignore utterances shorter than this when computing/writing (default 0.0)")
    ap.add_argument("--absolute-paths", action="store_true",
                    help="Resolve audio_path to absolute paths in metadata (recommended)")
    return ap.parse_args()


def validate_ratios(train: float, val: float, test: float):
    s = train + val + test
    if abs(s - 1.0) > 1e-4:
        console.print(f"[red]ERROR:[/red] ratios must sum to 1.0 (got {s:.6f})")
        sys.exit(1)
    if min(train, val, test) < 0:
        console.print("[red]ERROR:[/red] ratios cannot be negative.")
        sys.exit(1)


def pass1_scan(manifest_path: Path, min_clip_secs: float) -> Tuple[Dict[str, float], float, int, List[str]]:
    """Return (secs_per_speaker, total_secs, used_utt_count, sorted_unique_speakers)."""
    secs_per_speaker: Dict[str, float] = defaultdict(float)
    total_secs = 0.0
    used = 0
    speakers_set = set()

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            spk = str(rec.get("speaker_id", "unknown"))
            d = float(rec.get("duration_secs", 0.0) or 0.0)
            if d < min_clip_secs:
                continue
            secs_per_speaker[spk] += d
            total_secs += d
            used += 1
            speakers_set.add(spk)

    speakers = sorted(speakers_set)  # stable order → stable integer mapping
    return secs_per_speaker, total_secs, used, speakers


def build_speaker_map(speakers: List[str]) -> Dict[str, int]:
    """Map string speaker ids to consecutive integers [0..N-1] in a stable, deterministic way."""
    return {spk: i for i, spk in enumerate(speakers)}


def choose_speakers(secs_per_speaker: Dict[str, float], total_secs: float,
                    val_ratio: float, test_ratio: float, seed: int) -> Tuple[set, set, set]:
    """Return (train_spk, val_spk, test_spk) sets, targeting HOURS by greedy fill."""
    speakers = list(secs_per_speaker.items())  # (spk, secs)
    rnd = random.Random(seed)
    rnd.shuffle(speakers)

    target_test = total_secs * test_ratio
    target_val = total_secs * val_ratio

    test_spk, val_spk = set(), set()
    sum_test = 0.0
    sum_val = 0.0

    for spk, secs in speakers:
        if sum_test < target_test - EPS:
            test_spk.add(spk); sum_test += secs
        elif sum_val < target_val - EPS:
            val_spk.add(spk); sum_val += secs
        # leftover → train

    train_spk = {spk for spk, _ in speakers} - test_spk - val_spk

    if not test_spk and speakers:
        test_spk.add(speakers[0][0]); train_spk.discard(speakers[0][0])
    if not val_spk and len(speakers) > 1:
        val_spk.add(speakers[1][0]); train_spk.discard(speakers[1][0])

    return train_spk, val_spk, test_spk


def pass2_write_metadata(manifest_path: Path, out_dir: Path,
                         train_spk: set, val_spk: set, test_spk: set,
                         spk_map: Dict[str, int],
                         skip_missing: bool, min_clip_secs: float,
                         make_abs: bool) -> Dict[str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {
        "train": out_dir / "train_metadata.json",
        "val":   out_dir / "val_metadata.json",
        "test":  out_dir / "test_metadata.json",
    }
    writers = {k: JSONStreamArray(out_paths[k]) for k in out_paths}

    stats = {
        "train": {"clips": 0, "secs": 0.0, "speakers": len(train_spk)},
        "val":   {"clips": 0, "secs": 0.0, "speakers": len(val_spk)},
        "test":  {"clips": 0, "secs": 0.0, "speakers": len(test_spk)},
    }

    # Write speaker lists
    for split, spkset in [("train", train_spk), ("val", val_spk), ("test", test_spk)]:
        with (out_dir / f"speakers_{split}.txt").open("w", encoding="utf-8") as f:
            for spk in sorted(spkset):
                f.write(spk + "\n")

    # Stream manifest → write sesame-finetune rows
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            spk = str(rec.get("speaker_id", "unknown"))
            d = float(rec.get("duration_secs", 0.0) or 0.0)
            if d < min_clip_secs:
                continue

            ap = rec.get("audio_path", "")
            txt = rec.get("text", "")
            if not ap or not txt:
                continue

            p = Path(ap)
            if make_abs:
                p = p if p.is_absolute() else (Path.cwd() / p).resolve()

            if skip_missing and not p.exists():
                continue

            if   spk in train_spk: split = "train"
            elif spk in val_spk:   split = "val"
            elif spk in test_spk:  split = "test"
            else:                  split = "train"  # fallback

            meta = {
                "path": str(p),
                "text": txt,
                "speaker": int(spk_map.get(spk, -1)),
                "start": 0.0,
                "end": d if d > 0 else None,
            }
            if meta["end"] is None:
                del meta["end"]

            writers[split].write(meta)
            stats[split]["clips"] += 1
            stats[split]["secs"] += d

    for w in writers.values():
        w.close()

    # Stats file
    total_secs = sum(s["secs"] for s in stats.values())
    total_clips = sum(s["clips"] for s in stats.values())
    summary = {
        "totals": {"clips": total_clips, "secs": total_secs, "hours": total_secs / 3600.0},
        "splits": {k: {"clips": v["clips"], "secs": v["secs"], "hours": v["secs"]/3600.0, "speakers": v["speakers"]}
                   for k, v in stats.items()}
    }
    with (out_dir / "split_stats.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def pretty_report(summary: dict, secs_per_speaker: Dict[str, float]):
    tbl = Table(box=box.SIMPLE_HEAVY)
    tbl.add_column("Split", style="bold")
    tbl.add_column("Speakers", justify="right")
    tbl.add_column("Clips", justify="right")
    tbl.add_column("Hours", justify="right")
    for split in ["train","val","test"]:
        s = summary["splits"][split]
        tbl.add_row(split, f'{s["speakers"]:,}', f'{s["clips"]:,}', f'{s["hours"]:,.2f}')
    totals = summary["totals"]
    tbl.add_row("—", "—", "—", "—")
    tbl.add_row("TOTAL", f'{len(secs_per_speaker):,}', f'{totals["clips"]:,}', f'{totals["hours"]:,.2f}')
    console.print(Panel(tbl, title="[cyan]Speaker-level split → sesame JSON metadata complete[/cyan]", border_style="cyan"))


def main():
    args = parse_args()
    validate_ratios(args.train, args.val, args.test)

    if not args.manifest.exists():
        console.print(f"[red]ERROR:[/red] manifest not found: {args.manifest}")
        sys.exit(1)

    console.print(f"[bold]Pass 1[/bold]: scanning speakers & durations in {args.manifest} ...")
    secs_per_speaker, total_secs, used_utts, speakers = pass1_scan(args.manifest, args.min_clip_secs)
    if total_secs <= EPS or not secs_per_speaker:
        console.print("[red]ERROR:[/red] No usable data found (check paths/durations).")
        sys.exit(1)

    console.print(f" - Speakers: {len(secs_per_speaker):,}")
    console.print(f" - Utterances used: {used_utts:,}")
    console.print(f" - Total hours: {total_secs/3600.0:,.2f} h")

    # Build stable speaker map and save for reproducibility
    spk_map = build_speaker_map(speakers)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "speaker_map.json").open("w", encoding="utf-8") as f:
        json.dump(spk_map, f, ensure_ascii=False, indent=2)

    # Assign speakers to splits
    train_spk, val_spk, test_spk = choose_speakers(
        secs_per_speaker, total_secs, args.val, args.test, args.seed
    )
    # Safety check
    assert len(train_spk | val_spk | test_spk) == len(secs_per_speaker)

    console.print(f"[bold]Pass 2[/bold]: writing sesame-finetune JSON metadata to {args.out_dir} ...")
    summary = pass2_write_metadata(
        args.manifest, args.out_dir, train_spk, val_spk, test_spk,
        spk_map, args.skip_missing_files, args.min_clip_secs, args.absolute_paths
    )

    pretty_report(summary, secs_per_speaker)

    console.print("\nNext:")
    console.print("  python pretokenize.py \\")
    console.print("    --train_data {}/train_metadata.json \\".format(args.out_dir))
    console.print("    --val_data   {}/val_metadata.json   \\".format(args.out_dir))
    console.print("    --output     {}/tokenized.hdf5".format(args.out_dir))


if __name__ == "__main__":
    main()
