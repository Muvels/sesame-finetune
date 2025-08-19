#!/usr/bin/env python3
"""
Prepare Mozilla Common Voice (Delta 22.0) audio for Sesame CSM-1B fine-tuning.

What it does
------------
1) Scans <cv_delta_root>/<lang>/validated.tsv
2) Converts audio -> 24 kHz, mono, 16-bit PCM WAV (good for CSM-1B)
   - Loudness normalization (EBU R128): I=-23, TP=-2, LRA=11
   - Optional edge-silence trimming
3) Writes/updates a manifest.jsonl under <out_root>/<lang>/ with:
   {
     "audio_path": "/abs/path/to.wav",
     "text": "Exact transcript",
     "speaker_id": "client_id",
     "lang": "de",
     "duration_secs": 3.472,
     "sample_rate": 24000
   }

Features
--------
- Parallel conversion (ThreadPoolExecutor)
- Resume-safe: skips already converted files and already-manifested items
- Duration caching via ffprobe to speed up re-runs
- Can process a single language (e.g. --lang de) or all with --lang auto
- Optional manifest sharding (e.g., --shard-max-rows 200000)

Usage
-----
python prepare_csm_data.py \
  --cv-delta-root ./cv-corpus-22.0-delta-2025-06-20 \
  --out-root ./data_out \
  --lang de \
  --num-workers 8 \
  --trim-silence

# Process all locales in the delta folder:
python prepare_csm_data.py --cv-delta-root ./cv-corpus-22.0-delta-2025-06-20 --out-root ./data_out --lang auto

Notes
-----
- Install ffmpeg/ffprobe on your system (brew/apt).
- WAV @ 24 kHz mono PCM is the safest target for CSM-1B fine-tuning.
"""

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich import box

console = Console()


# -------------------- helpers --------------------

def which(cmd: str) -> Optional[str]:
    from shutil import which as _which
    return _which(cmd)

def ensure_ffmpeg():
    if which("ffmpeg") is None or which("ffprobe") is None:
        console.print("[red]ERROR[/red]: ffmpeg/ffprobe not found in PATH. Please install ffmpeg.")
        sys.exit(1)

def macos_unquarantine(path: Path):
    """Best-effort remove com.apple.quarantine recursively (no-op on non-macOS)."""
    try:
        subprocess.run(["xattr", "-rd", "com.apple.quarantine", str(path)], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def run(cmd_list: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr

def ffprobe_duration_seconds(audio_path: Path) -> Optional[float]:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=nw=1:nk=1", str(audio_path)]
    rc, out, _ = run(cmd)
    if rc != 0:
        return None
    try:
        return float(out.strip())
    except Exception:
        return None

def read_validated_tsv(tsv_path: Path) -> Iterable[dict]:
    import csv
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row

def find_audio(clips_dir: Path, rel: str) -> Optional[Path]:
    p = clips_dir / rel
    if p.exists():
        return p
    # try extension swaps
    for ext in (".mp3", ".wav", ".flac", ".m4a", ".ogg"):
        cand = p.with_suffix(ext)
        if cand.exists():
            return cand
    # try by basename search
    base = Path(rel).stem
    for ext in (".mp3", ".wav", ".flac", ".m4a", ".ogg"):
        cand = clips_dir / f"{base}{ext}"
        if cand.exists():
            return cand
    return None


# -------------------- conversion --------------------

@dataclass
class ConvertJob:
    src: Path
    dst: Path
    text: str
    speaker_id: str

def build_ffmpeg_cmd(src: Path, dst: Path, loudnorm: bool, trim_silence: bool) -> List[str]:
    # WAV 24k mono 16-bit PCM
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(src), "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le"]
    filters: List[str] = []
    if loudnorm:
        filters.append("loudnorm=I=-23:TP=-2:LRA=11")
    if trim_silence:
        # trim short edge silences; keep natural micro-pauses
        filters.append("silenceremove=start_periods=1:start_duration=0.1:start_threshold=-35dB:"
                       "stop_periods=1:stop_duration=0.2:stop_threshold=-35dB")
    if filters:
        base.extend(["-af", ",".join(filters)])
    base.append(str(dst))
    return base

def convert_one(job: ConvertJob, loudnorm: bool, trim_silence: bool, force: bool) -> Tuple[ConvertJob, bool, str]:
    """Return (job, ok, err_msg)."""
    try:
        if job.dst.exists() and not force:
            return (job, True, "")
        job.dst.parent.mkdir(parents=True, exist_ok=True)
        cmd = build_ffmpeg_cmd(job.src, job.dst, loudnorm=loudnorm, trim_silence=trim_silence)
        rc, _, err = run(cmd)
        if rc != 0:
            if job.dst.exists():
                try: job.dst.unlink()
                except Exception: pass
            return (job, False, err.strip())
        return (job, True, "")
    except Exception as e:
        return (job, False, str(e))


# -------------------- manifest & caching --------------------

def load_duration_cache(cache_path: Path) -> Dict[str, float]:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_duration_cache(cache_path: Path, cache: Dict[str, float]):
    try:
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    except Exception:
        pass

def existing_manifest_audio_paths(manifest_path: Path) -> set:
    s = set()
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    ap = rec.get("audio_path", "")
                    if ap:
                        s.add(ap)
                except Exception:
                    pass
    return s

def write_manifest_lines(manifest_path: Path, records: List[dict], append: bool = True):
    mode = "a" if append and manifest_path.exists() else "w"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open(mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------- core pipeline --------------------

def process_language(
    cv_delta_root: Path,
    out_root: Path,
    lang: str,
    num_workers: int,
    loudnorm: bool,
    trim_silence: bool,
    force: bool,
    shard_max_rows: int
):
    lang_root = cv_delta_root / lang
    clips_dir = lang_root / "clips"
    tsv = lang_root / "validated.tsv"

    if not lang_root.exists():
        console.print(f"[red]ERROR[/red]: language folder not found: {lang_root}")
        sys.exit(1)
    if not clips_dir.exists():
        console.print(f"[red]ERROR[/red]: clips not found: {clips_dir}")
        sys.exit(1)
    if not tsv.exists():
        console.print(f"[red]ERROR[/red]: validated.tsv not found: {tsv}")
        sys.exit(1)

    # Output dirs
    out_lang = out_root / lang
    out_wav = out_lang / "wav"
    out_lang.mkdir(parents=True, exist_ok=True)
    out_wav.mkdir(parents=True, exist_ok=True)
    manifest_path = out_lang / "manifest.jsonl"
    cache_path = out_lang / ".duration_cache.json"

    # macOS: remove quarantine bit to avoid dialogs
    macos_unquarantine(cv_delta_root)

    # Build job list
    jobs: List[ConvertJob] = []
    missing = 0
    rows = 0
    for row in read_validated_tsv(tsv):
        rows += 1
        sentence = (row.get("sentence") or "").strip()
        rel = (row.get("path") or row.get("clip") or "").strip()
        speaker_id = (row.get("client_id") or "unknown").strip() or "unknown"
        if not sentence or not rel:
            continue
        src = find_audio(clips_dir, rel)
        if src is None:
            missing += 1
            continue
        dst = out_wav / Path(rel).with_suffix(".wav")
        jobs.append(ConvertJob(src=src, dst=dst, text=sentence, speaker_id=speaker_id))

    if not jobs:
        console.print(f"[yellow]No convertible items for {lang}[/yellow]")
        return

    # Parallel convert
    ok_count = 0
    fail_count = 0
    failures: List[Tuple[ConvertJob, str]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
        t = progress.add_task(f"Converting {lang} MP3/FLAC/etc → WAV @ 24 kHz...", total=len(jobs))
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futs = [ex.submit(convert_one, j, loudnorm, trim_silence, force) for j in jobs]
            for fut in as_completed(futs):
                job, ok, err = fut.result()
                if ok:
                    ok_count += 1
                else:
                    fail_count += 1
                    failures.append((job, err))
                progress.advance(t, 1)

    # Duration probe (with cache) + manifest write (sharded if requested)
    dur_cache = load_duration_cache(cache_path)
    existing = existing_manifest_audio_paths(manifest_path)

    records: List[dict] = []
    total_secs = 0.0
    added = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
        t = progress.add_task(f"Building manifest for {lang}…", total=len(jobs))
        for job in jobs:
            progress.advance(t, 1)
            if not job.dst.exists():
                continue
            ap = str(job.dst.resolve())
            if ap in existing and not force:
                # already in manifest; we still roll its duration into total if cached
                d = dur_cache.get(ap)
                if d is None:
                    d = ffprobe_duration_seconds(job.dst) or 0.0
                    dur_cache[ap] = d
                total_secs += d
                continue

            d = dur_cache.get(ap)
            if d is None or force:
                d = ffprobe_duration_seconds(job.dst) or 0.0
                dur_cache[ap] = d
            total_secs += d
            rec = {
                "audio_path": ap,
                "text": job.text,
                "speaker_id": job.speaker_id,
                "lang": lang,
                "duration_secs": round(d, 3),
                "sample_rate": 24000,
            }
            records.append(rec)

            # shard if too large
            if shard_max_rows > 0 and len(records) >= shard_max_rows:
                write_manifest_lines(manifest_path, records, append=True)
                existing.update([r["audio_path"] for r in records])
                added += len(records)
                records.clear()

    if records:
        write_manifest_lines(manifest_path, records, append=True)
        existing.update([r["audio_path"] for r in records])
        added += len(records)

    save_duration_cache(cache_path, dur_cache)

    # Summary
    hrs = total_secs / 3600.0
    tbl = Table(box=box.SIMPLE_HEAVY)
    tbl.add_column("Metric", style="bold")
    tbl.add_column("Value", justify="right")
    tbl.add_row("Locale", lang)
    tbl.add_row("Rows in TSV", f"{rows:,}")
    tbl.add_row("Missing source files", f"{missing:,}")
    tbl.add_row("Converted OK", f"{ok_count:,}")
    tbl.add_row("Failed converts", f"{fail_count:,}")
    tbl.add_row("Manifest appended", f"{added:,} new rows")
    tbl.add_row("Total hours (from cache+probe)", f"{hrs:,.2f} h")
    console.print(Panel(tbl, title=f"[cyan]Finished {lang}[/cyan] → {manifest_path}", border_style="cyan"))

    if failures:
        console.print("\n[yellow]Some files failed to convert (showing up to 10):[/yellow]")
        for (job, err) in failures[:10]:
            console.print(f" - {job.src.name}: {err}")


def find_locales(cv_delta_root: Path) -> List[str]:
    langs = []
    for child in cv_delta_root.iterdir():
        if child.is_dir() and (child / "validated.tsv").exists() and (child / "clips").exists():
            langs.append(child.name)
    return sorted(langs)


# -------------------- CLI --------------------

def main():
    ensure_ffmpeg()
    ap = argparse.ArgumentParser(description="Convert Common Voice audio and build a CSM-ready manifest.")
    ap.add_argument("--cv-delta-root", type=Path, required=True,
                    help="Path to cv-corpus-22.0-delta-YYYY-MM-DD")
    ap.add_argument("--out-root", type=Path, default=Path("./data_out"),
                    help="Output root for WAV + manifest")
    ap.add_argument("--lang", type=str, default="de",
                    help="Locale (e.g., de, en). Use 'auto' to process all found locales.")
    ap.add_argument("--num-workers", type=int, default=os.cpu_count() or 8,
                    help="Parallel conversions")
    ap.add_argument("--trim-silence", action="store_true",
                    help="Trim leading/trailing silence")
    ap.add_argument("--no-loudnorm", action="store_true",
                    help="Disable loudness normalization (enabled by default)")
    ap.add_argument("--force", action="store_true",
                    help="Re-convert and re-probe even if outputs/manifests exist")
    ap.add_argument("--shard-max-rows", type=int, default=0,
                    help="If >0, periodically flush manifest after N rows (useful for huge sets)")
    args = ap.parse_args()

    if args.lang == "auto":
        langs = find_locales(args.cv_delta_root)
        if not langs:
            console.print("[yellow]No locales found (missing validated.tsv/clips).[/yellow]")
            sys.exit(0)
    else:
        langs = [args.lang]

    for lg in langs:
        process_language(
            cv_delta_root=args.cv_delta_root,
            out_root=args.out_root,
            lang=lg,
            num_workers=args.num_workers,
            loudnorm=(not args.no_loudnorm),
            trim_silence=args.trim_silence,
            force=args.force,
            shard_max_rows=args.shard_max_rows
        )

if __name__ == "__main__":
    main()
