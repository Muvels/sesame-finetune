#!/usr/bin/env python3
"""
CLI-Tool: Gespräche säubern (Intro/Outro schneiden), in WAV konvertieren,
Deutsch lokal transkribieren (Whisper/faster-whisper) und Speaker-Diarization
(pyannote) durchführen. Sprecher bekommen konsistente IDs über mehrere Dateien
mittels persistenter Embedding-Datenbank. Ergebnis-JSON je Datei im gewünschten
Format:

  {
    "text": "…",
    "path": "/pfad/zur/clean_audio.wav",
    "start": 171.1,
    "end": 182.6,
    "speaker": 30
  }

— Hauptfeatures —
- MP3 ➜ WAV (16 kHz, mono, PCM s16le)
- Entfernt erste & letzte Minute (Standard 60s/60s, konfigurierbar)
- Deutsche Transkription lokal (faster-whisper, Model wählbar, Standard: large-v3)
- Diarisierung (pyannote), Zuweisung der Sprecher zu jedem Textsegment
- Speaker-ID-Konsistenz über mehrere Läufe durch Speaker-Embedding-DB (resemblyzer)
- Saubere JSON-Ausgabe + gepfadete, geschnittene WAV-Datei

— Installation —
Python 3.10+ empfohlen. Systemweit muss ffmpeg installiert sein (z. B. über
apt, brew, choco). Prüfe mit: `ffmpeg -version`.

pip-Pakete:
  pip install faster-whisper pyannote.audio resemblyzer librosa soundfile numpy tqdm

Hinweise:
- pyannote-Modelle benötigen initiales Herunterladen von Hugging Face. Lege
  ein Token in der Umgebungsvariable HF_TOKEN ab oder übergib --hf-token.
  Empfohlenes Pipeline-Repo: "pyannote/speaker-diarization-3.1".
- Whisper-Modelle (faster-whisper) werden beim ersten Lauf lokal zwischengespeichert.

— Beispiele —
  # Ordner verarbeiten (rekursiv), Datensätze anhängen
  python audio_clean_transcribe_cli.py audio_in \
    --output-dir out \
    --dataset-json ds/segments.json \
    --dataset-manifest ds/files.json

Die resultierende JSON liegt in --output-dir neben der geschnittenen WAV.
"""
from __future__ import annotations

import argparse
import json
import os
import hashlib
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

# ----------------------------- Utils ----------------------------------------

def run(cmd: List[str]) -> None:
    """Run a subprocess command and raise on failure with nice message."""
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"[ffmpeg] Fehler bei Befehl: {' '.join(cmd)}", file=sys.stderr)
        print(e.stderr.decode("utf-8", errors="ignore"), file=sys.stderr)
        raise


def which_or_die(exe: str) -> None:
    if shutil.which(exe) is None:
        print(f"Fehlt: {exe}. Bitte installieren und in PATH bereitstellen.", file=sys.stderr)
        sys.exit(2)


def ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1", str(path)
    ]
    out = subprocess.check_output(cmd)
    return float(out.decode("utf-8").strip())


def convert_to_wav(in_path: Path, out_path: Path, sr: int = 16000, mono: bool = True) -> None:
    """Convert input audio to PCM WAV (s16le), target sample rate and channels."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(in_path),
        "-ac", "1" if mono else "2",
        "-ar", str(sr),
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    run(cmd)


def trim_audio(input_wav: Path, output_wav: Path, trim_start_s: float, trim_end_s: float) -> Tuple[float, float, float]:
    """Trim first and last seconds using ffmpeg. Returns (orig_dur, start, end)."""
    dur = ffprobe_duration(input_wav)
    # Guard rails
    trim_start_s = max(0.0, trim_start_s)
    trim_end_s = max(0.0, trim_end_s)
    if trim_start_s + trim_end_s >= dur:
        # If trimming exceeds duration, produce a tiny 0.1s file to keep pipeline consistent
        print("Warnung: Trimmlängen >= Gesamtdauer. Erzeuge minimale Datei (0.1s).", file=sys.stderr)
        start = 0.0
        length = 0.1
    else:
        start = trim_start_s
        length = max(0.0, dur - trim_start_s - trim_end_s)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(input_wav),
        "-c:a", "copy", str(output_wav)
    ]
    # Using -c:a copy since it's already PCM; if not, ffmpeg will still cut frame-accurately enough.
    run(cmd)
    return dur, start, start + length

# ---- JSON helpers (for incremental dataset as JSON arrays) ----

def _load_json_array(path: Path) -> List:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):  # single dict → wrap
            return [data]
        return []
    except Exception:
        return []  # start fresh on invalid/empty file


def _save_json_array(path: Path, arr: List) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def append_json_array(path: Path, new_items: List[dict] | dict) -> None:
    arr = _load_json_array(path)
    if isinstance(new_items, list):
        arr.extend(new_items)
    else:
        arr.append(new_items)
    _save_json_array(path, arr)

# ---- Dedupe helpers ----

def _hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_processed_index(manifest_path: Optional[Path]) -> Tuple[set, set]:
    """Return (abs_path_set, sha256_set) of already processed files from manifest JSON array."""
    paths, hashes = set(), set()
    if manifest_path and manifest_path.exists():
        for rec in _load_json_array(manifest_path):
            p = rec.get("file") or rec.get("original")
            if p:
                try:
                    paths.add(str(Path(p).resolve()))
                except Exception:
                    paths.add(str(p))
            h = rec.get("file_sha256")
            if h:
                hashes.add(h)
    return paths, hashes


def iter_audio_files(input_dir: Path, recursive: bool = True) -> List[Path]:
    pattern = "**/*.mp3" if recursive else "*.mp3"
    return sorted(input_dir.glob(pattern))

# ---- Pretty timers & loading states ----

def _fmt_seconds(s: float) -> str:
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"

@contextmanager
def step(title: str):
    t0 = time.perf_counter()
    print(f"[•] {title} …", flush=True)
    try:
        yield
        dt = time.perf_counter() - t0
        print(f"[✓] {title} ({dt:.1f}s)")
    except Exception:
        dt = time.perf_counter() - t0
        print(f"[x] {title} fehlgeschlagen nach {dt:.1f}s", file=sys.stderr)
        raise

# ------------------------- Speaker DB (persistent) ---------------------------

@dataclass
class SpeakerEntry:
    numeric_id: int
    vector: List[float]
    count: int = 1


class SpeakerDB:
    def __init__(self, path: Path):
        self.path = path
        self.entries: List[SpeakerEntry] = []
        self.next_numeric_id: int = 1
        if path.exists():
            self._load()

    def _load(self):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.next_numeric_id = data.get("next_numeric_id", 1)
        self.entries = [SpeakerEntry(**e) for e in data.get("entries", [])]

    def save(self):
        data = {
            "next_numeric_id": self.next_numeric_id,
            "entries": [e.__dict__ for e in self.entries],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def match_or_add(self, emb: np.ndarray, threshold: float) -> int:
        """Return existing numeric_id if similarity >= threshold; else add new."""
        best_sim = -1.0
        best_idx = -1
        for i, e in enumerate(self.entries):
            sim = self._cosine(emb, np.asarray(e.vector, dtype=np.float32))
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        if best_sim >= threshold and best_idx >= 0:
            # update centroid by running average
            entry = self.entries[best_idx]
            new_vec = (np.asarray(entry.vector) * entry.count + emb) / (entry.count + 1)
            entry.vector = new_vec.astype(np.float32).tolist()
            entry.count += 1
            return entry.numeric_id
        # add new
        nid = self.next_numeric_id
        self.next_numeric_id += 1
        self.entries.append(SpeakerEntry(numeric_id=nid, vector=emb.astype(np.float32).tolist(), count=1))
        return nid

# ------------------------------ Diarization ---------------------------------

# Cache for pyannote pipeline
_PYANNOTE_PIPELINE = None

def torch_device(device: str) -> str:
    device = device.lower()
    if device in {"cpu", "cuda"}:
        return device
    if device.startswith("cuda"):
        return device
    return "cpu"


def get_pyannote_pipeline(pipeline_repo: str, hf_token: Optional[str], device: str):
    global _PYANNOTE_PIPELINE
    if _PYANNOTE_PIPELINE is not None:
        return _PYANNOTE_PIPELINE
    from pyannote.audio import Pipeline
    kwargs = {}
    if hf_token:
        kwargs["use_auth_token"] = hf_token
    with step(f"Lade Diarisierungs-Pipeline: {pipeline_repo}"):
        try:
            pipeline = Pipeline.from_pretrained(pipeline_repo, **kwargs)
        except Exception as e:
            raise RuntimeError(
                "Konnte pyannote-Pipeline nicht laden. Vermutlich fehlende Berechtigung/Token. "
                "Bitte akzeptiere die Bedingungen/zugelassenen Modelle auf Hugging Face (z.B. pyannote/segmentation-3.0) oder setze --hf-token.\n"
                f"Ursprünglicher Fehler: {e}"
            )
    # Move to device if possible
    with step(f"Verschiebe Pipeline auf {device}"):
        try:
            pipeline.to(torch_device(device))
        except Exception:
            try:
                pipeline.to(device)
            except Exception:
                pass
    _PYANNOTE_PIPELINE = pipeline
    return pipeline


def diarize_with_pyannote(wav_path: Path, hf_token: Optional[str], pipeline_repo: str, device: str = "cpu"):
    """Run pyannote diarization pipeline and return list of (start, end, label)."""
    pipeline = get_pyannote_pipeline(pipeline_repo, hf_token, device)
    with step(f"Diarisiere Audio ({wav_path.name})"):
        diarization = pipeline(str(wav_path))
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((float(turn.start), float(turn.end), str(speaker)))
    segments.sort(key=lambda x: (x[0], x[1]))
    return segments

# --------------------------- Speaker Embeddings ------------------------------

def speaker_embeddings_resemblyzer(wav_path: Path, speaker_segments: Dict[str, List[Tuple[float,float]]],
                                   max_seconds_per_speaker: float = 120.0) -> Dict[str, np.ndarray]:
    """Compute one embedding per speaker by concatenating up to N seconds of their segments."""
    import librosa
    from resemblyzer import VoiceEncoder, preprocess_wav

    # Load once (keep native sr; we'll resample per preprocess_wav)
    with step("Lade Audio für Embeddings"):
        y, sr = librosa.load(str(wav_path), sr=None, mono=True)

    embeddings: Dict[str, np.ndarray] = {}
    encoder = VoiceEncoder()

    for spk, segs in speaker_segments.items():
        total = 0.0
        pieces = []
        for (s, e) in segs:
            if total >= max_seconds_per_speaker:
                break
            s_idx = max(0, int(s * sr))
            e_idx = min(len(y), int(e * sr))
            if e_idx <= s_idx:
                continue
            wav_seg = y[s_idx:e_idx]
            seg_dur = (e_idx - s_idx) / sr
            pieces.append(wav_seg)
            total += seg_dur
        if not pieces:
            continue
        concat = np.concatenate(pieces)
        proc = preprocess_wav(concat, source_sr=sr)  # resampling etc.
        emb = encoder.embed_utterance(proc)
        embeddings[spk] = emb.astype(np.float32)
    return embeddings

# --------------------------- Transcription ----------------------------------

_WHISPER_MODEL = None

def get_whisper_model(model_size: str, device: str):
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL
    from faster_whisper import WhisperModel
    with step(f"Lade Whisper: {model_size} auf {device}"):
        _WHISPER_MODEL = WhisperModel(model_size, device=torch_device(device))
    return _WHISPER_MODEL


def transcribe_faster_whisper(wav_path: Path, device: str, model_size: str, language: str,
                              beam_size: int = 5) -> List[Dict]:
    model = get_whisper_model(model_size, device)
    # VAD filtering helps skip residual music/pauses
    with step(f"Transkribiere ({wav_path.name})"):
        segments, info = model.transcribe(str(wav_path), language=language, beam_size=beam_size, vad_filter=True)
        out: List[Dict] = []
        # progress by segment end time relative to total duration if available
        total = getattr(info, "duration", None) or 0
        last = 0.0
        pbar = None
        if total:
            pbar = tqdm(total=total, unit="s", desc="Transkription", leave=False)
        for seg in segments:
            out.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
            })
            if pbar is not None:
                inc = max(0.0, float(seg.end) - last)
                last = float(seg.end)
                pbar.update(inc)
        if pbar is not None:
            pbar.close()
    return out

# ------------------------- Alignment: who said what -------------------------

def assign_speaker_to_text(text_segments: List[Dict], diar_segments: List[Tuple[float,float,str]]) -> List[Dict]:
    """For each text segment, pick the diarization speaker with max temporal overlap."""
    diar_array = np.array([(s, e, i) for i, (s, e, lab) in enumerate(diar_segments)], dtype=float)
    labels = [lab for (_, _, lab) in diar_segments]

    def best_label(ts: float, te: float) -> str:
        best_overlap = -1.0
        best_lab = None
        for (s, e, idx) in diar_array:
            s = float(s); e = float(e)
            ov = max(0.0, min(te, e) - max(ts, s))
            if ov > best_overlap:
                best_overlap = ov
                best_lab = labels[int(idx)]
        return best_lab or "SPEAKER_00"

    enriched: List[Dict] = []
    for seg in text_segments:
        ts, te = seg["start"], seg["end"]
        lab = best_label(ts, te)
        enriched.append({**seg, "speaker_label": lab})
    return enriched

# ------------------------------ Main pipeline -------------------------------

def process_file(input_path: Path, args, speaker_db: SpeakerDB, original_sha: Optional[str] = None) -> Path:
    print(f"\n=== Verarbeite: {input_path.name} ===")
    file_t0 = time.perf_counter()

    # 1) MP3 ➜ WAV
    wav_path = args.output_dir / (input_path.stem + ".wav")
    with step("Konvertiere MP3 → WAV"):
        convert_to_wav(input_path, wav_path, sr=args.sr, mono=True)

    # 2) Trimmen (Intro/Outro entfernen)
    clean_wav = args.output_dir / (input_path.stem + ".clean.wav")
    with step(f"Schneide Intro/Outro (start={args.trim_start:.0f}s, end={args.trim_end:.0f}s)"):
        orig_dur, t0, t1 = trim_audio(wav_path, clean_wav, args.trim_start, args.trim_end)
    kept = t1 - t0
    print(f"   Dauer: original {_fmt_seconds(orig_dur)} → kept {_fmt_seconds(kept)}")

    # 3) Diarisierung
    with step("Diarisierung (Sprechertrennung)"):
        diar = diarize_with_pyannote(clean_wav, args.hf_token, args.pyannote_repo, device=args.device)

    # Map label -> list of (start, end)
    label2segs: Dict[str, List[Tuple[float,float]]] = {}
    for s, e, lab in diar:
        label2segs.setdefault(lab, []).append((s, e))

    # 4) Embeddings je Sprecher für DB (konsistente IDs über Dateien)
    with step("Berechne Sprecher-Embeddings"):
        emb_by_label = speaker_embeddings_resemblyzer(clean_wav, label2segs, max_seconds_per_speaker=args.embed_seconds)

    label2id: Dict[str, int] = {}
    with step("Mappe Sprecher auf stabile IDs"):
        for lab, emb in emb_by_label.items():
            sid = speaker_db.match_or_add(emb, threshold=args.similarity_threshold)
            label2id[lab] = sid

    # 5) Transkription (optional)
    items: List[Dict] = []
    if not args.no_transcribe:
        text_segments = transcribe_faster_whisper(clean_wav, device=args.device, model_size=args.whisper_model, language=args.language)
        # 6) Speaker zuweisen
        with step("Weise Sprecher-Labels den Textsegmenten zu"):
            enriched = assign_speaker_to_text(text_segments, diar)
        # 7) Output-Items bauen
        with step("Baue JSON-Segmente"):
            for seg in enriched:
                lab = seg.get("speaker_label", "SPEAKER_00")
                speaker_num = int(label2id.get(lab, -1))
                items.append({
                    "text": seg["text"],
                    "path": str(clean_wav.resolve()),
                    "start": round(float(seg["start"]), 3),
                    "end": round(float(seg["end"]), 3),
                    "speaker": speaker_num,
                })

    # 8) JSON speichern
    with step("Schreibe per-File JSON"):
        out_json = args.output_dir / f"{input_path.stem}.json"
        payload = {
            "audio": {
                "original": str(input_path.resolve()),
                "wav": str(wav_path.resolve()),
                "clean_wav": str(clean_wav.resolve()),
                "original_duration": round(orig_dur, 3),
                "kept_start": round(t0, 3),
                "kept_end": round(t1, 3),
                "sr": args.sr,
            },
            "speakers": [
                {"label": lab, "id": int(label2id.get(lab, -1))}
                for lab in sorted(label2id.keys())
            ],
            "segments": items,
        }
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 9) Optional: Segmente an ein zentrales JSON-Dataset (Array) anhängen
    if args.dataset_json and items:
        with step("Hänge Segmente an Dataset-JSON an"):
            append_json_array(args.dataset_json, items)

    # 10) Optional: Pro-Datei-Manifest (JSON-Array)
    if args.dataset_manifest:
        with step("Aktualisiere Manifest-JSON"):
            manifest_rec = {
                "file": str(input_path.resolve()),
                "file_sha256": original_sha,
                "clean_wav": str(clean_wav.resolve()),
                "json": str(out_json.resolve()),
                "speakers": [
                    {"label": lab, "id": int(label2id.get(lab, -1))}
                    for lab in sorted(label2id.keys())
                ],
                "duration_original": round(orig_dur, 3),
                "kept_start": round(t0, 3),
                "kept_end": round(t1, 3),
                "sr": args.sr,
            }
            append_json_array(args.dataset_manifest, manifest_rec)

    total_dt = time.perf_counter() - file_t0
    print(f"[✓] {input_path.name} fertig in {total_dt:.1f}s → {clean_wav.name}, {out_json.name}")
    return out_json

# ---------------------------------- CLI -------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ordner scannen, MP3 säubern, diarisiert transkribieren (DE) und JSON exportieren.")
    p.add_argument("input_dir", type=Path, help="Ordner mit MP3-Dateien")
    p.add_argument("--recursive", action="store_true", help="Rekursiv nach *.mp3 suchen")

    p.add_argument("--output-dir", type=Path, default=Path("out"), help="Ausgabeordner")

    p.add_argument("--trim-start", type=float, default=60.0, help="Sekunden am Anfang entfernen")
    p.add_argument("--trim-end", type=float, default=60.0, help="Sekunden am Ende entfernen")
    p.add_argument("--sr", type=int, default=16000, help="Ziel-Samplerate für WAV")

    p.add_argument("--no-transcribe", action="store_true", help="Nur schneiden & konvertieren (keine Transkription)")
    p.add_argument("--language", default="de", help="Sprachcode für Whisper (Standard: de)")
    p.add_argument("--whisper-model", default="large-v3", help="faster-whisper Modelgröße/Pfad (z. B. large-v3, medium, small)")
    p.add_argument("--device", default="cpu", help="cpu oder cuda[:index]")

    p.add_argument("--pyannote-repo", default="pyannote/speaker-diarization-3.1", help="HF Repo der Diarisierungs-Pipeline oder lokaler Pfad")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Hugging Face Token (oder via ENV HF_TOKEN)")

    p.add_argument("--speaker-db", type=Path, default=Path("out/speakers_db.json"), help="Persistente Speaker-DB (JSON)")
    p.add_argument("--similarity-threshold", type=float, default=0.80, help="Kosinus-Schwelle für Speaker-Matching über Dateien")
    p.add_argument("--embed-seconds", type=float, default=120.0, help="Max. Sekunden pro Sprecher für Embedding")

    # Inkrementelles Dataset als JSON-Arrays
    p.add_argument("--dataset-json", type=Path, required=True, help="Pfad zu einer JSON-Datei (Array), an die alle Segmente angehängt werden")
    p.add_argument("--dataset-manifest", type=Path, required=True, help="JSON (Array) mit einem Eintrag pro Datei (Metadaten & Speaker). Dient auch zum Deduplizieren.")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    which_or_die("ffmpeg")
    which_or_die("ffprobe")

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        print(f"Eingabeordner nicht gefunden: {args.input_dir}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    speaker_db = SpeakerDB(args.speaker_db)

    # Load existing index from manifest for dedupe
    print("[i] Lade Index aus Manifest …")
    seen_paths, seen_hashes = load_processed_index(args.dataset_manifest)

    # Gather candidates
    all_files = iter_audio_files(args.input_dir, recursive=args.recursive)
    if not all_files:
        print("Keine MP3-Dateien gefunden.", file=sys.stderr)
        return 1

    candidates: List[Tuple[Path, Optional[str]]] = []
    print(f"[i] Scanne {len(all_files)} Dateien …")
    for f in all_files:
        absf = str(f.resolve())
        if absf in seen_paths:
            tqdm.write(f"→ Überspringe (bereits im Manifest): {f.name}")
            continue
        try:
            sha = _hash_file(f)
        except Exception as e:
            tqdm.write(f"! Warnung: Hash fehlgeschlagen für {f}: {e}")
            sha = None
        if sha and sha in seen_hashes:
            tqdm.write(f"→ Überspringe (SHA bekannt): {f.name}")
            continue
        candidates.append((f, sha))

    if not candidates:
        print("Nichts zu tun – alle Dateien sind bereits im Dataset.")
        return 0

    processed = 0
    skipped = len(all_files) - len(candidates)

    with tqdm(total=len(candidates), desc="Dateien", unit="file") as pbar:
        for f, sha in candidates:
            try:
                process_file(f, args, speaker_db, original_sha=sha)
                # Save DB after each file to keep state even if later files fail
                speaker_db.save()
                processed += 1
            except Exception as e:
                tqdm.write(f"FEHLER bei {f}: {e}")
            finally:
                pbar.update(1)

    print(f"Fertig. Verarbeitet: {processed}, übersprungen: {skipped}, gesamt: {len(all_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
