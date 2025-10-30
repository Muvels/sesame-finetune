"""
Script to pre-tokenize training/validation data for Sesame finetuning and save incrementally in HDF5.

Now supports *context refs* (zero-shot voice cloning prompts):
Each JSON row may include:
{
  "text": "...", "path": "...", "start": 1.23, "end": 4.56, "speaker": 0,
  "refs": [
    {"path": "ref1.wav", "text": "optional transcript", "start": 0.0, "end": 2.0},
    {"path": "ref2.wav", "text": "..."}
  ]
}

Refs are concatenated BEFORE the target segment in both text and audio.
We also store a frame-level mask `frame_loss_mask` (0 for refs, 1 for target) so the trainer can
exclude ref frames from the audio loss while still using them as conditioning context.
"""

import argparse
from pathlib import Path
import sqlite3
import pandas as pd
import torch
import torchaudio
import h5py
import numpy as np
from tqdm import tqdm

from utils import load_tokenizers, MIMI_SAMPLE_RATE, AUDIO_NUM_CODEBOOKS


def parse_args(arg_string=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=Path, required=True)
    parser.add_argument("--val_data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default="./data/tokens.hdf5")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--audio_root", type=Path, default=None, help="Optional root to resolve relative audio paths")
    parser.add_argument("--save_every", type=int, default=100, help="Save every N samples")
    parser.add_argument("--omit_speaker_id", action="store_true", help="Don't prepend text with a speaker id")
    args = parser.parse_args(arg_string.split() if arg_string else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return args


def load_metadata(data_path: Path | str) -> pd.DataFrame:
    """
    Load metadata from various formats.
    """
    if isinstance(data_path, str):
        data_path = Path(data_path)

    if data_path.suffix == ".json":
        return pd.read_json(data_path)
    elif data_path.suffix == ".csv":
        return pd.read_csv(data_path)
    elif data_path.suffix == ".sql":
        return pd.read_sql_query("SELECT * FROM data", sqlite3.connect(data_path))
    elif data_path.suffix == ".parquet":
        return pd.read_parquet(data_path)
    elif data_path.suffix == ".pkl":
        return pd.read_pickle(data_path)
    else:
        raise NotImplementedError(f"Unsupported file format: {data_path}")


def append_to_hdf5(file_path, split, audio_tokens_batch, text_tokens_batch, frame_loss_masks_batch=None):
    """
    Append audio, text, frame_loss_mask, and length information to the HDF5 file.
    Audio is flattened (vlen) for space efficiency.
    """
    with h5py.File(file_path, "a") as f:
        grp = f.require_group(split)

        vlen_i32 = h5py.special_dtype(vlen=np.int32)
        vlen_i8  = h5py.special_dtype(vlen=np.int8)

        audio_ds  = grp.get("audio")  or grp.create_dataset("audio",  shape=(0,), maxshape=(None,), dtype=vlen_i32)
        text_ds   = grp.get("text")   or grp.create_dataset("text",   shape=(0,), maxshape=(None,), dtype=vlen_i32)
        flm_ds    = grp.get("frame_loss_mask") or grp.create_dataset("frame_loss_mask", shape=(0,), maxshape=(None,), dtype=vlen_i8)
        length_ds = grp.get("length") or grp.create_dataset("length", shape=(0,), maxshape=(None,), dtype=np.int32)

        n = len(audio_tokens_batch)
        audio_ds.resize(audio_ds.shape[0] + n, axis=0)
        text_ds.resize(text_ds.shape[0] + n, axis=0)
        flm_ds.resize(flm_ds.shape[0] + n, axis=0)
        length_ds.resize(length_ds.shape[0] + n, axis=0)

        for i in range(n):
            audio_array = np.array(audio_tokens_batch[i], dtype=np.int32).flatten()  # [n_codebooks * seq_len]
            text_array  = np.array(text_tokens_batch[i],  dtype=np.int32)

            # Use provided frame loss mask or default to "all predicted"
            if frame_loss_masks_batch is not None:
                flm_array = np.array(frame_loss_masks_batch[i], dtype=np.int8)
            else:
                flm_array = np.ones(audio_array.shape[0] // AUDIO_NUM_CODEBOOKS, dtype=np.int8)

            # Sanity: frame_loss_mask length must match number of audio frames
            seq_len_frames = audio_array.shape[0] // AUDIO_NUM_CODEBOOKS
            if flm_array.shape[0] != seq_len_frames:
                raise ValueError(f"frame_loss_mask length {flm_array.shape[0]} != audio frames {seq_len_frames}")

            total_len = seq_len_frames + len(text_array) + 1  # +1 for EOS frame

            audio_ds[-n + i] = audio_array
            text_ds[-n + i]  = text_array
            flm_ds[-n + i]   = flm_array
            length_ds[-n + i] = total_len



def _resolve_audio_path(path_str, base_dir: Path, audio_root: Path | None):
    P = Path(path_str)
    # 1) absolute
    if P.is_file():
        return str(P)
    # 2) relative to audio_root if provided
    if audio_root is not None:
        Q = (audio_root / P).expanduser()
        if Q.is_file():
            return str(Q)
    # 3) relative to dataset directory
    R = (base_dir / P).expanduser()
    if R.is_file():
        return str(R)
    # 4) as-is (torchaudio will throw and we catch upstream)
    return str(P)


def get_num_existing_samples(file_path, split):
    """Return the number of existing samples in the HDF5 file for the given split, using the 'length' dataset."""
    try:
        with h5py.File(file_path, "r") as f:
            return f[split]["length"].shape[0]
    except Exception:
        return 0


def _load_slice_to_24k_device(path, device, start_s=None, end_s=None):
    """Load a (start, end) slice and resample to Mimi's sample rate, shaped [1,1,T]."""
    info = torchaudio.info(path)
    sr = info.sample_rate

    has_start = start_s is not None and not pd.isna(start_s)
    has_end = end_s is not None and not pd.isna(end_s)

    if has_start or has_end:
        s0 = float(start_s) if has_start else 0.0
        frame_offset = int(round(max(0.0, s0) * sr))
        if has_end:
            span_s = max(0.0, float(end_s) - s0)
            num_frames = int(round(span_s * sr)) if span_s > 0 else -1
        else:
            num_frames = -1
    else:
        frame_offset = 0
        num_frames = -1

    wav, sr_loaded = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
    # Mix to mono if multi-channel
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # Ensure shape [1, T]
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    wav = torchaudio.functional.resample(wav, orig_freq=sr_loaded, new_freq=MIMI_SAMPLE_RATE)
    # Return [B=1, C=1, T]
    return wav.unsqueeze(0).to(device)


def _encode_audio(audio_tokenizer, wav_1x1T):
    """audio_tokenizer.encode -> tensor [1, n_codebooks, T]; return list [n_codebooks, T]."""
    return audio_tokenizer.encode(wav_1x1T)[0].tolist()


def tokenize_and_store(data_path, output_path, split, audio_tokenizer, text_tokenizer, device, save_every=100, omit_speaker_id=False, audio_root: Path | None=None):
    """
    Tokenize the dataset and save in HDF5 incrementally, resuming if interrupted.
    Supports refs/segments and writes frame_loss_mask.
    """
    df = load_metadata(data_path)
    base_dir = Path(data_path).parent if not isinstance(data_path, Path) else data_path.parent
    n_existing = get_num_existing_samples(output_path, split)
    if n_existing:
        print(f"⏩ Resuming {split}: skipping {n_existing} already processed samples")
        df = df.iloc[n_existing:]
    else:
        print(f"🔄 Processing {split} split: {len(df)} samples")

    audio_tokens_batch, text_tokens_batch, frame_loss_masks_batch = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Speaker handling
        speaker_val = row["speaker"] if ("speaker" in df.columns and not pd.isna(row["speaker"])) else 999
        spk_prefix = f"[{int(speaker_val)}]" if not omit_speaker_id else ""

        segments = []  # list of dict(text_ids, audio_tokens, is_context)

        # Refs first (if any)
        refs = row.get("refs", [])
        if isinstance(refs, (list, tuple)):
            for r in refs:
                if not isinstance(r, dict) or "path" not in r:
                    continue
                r_text  = r.get("text", "") or ""
                r_start = r.get("start", None)
                r_end   = r.get("end", None)
                try:
                    ref_path = _resolve_audio_path(r["path"], base_dir, audio_root)
                    wav = _load_slice_to_24k_device(ref_path, device, r_start, r_end)
                    a_tok = _encode_audio(audio_tokenizer, wav)
                except Exception as e:
                    print(f"⚠️  Skipping ref due to error loading/encoding {r.get('path')}: {e}")
                    continue
                t_tok = text_tokenizer.encode(spk_prefix + r_text) if r_text else []
                segments.append({"text_ids": t_tok, "audio_tokens": a_tok, "is_context": True})

        # Target segment (required)
        try:
            target_path = _resolve_audio_path(row["path"], base_dir, audio_root)
            wav = _load_slice_to_24k_device(
                target_path,
                device,
                row.get("start", None) if "start" in df.columns else None,
                row.get("end", None) if "end" in df.columns else None,
            )
            a_tok = _encode_audio(audio_tokenizer, wav)
        except Exception as e:
            print(f"⚠️  Skipping row due to error loading/encoding {row.get('path')}: {e}")
            continue

        t_tok = text_tokenizer.encode(spk_prefix + str(row["text"]))
        segments.append({"text_ids": t_tok, "audio_tokens": a_tok, "is_context": False})

        # ---- Concatenate across segments ----
        # text
        all_text_ids = []
        for seg in segments:
            if len(seg["text_ids"]) > 0:
                all_text_ids.extend(seg["text_ids"])
                # add a newline as light-weight separator
                all_text_ids.extend(text_tokenizer.encode("\n"))
        if len(all_text_ids) and all_text_ids[-1] == text_tokenizer.encode("\n")[-1]:
            all_text_ids = all_text_ids[:-1]  # drop trailing sep

        # audio + frame_loss_mask
        audio_list = []
        flm_list = []
        for seg in segments:
            a = np.array(seg["audio_tokens"], dtype=np.int32)  # [n_codebooks, T]
            audio_list.append(a)
            T = a.shape[1]
            flm_list.append(np.zeros(T, dtype=np.int8) if seg["is_context"] else np.ones(T, dtype=np.int8))

        if len(audio_list) == 0:
            continue

        audio_concat = np.concatenate(audio_list, axis=1)  # [n_codebooks, sum_T]
        flm_concat   = np.concatenate(flm_list, axis=0)    # [sum_T]

        # Accumulate batch
        audio_tokens_batch.append(audio_concat)
        text_tokens_batch.append(np.array(all_text_ids, dtype=np.int32))
        frame_loss_masks_batch.append(flm_concat)

        if len(audio_tokens_batch) >= save_every:
            append_to_hdf5(output_path, split, audio_tokens_batch, text_tokens_batch, frame_loss_masks_batch)
            audio_tokens_batch, text_tokens_batch, frame_loss_masks_batch = [], [], []

    # Final flush
    if audio_tokens_batch:
        append_to_hdf5(output_path, split, audio_tokens_batch, text_tokens_batch, frame_loss_masks_batch)


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)

    text_tokenizer, audio_tokenizer = load_tokenizers(device)

    tokenize_and_store(
        args.train_data, output_path=args.output, split="train",
        audio_tokenizer=audio_tokenizer, text_tokenizer=text_tokenizer,
        device=device, save_every=args.save_every, omit_speaker_id=args.omit_speaker_id, audio_root=args.audio_root
    )

    tokenize_and_store(
        args.val_data, output_path=args.output, split="val",
        audio_tokenizer=audio_tokenizer, text_tokenizer=text_tokenizer,
        device=device, save_every=args.save_every, omit_speaker_id=args.omit_speaker_id, audio_root=args.audio_root
    )

    print(f"\\n✅ Done. Tokenized data (with refs + frame_loss_mask) saved to: {args.output}")
