# infer_tts.py
import argparse
from pathlib import Path
import math
import numpy as np
import torch

# use your project utilities
from utils import (
    load_model,
    load_tokenizers,
    generate_audio,
    load_watermarker,
    MIMI_SAMPLE_RATE,
)

def pick_device():
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_checkpoint(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Saved during training: {"model": state_dict, "config": ..., ...}
    if "model" in state:
        sd = state["model"]
        cfg = state.get("config", {})
    else:
        # in case a raw state_dict was saved
        sd = state
        cfg = {}
    return sd, cfg

def save_wav(path: Path, y: np.ndarray, sr: int):
    """Save float32 [-1,1] audio. Tries soundfile; falls back to wave."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf  # pip install soundfile  (brew install libsndfile)
        sf.write(str(path), y, sr)
    except Exception:
        import wave
        y16 = np.int16(np.clip(y, -1.0, 1.0) * 32767)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1 if y.ndim == 1 else y.shape[0])
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(y16.tobytes())

def trim_leading_silence(y: np.ndarray, sr: int, top_db: float = 40.0, prepad_ms: int = 25):
    """Optional: remove leading silence so audio starts instantly."""
    try:
        import librosa
        yt, idx = librosa.effects.trim(y, top_db=top_db)
        start = max(0, idx[0] - int(prepad_ms / 1000 * sr))
        return y[start:]
    except Exception:
        # fallback: simple energy gate
        frame = max(1, int(0.01 * sr))
        energy = np.convolve(y**2, np.ones(frame)/frame, mode="same")
        thr = (10 ** (-top_db / 10.0)) * np.max(energy)
        start = int(np.argmax(energy > thr))
        start = max(0, start - int(prepad_ms / 1000 * sr))
        return y[start:]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to model_*.pt (e.g., exp/model_bestval.pt)")
    p.add_argument("--text", type=str, nargs="+", required=True,
                   help='One or more prompts, e.g. --text "Hallo Welt" "Wie geht es dir?"')
    p.add_argument("--speaker", type=int, default=999, help="Speaker ID to use")
    p.add_argument("--out_dir", type=Path, default=Path("tts_out"))
    p.add_argument("--trim_silence", action="store_true",
                   help="Trim leading silence so output starts right away")
    args = p.parse_args()

    device = pick_device()
    print(f"[info] device: {device}")

    # 1) Rebuild model & load weights
    state_dict, cfg = load_checkpoint(args.checkpoint, device)
    dec_w = cfg.get("decoder_loss_weight", 1.0)
    model = load_model(model_name_or_checkpoint_path=None,
                       device=device,
                       decoder_loss_weight=dec_w)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[warn] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    model.eval()

    # 2) Tokenizers + (optional) watermarker used by generate_audio
    text_tok, audio_tok = load_tokenizers(device)
    wm = load_watermarker(device=device)

    # 3) Generate for each prompt
    for i, sentence in enumerate(args.text):
        with torch.inference_mode():
            audio = generate_audio(
                model,
                audio_tok,
                text_tok,
                wm,
                sentence,          # text
                args.speaker,      # speaker id
                device,
                use_amp=False,
            )

        audio = np.asarray(audio, dtype=np.float32)
        if args.trim_silence:
            audio = trim_leading_silence(audio, MIMI_SAMPLE_RATE)

        out_path = args.out_dir / f"gen_{i:02d}.wav"
        save_wav(out_path, audio, MIMI_SAMPLE_RATE)
        print(f"[ok] saved: {out_path}  |  text: {sentence}")

if __name__ == "__main__":
    main()
