#!/usr/bin/env python3
"""Pre-download server-side ML models so the Pi doesn't fetch them at startup.

Downloads (idempotently):
  - FER+ emotion ONNX model  (vision/emotion.py)
  - Piper TTS voice           (tts/engine.py, voice from settings/.env)
  - faster-whisper STT model  (stt/engine.py, model size from settings/.env)

Each step is independent: a failure in one is reported and the others still run.
Run from the philosopher-server directory:

    python scripts/download_models.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# Single source of truth — reuse the constants/classes the runtime uses.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from philosopher.vision.emotion import _DEFAULT_MODEL_PATH, _MODEL_URL  # noqa: E402


def _download(url: str, dest: str) -> bool:
    path = Path(dest)
    if path.exists():
        print(f"[skip] {path} already exists ({path.stat().st_size // 1024} KB)")
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {url}\n       -> {path}")
    try:
        urllib.request.urlretrieve(url, str(path))  # noqa: S310
    except Exception as exc:  # noqa: BLE001
        print(f"[fail] {exc}")
        return False
    print(f"[ok  ] {path} ({path.stat().st_size // 1024} KB)")
    return True


def fetch_emotion() -> bool:
    print("== FER+ emotion model ==")
    return _download(_MODEL_URL, _DEFAULT_MODEL_PATH)


def fetch_piper() -> bool:
    print("== Piper TTS voice ==")
    try:
        from philosopher.config.settings import get_settings
        from philosopher.tts.engine import PiperTTS
        voice = get_settings().tts.voice
        # Constructing PiperTTS (non-mock, empty path) triggers the download.
        tts = PiperTTS(voice=voice)
        ok = Path(tts.model_path).exists()
        print(f"[{'ok  ' if ok else 'fail'}] {tts.model_path}")
        return ok
    except Exception as exc:  # noqa: BLE001
        print(f"[fail] {exc}")
        return False


def fetch_whisper() -> bool:
    print("== faster-whisper STT model ==")
    try:
        from faster_whisper import WhisperModel

        from philosopher.config.settings import get_settings
        s = get_settings()
        WhisperModel(s.stt.model, device=s.stt.device, compute_type="int8")
        print(f"[ok  ] whisper '{s.stt.model}' cached")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[fail] {exc}")
        return False


def main() -> int:
    results = [fetch_emotion(), fetch_piper(), fetch_whisper()]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
