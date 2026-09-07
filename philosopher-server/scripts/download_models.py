#!/usr/bin/env python3
"""Pre-download server-side ML models so the Pi doesn't fetch them at startup.

Downloads (idempotently):
  - FER+ emotion ONNX model  (vision/emotion.py)
  - Piper TTS voice           (tts/engine.py, voice from settings/.env)
  - Kokoro TTS model          (tts/kokoro_engine.py, only if provider=kokoro)
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

_KOKORO_BASE = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
)


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
    from philosopher.config.settings import get_settings
    if get_settings().tts.provider != "piper":
        # settings.tts.voice belongs to the active provider (e.g. a kokoro voice
        # id like "em_santa"), which is not a piper voice — don't try to fetch it.
        print("== Piper TTS voice (skipped; provider != piper) ==")
        return True
    print("== Piper TTS voice ==")
    try:
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


def fetch_kokoro() -> bool:
    from philosopher.config.settings import get_settings
    if get_settings().tts.provider != "kokoro":
        print("== Kokoro TTS model (skipped; provider != kokoro) ==")
        return True
    print("== Kokoro TTS model ==")
    d = "./data/kokoro"
    ok1 = _download(f"{_KOKORO_BASE}/kokoro-v1.0.onnx", f"{d}/kokoro-v1.0.onnx")
    ok2 = _download(f"{_KOKORO_BASE}/voices-v1.0.bin", f"{d}/voices-v1.0.bin")
    return ok1 and ok2


def main() -> int:
    results = [fetch_emotion(), fetch_piper(), fetch_whisper(), fetch_kokoro()]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
