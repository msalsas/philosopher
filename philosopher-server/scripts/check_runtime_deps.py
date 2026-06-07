#!/usr/bin/env python3
"""Readiness check for the Philosopher server on real hardware (RPi4).

Verifies the things that DON'T crash the socket but make it "run and do nothing":
Python libs (incl. dlib, which has no ARM64 PyPI wheel), the `piper` system
binary, the OpenCV Haar cascade, and that the three ML models are on disk. Each
check is independent and reported; exit code is non-zero if any REQUIRED check
failed (missing models are warnings, since the runtime auto-downloads them).

Run from the philosopher-server directory, on the Pi:

    python scripts/check_runtime_deps.py
"""
from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK, WARN, FAIL = "[ ok ]", "[warn]", "[FAIL]"


def _check_import(mod: str, hint: str = "") -> bool:
    try:
        m = importlib.import_module(mod)
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} import {mod}: {exc}" + (f"\n        ↳ {hint}" if hint else ""))
        return False
    print(f"{OK} import {mod} {getattr(m, '__version__', '')}".rstrip())
    return True


def _check_binary(name: str, hint: str = "") -> bool:
    path = shutil.which(name)
    if path:
        print(f"{OK} binary {name} -> {path}")
        return True
    print(f"{FAIL} binary {name} not on PATH" + (f"\n        ↳ {hint}" if hint else ""))
    return False


def _check_file(label: str, path: str | Path, required: bool) -> bool:
    p = Path(path)
    if p.exists():
        print(f"{OK} {label}: {p} ({p.stat().st_size // 1024} KB)")
        return True
    tag = FAIL if required else WARN
    note = "" if required else " (auto-downloads on first use / run download_models.py)"
    print(f"{tag} {label}: {p} missing{note}")
    return not required


def main() -> int:
    required_ok = True

    print("== Python libraries ==")
    required_ok &= _check_import("cv2", "pip install opencv-python-headless")
    required_ok &= _check_import("numpy")
    required_ok &= _check_import("onnxruntime", "emotion recognition; pip install onnxruntime")
    required_ok &= _check_import("faster_whisper", "STT; pip install faster-whisper")
    required_ok &= _check_import("fastapi")
    required_ok &= _check_import(
        "face_recognition",
        "dlib has no ARM64 PyPI wheel — "
        "pip install dlib --index-url https://www.piwheels.org/simple",
    )

    print("\n== System binaries ==")
    required_ok &= _check_binary(
        "piper", "TTS engine; install the piper binary separately on the Pi",
    )

    print("\n== OpenCV assets ==")
    try:
        import cv2
        cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        required_ok &= _check_file("haar cascade (presence gate)", cascade, required=True)
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} cannot locate OpenCV cascades: {exc}")
        required_ok = False

    print("\n== ML models (warnings — runtime auto-downloads) ==")
    try:
        from philosopher.config.settings import get_settings
        from philosopher.tts.engine import PiperTTS
        from philosopher.vision.emotion import _DEFAULT_MODEL_PATH
        s = get_settings()
        emo = s.vision.emotion_model_path or _DEFAULT_MODEL_PATH
        _check_file("FER+ emotion model", emo, required=False)
        voice_path = s.tts.model_path or PiperTTS._default_model_path(s.tts.voice)
        _check_file("Piper voice", voice_path, required=False)
        _check_file("Piper voice config", voice_path + ".json", required=False)
        print(f"[info] Whisper STT size = '{s.stt.model}' (cached under ~/.cache/huggingface)")
    except Exception as exc:  # noqa: BLE001
        print(f"{WARN} could not resolve model paths from settings: {exc}")

    print("\n" + ("All required checks passed." if required_ok
                  else "MISSING required deps — see [FAIL] lines above."))
    return 0 if required_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
