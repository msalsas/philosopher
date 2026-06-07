"""Server-side TTS using Piper (invoked via subprocess)."""
from __future__ import annotations

import subprocess
import tempfile
import urllib.request
from pathlib import Path

# Piper voices are hosted on HuggingFace (rhasspy/piper-voices). A voice id like
# "es_ES-carlfm-x_low" maps to the path "es/es_ES/carlfm/x_low/<voice>.onnx".
_PIPER_VOICES_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


class PiperTTS:
    """Synthesizes speech using Piper via subprocess."""

    def __init__(self, model_path: str = "", voice: str = "es_ES-carlfm-x_low", mock: bool = False):
        self.voice = voice
        self.mock = mock
        # Always resolve a deterministic model path; only fetch when needed.
        self.model_path = model_path or self._default_model_path(voice)
        if not model_path and not mock:
            self._download_model(voice)

    @staticmethod
    def _default_model_path(voice: str) -> str:
        model_dir = Path("./data/piper_models")
        model_dir.mkdir(parents=True, exist_ok=True)
        return str(model_dir / f"{voice}.onnx")

    @staticmethod
    def _voice_repo_path(voice: str) -> str:
        """'es_ES-carlfm-x_low' -> 'es/es_ES/carlfm/x_low/es_ES-carlfm-x_low'."""
        lang_region, name, quality = voice.split("-", 2)
        lang = lang_region.split("_")[0]
        return f"{lang}/{lang_region}/{name}/{quality}/{voice}"

    def _download_model(self, voice: str) -> None:
        """Download the Piper .onnx + .onnx.json if missing. Idempotent.

        Any failure is swallowed; ``synthesize`` then degrades to empty bytes
        (text-only) rather than crashing.
        """
        model_file = Path(self.model_path)
        config_file = Path(self.model_path + ".json")  # <voice>.onnx.json
        if model_file.exists() and config_file.exists():
            return
        try:
            base = f"{_PIPER_VOICES_BASE}/{self._voice_repo_path(voice)}"
            if not model_file.exists():
                urllib.request.urlretrieve(f"{base}.onnx", str(model_file))  # noqa: S310
            if not config_file.exists():
                urllib.request.urlretrieve(f"{base}.onnx.json", str(config_file))  # noqa: S310
        except Exception:
            pass

    async def synthesize(self, text: str) -> bytes:
        if self.mock:
            return b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 40

        config_path = self.model_path.replace(".onnx", ".onnx.json")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            subprocess.run(
                [
                    "piper",
                    "--model", self.model_path,
                    "--config", config_path,
                    "--output_file", output_path,
                ],
                input=text.encode(),
                check=True,
                capture_output=True,
            )
            return Path(output_path).read_bytes()
        except (FileNotFoundError, subprocess.CalledProcessError, OSError):
            # piper binary missing, model/config missing, or synthesis failed.
            return b""
        finally:
            p = Path(output_path)
            if p.exists():
                p.unlink()
