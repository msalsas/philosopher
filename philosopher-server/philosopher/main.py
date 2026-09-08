"""Entry point for the Philosopher server."""
from __future__ import annotations

import argparse
import asyncio
import os

from philosopher.config.settings import get_settings
from philosopher.core.orchestrator import Orchestrator
from philosopher.stt.engine import StreamingSTT
from philosopher.tts.edge_engine import EdgeTTS
from philosopher.tts.engine import PiperTTS
from philosopher.tts.kokoro_engine import KokoroTTS
from philosopher.utils.logger import configure
from philosopher.vision.engine import VisionProcessor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Philosopher Conversational AI Server")
    p.add_argument("--server", action="store_true", help="Run API server")
    p.add_argument("--host", default="0.0.0.0", help="API host")
    p.add_argument("--port", type=int, default=8080, help="API port")
    p.add_argument("--personality", help="Personality to use")
    p.add_argument("--language", help="Language folder for personalities")
    p.add_argument("--llm-url", help="LLM endpoint URL")
    p.add_argument("--llm-model", help="LLM model name")
    p.add_argument("--debug", action="store_true", help="Debug logging")
    return p.parse_args()


async def run_chat(orch: Orchestrator):
    print(f"\n{'='*50}")
    print(f"  {orch.personality.name} is ready to chat")
    print("  Type 'exit' to quit, '/personality <name>' to switch")
    print(f"{'='*50}\n")
    while True:
        try:
            text = input("You: ").strip()
            if text.lower() in ("exit", "quit", "q"):
                break
            if text.startswith("/personality "):
                orch.personality.load(text.split(" ", 1)[1])
                print(f"Switched to: {orch.personality.s.personality}")
                continue
            result = await orch.process(text)
            print(f"\n{orch.personality.name}: {result.formatted}\n")
        except KeyboardInterrupt:
            break
    await orch.stop()


def main():
    args = parse_args()
    settings = get_settings()
    if args.debug:
        settings.logging.log_level = "DEBUG"
    if args.personality:
        settings.personality.personality = args.personality
    if args.language:
        settings.personality.language = args.language
    if args.llm_url:
        settings.llm.base_url = args.llm_url
    if args.llm_model:
        settings.llm.model = args.llm_model
    configure(settings.logging.log_level, settings.logging.log_format)

    async def _run():
        orch = Orchestrator()
        await orch.init()

        mock = os.getenv("MOCK_MODE", "false").lower() == "true"

        orch.stt = StreamingSTT(
            model_size=settings.stt.model,
            device=settings.stt.device,
            compute_type=settings.stt.compute_type,
            cpu_threads=settings.stt.cpu_threads,
            mock=mock,
        )
        orch.vision = VisionProcessor(
            memory_manager=orch.memory,
            mock=mock,
            emotion_model_path=settings.vision.emotion_model_path,
            tolerance=settings.vision.tolerance,
            detect_width=settings.vision.detect_width,
            presence_gate=settings.vision.presence_gate,
            skip_similar=settings.vision.skip_similar,
            skip_threshold=settings.vision.skip_threshold,
            merge_band=settings.vision.merge_band,
        )
        # Let the name-registration node reach vision for biometric dedup.
        orch.graph.ctx.vision = orch.vision
        # Voice + kokoro lang follow PHILOSOPHER_LANGUAGE unless explicitly set.
        _lang = settings.personality.language
        _voice = settings.tts.resolved_voice(_lang)
        if settings.tts.provider == "edge":
            orch.tts = EdgeTTS(
                voice=_voice,
                rate=settings.tts.rate,
                pitch_hz=settings.tts.pitch_hz,
                pitch=settings.tts.pitch,
                mock=mock,
            )
        elif settings.tts.provider == "kokoro":
            orch.tts = KokoroTTS(
                voice=_voice,
                lang=settings.tts.resolved_lang(_lang),
                speed=settings.tts.speed,
                pitch=settings.tts.pitch,
                mock=mock,
            )
        else:
            orch.tts = PiperTTS(
                model_path=settings.tts.model_path,
                voice=_voice,
                length_scale=settings.tts.length_scale,
                noise_scale=settings.tts.noise_scale,
                noise_w=settings.tts.noise_w,
                pitch=settings.tts.pitch,
                mock=mock,
            )

        if args.server:
            await orch.run()
        else:
            await run_chat(orch)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
