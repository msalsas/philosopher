#!/usr/bin/env python3
r"""Measure what the test suite can't: real engine accuracy + latency on hardware.

The unit suite runs everything in MOCK_MODE, so it validates the orchestration
glue but never the actual ML (dlib face-ID, Whisper STT, FER+ emotion, Piper
TTS). This harness drives the **real** engines on the RPi4 and prints three
numbers the product lives or dies by:

    1. Face-identity accuracy   (probe images resolve to the right person)
    2. Emotion accuracy         (FER+ vs labelled expressions, confusion matrix)
    3. End-to-end latency       (per stage + the speech_ended -> first audio path)

Run from the philosopher-server directory, ON the Pi (real models installed):

    # Latency only (no labelled data needed — synthesises its own inputs):
    python scripts/validate_hardware.py

    # Full accuracy run, pointing at your own captured samples:
    python scripts/validate_hardware.py \
        --faces samples/faces \          # samples/faces/<person>/*.jpg
        --emotions samples/emotions \    # samples/emotions/<happy|sad|...>/*.jpg
        --audio samples/audio \          # samples/audio/*.wav (+ optional *.txt transcript)
        --llm --tts --iters 7

Every section is independent and fail-soft: a missing model/library or absent
sample dir is reported and skipped, never crashed. Exit code is non-zero only if
a measured number misses its target (so it's usable as a CI/bring-up gate).
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
import wave
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK, WARN, FAIL = "[ ok ]", "[warn]", "[FAIL]"
SAMPLE_RATE = 16000  # Whisper / toy mic operate at 16 kHz mono int16.


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _jpeg(path: Path) -> bytes | None:
    """Read any image and re-encode as JPEG (the wire format the toy sends)."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else None


def _images(root: Path) -> dict[str, list[Path]]:
    """{label: [image paths]} from a <label>/*.{jpg,png} directory layout."""
    out: dict[str, list[Path]] = {}
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = sorted(
            p for p in sub.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if imgs:
            out[sub.name] = imgs
    return out


def _summarize(label: str, samples_ms: list[float]) -> dict | None:
    if not samples_ms:
        return None
    s = sorted(samples_ms)
    p90 = s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))]
    row = {
        "label": label,
        "median": statistics.median(s),
        "mean": statistics.fmean(s),
        "p90": p90,
        "n": len(s),
    }
    print(f"{OK} {label:<28} median {row['median']:7.0f} ms   "
          f"p90 {row['p90']:7.0f} ms   (n={row['n']})")
    return row


def _wer(ref: str, hyp: str) -> float:
    """Word error rate via word-level Levenshtein distance."""
    r, h = ref.lower().split(), hyp.lower().split()
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(
                d[j] + 1,           # deletion
                d[j - 1] + 1,       # insertion
                prev + (r[i - 1] != h[j - 1]),  # substitution
            )
            prev = cur
    return d[len(h)] / len(r)


# --------------------------------------------------------------------------- #
# 1. face identity accuracy
# --------------------------------------------------------------------------- #
async def measure_identity(faces_dir: Path, settings) -> bool:
    from philosopher.vision.engine import VisionProcessor

    people = _images(faces_dir)
    if not people:
        print(f"{WARN} identity: no <person>/*.jpg under {faces_dir}; skipping")
        return True
    if any(len(v) < 2 for v in people.values()):
        print(f"{WARN} identity: each person needs >=2 images "
              "(1 to enrol, rest as probes); skipping thin folders")

    # No memory backend: known_faces lives purely in RAM for the run. Disable the
    # static-frame skip so distinct probe images aren't collapsed into one result.
    vp = VisionProcessor(
        memory_manager=None, mock=False,
        emotion_model_path=settings.vision.emotion_model_path,
        tolerance=getattr(settings.vision, "tolerance", 0.6),
        detect_width=settings.vision.detect_width,
        presence_gate=settings.vision.presence_gate,
        skip_similar=False,
        merge_band=settings.vision.merge_band,
    )

    enrolled: dict[str, str] = {}      # person -> face_id minted from first image
    probes = correct = detected = 0

    for person, imgs in people.items():
        if len(imgs) < 2:
            continue
        jpg = _jpeg(imgs[0])
        if jpg is None:
            continue
        res = await vp.process_frame(jpg)
        face = res.get("primary_face")
        if not face:
            print(f"{WARN}   no face detected in enrolment image for '{person}': {imgs[0].name}")
            continue
        enrolled[person] = face["face_id"]

    for person, imgs in people.items():
        if person not in enrolled:
            continue
        for img in imgs[1:]:
            jpg = _jpeg(img)
            if jpg is None:
                continue
            probes += 1
            face = (await vp.process_frame(jpg)).get("primary_face")
            if not face:
                continue
            detected += 1
            if face["face_id"] == enrolled[person]:
                correct += 1

    if probes == 0:
        print(f"{WARN} identity: no probe images resolved; skipping")
        return True

    det_rate = detected / probes
    acc = correct / probes                 # over ALL probes (undetected = wrong)
    acc_det = correct / detected if detected else 0.0
    target = 0.85
    ok = acc >= target
    tag = OK if ok else FAIL
    print(f"{tag} identity accuracy   {acc:6.1%}  ({correct}/{probes} probes; "
          f"detected {det_rate:.0%}, accuracy-when-detected {acc_det:.0%})  "
          f"target >= {target:.0%}")
    return ok


# --------------------------------------------------------------------------- #
# 2. emotion accuracy
# --------------------------------------------------------------------------- #
async def measure_emotion(emotions_dir: Path, settings) -> bool:
    from philosopher.vision.engine import VisionProcessor

    labelled = _images(emotions_dir)
    if not labelled:
        print(f"{WARN} emotion: no <emotion>/*.jpg under {emotions_dir}; skipping")
        return True

    vp = VisionProcessor(
        memory_manager=None, mock=False,
        emotion_model_path=settings.vision.emotion_model_path,
        detect_width=settings.vision.detect_width,
        presence_gate=settings.vision.presence_gate,
        skip_similar=False,
        merge_band=settings.vision.merge_band,
    )

    labels = sorted(labelled.keys())
    matrix: dict[str, dict[str, int]] = {t: {} for t in labels}
    total = correct = 0

    for truth, imgs in labelled.items():
        for img in imgs:
            jpg = _jpeg(img)
            if jpg is None:
                continue
            face = (await vp.process_frame(jpg)).get("primary_face")
            # Fall back to a direct ROI predict if dlib found no face on a tight crop.
            if face:
                pred = face["emotion"]
            else:
                full = cv2.imread(str(img))
                pred = vp.emotion.predict(full) or "undetected"
            total += 1
            correct += (pred == truth)
            matrix[truth][pred] = matrix[truth].get(pred, 0) + 1

    if total == 0:
        print(f"{WARN} emotion: no images classified; skipping")
        return True

    acc = correct / total
    target = 0.60  # FER+ on in-the-wild toy-camera crops is modest; calibrate.
    ok = acc >= target
    tag = OK if ok else FAIL
    print(f"{tag} emotion accuracy    {acc:6.1%}  ({correct}/{total})  "
          f"target >= {target:.0%}")
    # Confusion matrix (truth row -> predicted columns).
    preds = sorted({p for row in matrix.values() for p in row})
    head = "          " + "".join(f"{p[:6]:>8}" for p in preds)
    print("    confusion (truth \\ pred):")
    print(head)
    for t in labels:
        cells = "".join(f"{matrix[t].get(p, 0):>8}" for p in preds)
        print(f"    {t[:9]:<9}{cells}")
    return ok


# --------------------------------------------------------------------------- #
# 3. latency (always runs) + optional STT WER
# --------------------------------------------------------------------------- #
def _read_wav_pcm(path: Path) -> bytes | None:
    """Return 16 kHz mono int16 PCM bytes, or None if the WAV doesn't match."""
    try:
        with wave.open(str(path), "rb") as w:
            if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
                print(f"{WARN}   {path.name}: not 16kHz/mono/16-bit; skipping")
                return None
            return w.readframes(w.getnframes())
    except Exception as exc:  # noqa: BLE001
        print(f"{WARN}   {path.name}: cannot read ({exc})")
        return None


async def measure_latency(args, settings) -> bool:
    rows: list[dict] = []
    targets_met = True

    # -- Vision: real frame through the full gated pipeline. --
    frame_jpeg = None
    if args.faces and Path(args.faces).is_dir():
        for imgs in _images(Path(args.faces)).values():
            if imgs:
                frame_jpeg = _jpeg(imgs[0])
                break
    if frame_jpeg is None:
        # Synthesise a frame so the latency path runs even with no samples.
        synth = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame_jpeg = cv2.imencode(".jpg", synth)[1].tobytes()
        print(f"{WARN} vision latency uses a synthetic frame "
              "(pass --faces for a realistic face frame)")

    try:
        from philosopher.vision.engine import VisionProcessor
        vp = VisionProcessor(
            memory_manager=None, mock=False,
            emotion_model_path=settings.vision.emotion_model_path,
            detect_width=settings.vision.detect_width,
            presence_gate=settings.vision.presence_gate,
            skip_similar=False,
            merge_band=settings.vision.merge_band,
        )
        samples = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            await vp.process_frame(frame_jpeg)
            samples.append((time.perf_counter() - t0) * 1000)
        vision_row = _summarize("vision: process_frame", samples)
        if vision_row:
            rows.append(vision_row)
    except Exception as exc:  # noqa: BLE001
        print(f"{WARN} vision latency skipped: {exc}")

    # -- STT: real Whisper transcription (+ WER if transcripts provided). --
    stt_row = None
    if args.audio and Path(args.audio).is_dir():
        wavs = sorted(Path(args.audio).glob("*.wav"))
        if not wavs:
            print(f"{WARN} STT: no *.wav under {args.audio}; skipping")
        else:
            try:
                from philosopher.stt.engine import StreamingSTT
                stt = StreamingSTT(
                    model_size=settings.stt.model, device=settings.stt.device,
                    compute_type=settings.stt.compute_type, mock=False,
                )
                samples, wers = [], []
                for wav in wavs:
                    pcm = _read_wav_pcm(wav)
                    if pcm is None:
                        continue
                    stt.clear_buffer("bench")
                    stt.add_audio_chunk("bench", pcm)
                    t0 = time.perf_counter()
                    res = await stt.transcribe("bench")
                    samples.append((time.perf_counter() - t0) * 1000)
                    ref_file = wav.with_suffix(".txt")
                    if ref_file.exists():
                        wers.append(_wer(ref_file.read_text().strip(), res.get("text", "")))
                stt_row = _summarize("stt: transcribe", samples)
                if stt_row:
                    rows.append(stt_row)
                if wers:
                    wer = statistics.fmean(wers)
                    tag = OK if wer <= 0.25 else FAIL
                    targets_met &= wer <= 0.25
                    print(f"{tag} stt word error rate  {wer:6.1%}  (n={len(wers)})  "
                          "target <= 25%")
            except Exception as exc:  # noqa: BLE001
                print(f"{WARN} STT latency skipped: {exc}")
    else:
        print(f"{WARN} STT latency skipped (pass --audio DIR with 16kHz mono WAVs)")

    # -- LLM: real endpoint, time-to-first-token + full stream. --
    llm_first_row = None
    if args.llm:
        try:
            from philosopher.llm.client import LLMClient, LLMMessage
            client = LLMClient(settings)
            firsts, fulls = [], []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                first = None
                async for tok in client.chat_stream(
                    [LLMMessage(role="user", content="Hola, ¿cómo estás?")],
                    system_prompt="Responde en una frase corta.",
                    max_tokens=60,
                ):
                    if first is None and tok.strip():
                        first = (time.perf_counter() - t0) * 1000
                fulls.append((time.perf_counter() - t0) * 1000)
                if first is not None:
                    firsts.append(first)
            llm_first_row = _summarize("llm: time-to-first-token", firsts)
            if llm_first_row:
                rows.append(llm_first_row)
            _summarize("llm: full response", fulls)
        except Exception as exc:  # noqa: BLE001
            print(f"{WARN} LLM latency skipped ({exc}); pass --llm with a reachable endpoint")
    else:
        print(f"{WARN} LLM latency skipped (pass --llm to hit the configured endpoint)")

    # -- TTS: real Piper synthesis of one sentence. --
    tts_row = None
    if args.tts:
        try:
            from philosopher.tts.engine import PiperTTS
            tts = PiperTTS(model_path=settings.tts.model_path,
                           voice=settings.tts.voice, mock=False)
            samples = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                audio = await tts.synthesize("Hola, soy tu amigo filósofo.")
                dt = (time.perf_counter() - t0) * 1000
                if audio:
                    samples.append(dt)
            if samples:
                tts_row = _summarize("tts: synthesize (1 sentence)", samples)
                if tts_row:
                    rows.append(tts_row)
            else:
                print(f"{WARN} TTS produced no audio (piper binary/voice missing?)")
        except Exception as exc:  # noqa: BLE001
            print(f"{WARN} TTS latency skipped: {exc}")
    else:
        print(f"{WARN} TTS latency skipped (pass --tts to synthesize via piper)")

    # -- Composite: the user-perceived "stop talking -> hear first reply". --
    def med(row):
        return row["median"] if row else None

    stt_m, llm_m, tts_m = med(stt_row), med(llm_first_row), med(tts_row)
    if stt_m is not None and llm_m is not None and tts_m is not None:
        composite = stt_m + llm_m + tts_m
        target_ms = args.latency_target * 1000
        ok = composite <= target_ms
        targets_met &= ok
        tag = OK if ok else FAIL
        print(f"\n{tag} END-TO-END (speech_ended -> first audio): "
              f"{composite:.0f} ms  =  STT {stt_m:.0f} + LLM-1st {llm_m:.0f} "
              f"+ TTS {tts_m:.0f}   target <= {target_ms:.0f} ms")
    else:
        print(f"\n{WARN} end-to-end composite needs --audio + --llm + --tts to be meaningful")

    return targets_met


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--faces", help="dir of <person>/*.jpg for identity accuracy + latency frame")
    p.add_argument("--emotions", help="dir of <emotion>/*.jpg for emotion accuracy")
    p.add_argument("--audio", help="dir of *.wav (16kHz mono) + optional *.txt for STT WER")
    p.add_argument("--llm", action="store_true", help="hit the configured LLM endpoint for latency")
    p.add_argument("--tts", action="store_true", help="synthesize via piper for TTS latency")
    p.add_argument("--iters", type=int, default=5, help="iterations per latency stage (default 5)")
    p.add_argument("--latency-target", type=float, default=1.5,
                   help="end-to-end target in seconds for the pass/fail gate (default 1.5)")
    return p.parse_args()


async def _run(args) -> int:
    from philosopher.config.settings import get_settings
    settings = get_settings()
    all_ok = True

    print("== 1. Face identity accuracy ==")
    if args.faces:
        all_ok &= await measure_identity(Path(args.faces), settings)
    else:
        print(f"{WARN} skipped (pass --faces DIR)")

    print("\n== 2. Emotion accuracy ==")
    if args.emotions:
        all_ok &= await measure_emotion(Path(args.emotions), settings)
    else:
        print(f"{WARN} skipped (pass --emotions DIR)")

    print("\n== 3. Latency ==")
    all_ok &= await measure_latency(args, settings)

    print("\n" + ("All measured targets met." if all_ok
                  else "Some measured numbers MISSED target — see [FAIL] lines."))
    return 0 if all_ok else 1


def main() -> int:
    return asyncio.run(_run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
