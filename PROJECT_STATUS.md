# Project status — parked (body hardware blocked on the Banana Pi M2 Zero)

The **server/brain is done and works**. The project is paused on **one** thing:
the toy's host board (Banana Pi BPI-M2 Zero) is the wrong board for this, and
its onboard Wi-Fi makes the toy unusable. This doc says what works, why the
board doesn't fit, what carries over, and how to resume — so a cold restart
needs no re-discovery.

## What works (and is NOT tied to the M2 Zero)
- **The whole brain**, on the Raspberry Pi 4: STT (faster-whisper), LLM proxy
  (OpenAI-compatible, e.g. hermes-3-llama-3.1-8b), TTS (Piper, es), vision
  (face + emotion), memory, personality, the LangGraph pipeline, the WebSocket
  server. `/health` reports all engines ok.
- **The toy client** (`philosopher-toy/`): pure Python over WebSocket + audio
  (arecord/aplay) + camera + servos. Portable — it does not depend on this
  specific board.
- **Audio I/O**, proven on real hardware: USB mic (card 0, PCM2902) and USB
  speaker (card 1) work; the board plays sustained audio without brownout.
- **End-to-end path is connected**: the toy captures speech, the server receives
  it (`Processing audio ...`), transcribes, thinks, and replies. The plumbing is
  all there.

## Why the M2 Zero is the wrong board (the blocker)
Every layer of the *body* fought this board; the fatal one is the last:
- **Wi-Fi radio is bad.** The onboard AP6212 2.4 GHz loses ~35–63% of packets at
  4–5 m from the router — a normal link is ~0%. Disabling power-save barely moved
  it (to ~36%). It is the antenna/radio itself, not distance, config, Tailscale,
  or the SD (all ruled out). This alone makes the toy unusable: frozen sessions,
  WebSocket churn, ~1-minute reply latency.
  - Fixes that would work but **break the product**: gluing the board to the
    router (not plug&play), a USB-Ethernet cable (a plush toy can't be tethered),
    a USB Wi-Fi dongle (extra USB power draw — this board is already so
    power-tight that servos had to be dropped), or changing the home router's
    channel (shouldn't be required for plug&play).
- **512 MB RAM + weak single micro-USB rail** -> power is always the constraint
  (servo brownouts, USB draw budget).
- **No MIPI camera** — only parallel DVP; RPi cameras (OV5647) don't work, needs
  a specific OV5640 DVP module + a custom device-tree overlay.
- **40-pin header ships unsoldered** and has no hardware PWM on the header ->
  servos need hand-soldering + bit-banged PWM; joints proved marginal.

None of these are user error; they are the wrong board for a wireless,
plug&play, moving+seeing+hearing toy.

## Verdict / recommendation
Rebuild the **body** on a **genuine Raspberry Pi Zero 2 WH** (not a clone — the
M2 Zero's failures were clone corner-cuts behind a Pi-lookalike shell). Same
tiny form factor (fits the plush), and it fixes each failure we actually hit:
- **Wi-Fi** — proper Cypress radio + PCB antenna, well-supported (fixes THE blocker).
- **Camera** — real **MIPI CSI-2**: RPi cameras just work, no custom overlay.
- **Header** — the "**WH**" variant ships with the header **pre-soldered** (no soldering).
- **Servos** — **hardware PWM** + mature libs (gpiozero/pigpio), no bit-banging.
- **Software** — Raspberry Pi OS + huge community; almost none of the sunxi/Armbian
  pain applies.

This is **not** starting from zero — the RPi4 server and the toy code are reused
as-is; only the host board + minor pin/driver config change.

**Does NOT fix (board-agnostic, plan for it):** servo 5 V draw is still a design
task — power servos from a **separate 5 V rail / the toy's battery**, not the
board pin. Still 512 MB (fine — the toy is I/O only). Confirm exact parts/cables
before buying (the lesson from this build).

## Parts that carry over to a new board
USB mic, USB speaker, the servos — reusable on any board.

**Cameras (note the reversal on a MIPI board like the Pi Zero 2 W):**
- ✅ **OV5647** (the RPi MIPI "night-vision" module bought first) — **works** on
  the Pi Zero 2 W (it's MIPI; the Pi has MIPI). ⚠️ Needs the **narrow Pi-Zero
  camera ribbon** (mini 22-pin connector), often bundled with "for Pi Zero"
  modules or ~2-3€; verify against the exact module before relying on it.
- ❌ **OV5640 DVP** (the "for Banana Pi M2 Zero" module ordered later) — **does
  NOT work** on a MIPI board; it was DVP/parallel, only for the abandoned Banana.

So switching to the Pi Zero 2 W rescues the OV5647 (the earlier "wasted" cam) and
strands the OV5640 DVP instead. Wasted/board-specific: the M2 Zero itself and the
OV5640 DVP; see `memory/wasted-hardware-inventory` for reuse ideas.

## How to resume (cold start)
1. Pick/confirm the replacement board (see verdict).
2. Flash it; install the toy client (`philosopher-toy/`, `pip install -e .`).
3. Set the toy `.env`: `PHILOSOPHER_SERVER_URL=ws://<rpi4-ip>:8080`, ALSA
   devices for mic/speaker, `MIC_GAIN`/`VAD_THRESHOLD` (tune with
   `PHILOSOPHER_MIC_DEBUG=1`).
4. Start the RPi4 server (`python -m philosopher.main --server`) with the LLM up;
   consider `PHILOSOPHER_STT_MODEL=base` for accuracy over `tiny`.
5. First conversation: run the toy, talk, confirm mic->STT->LLM->TTS->speaker.

Detailed device-level notes and the full history live in `HARDWARE_RUNBOOK.md`
and the project memory (`banana-pi-bringup`).
