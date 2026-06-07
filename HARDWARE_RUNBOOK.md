# Hardware runbook — bringing Philosopher up on real devices

> v2 architecture. The **server** (brain) runs on a Raspberry Pi 4; the **toy**
> (dumb I/O) runs on a Banana Pi BPi-M2 Zero. See `CLAUDE.md` for the design.
> This is the step-by-step for a from-scratch bring-up + a smoke checklist.

---

## A. Server — Raspberry Pi 4 (the brain)

### 1. System packages
```bash
sudo apt update
sudo apt install -y python3-venv python3-pip cmake build-essential \
                    libopenblas-dev liblapack-dev libjpeg-dev libatlas-base-dev
```

### 2. dlib / face_recognition (the ARM64 gotcha)
There is **no PyPI ARM64 wheel** for dlib. Use piwheels first; only fall back to
a source build (2–4 h on a Pi) if that fails:
```bash
pip install dlib --index-url https://www.piwheels.org/simple
pip install face_recognition
```

### 3. piper TTS binary (a system binary, NOT a pip package)
Without it, TTS returns empty bytes → the toy gets text but no speech.
```bash
# Grab the arm64 release from github.com/rhasspy/piper/releases, then:
sudo install -m755 piper/piper /usr/local/bin/piper
piper --help     # must succeed
```

### 4. The server itself + models
```bash
cd philosopher-server
pip install -e ".[dev]"
python scripts/download_models.py        # pre-fetch FER+/Piper/Whisper (idempotent)
python scripts/check_runtime_deps.py      # <-- readiness gate; expect all [ ok ]
```
`check_runtime_deps.py` exits non-zero and prints `[FAIL]` lines for anything
missing (dlib, piper, haar cascade). Get it fully green before going further.

### 5. Run
```bash
python -m philosopher.main --server       # WebSocket + HTTP on :8080
# Sanity from another shell:
curl -s localhost:8080/health
# open http://<pi-ip>:8080/dashboard in a browser
```
Tip: `MOCK_MODE=true python -m philosopher.main --server` runs with stubbed
STT/vision/TTS — use it to prove the WebSocket plumbing before models are ready.

---

### Variant — server on a laptop (x86_64) instead of the RPi4

The server is just a networked Python service; it never touches the host's
mic/camera/speaker (those live on the toy). So a laptop is a drop-in replacement
and usually *better* — the architecture and the toy are unchanged. Differences:

- **Step 2 (dlib) simplifies.** On x86_64 Linux there are normal PyPI wheels —
  no piwheels, no source build: `pip install dlib face_recognition`.
- **`piper` is per-OS** — grab the x86_64 (and matching OS) release binary.
- **Headroom.** Bump `PHILOSOPHER_STT_MODEL=base` (better STT) and/or raise
  `PHILOSOPHER_CAMERA_FPS`; the vision gates (`PRESENCE_GATE`/`SKIP_SIMILAR`)
  become optional niceties rather than necessities.
- **GPU (optional).** `PHILOSOPHER_STT_DEVICE=cuda` if available — note
  `stt/engine.py` hardcodes `compute_type="int8"`; switch to `float16` for GPU.
- **Networking.** Laptops ship stricter firewalls — open TCP **8080**. Give the
  laptop a stable IP/hostname (DHCP changes break the toy's `PHILOSOPHER_SERVER_URL`).
- **Non-Linux hosts** (macOS/Windows) work but are less battle-tested here: dlib
  may need `cmake`/build tools. Linux x86_64 is the smoothest path.

Everything else below (toy setup, smoke checklist, perf knobs) applies unchanged.

---

## B. Toy — Banana Pi BPi-M2 Zero (the body, 512 MB)

Zero ML here — only capture/playback/servos. Keep its footprint tiny.
```bash
cd philosopher-toy
pip install -e ".[dev]"
# Point it at the server (base URL only — the client appends /ws?toy_id=… itself):
export PHILOSOPHER_SERVER_URL=ws://<pi-ip>:8080
export PHILOSOPHER_TOY_ID=banana_01        # optional; defaults to banana_01
python -m banana_client.main
# No hardware wired yet? Prove the client loop first:
PHILOSOPHER_MOCK=true python -m banana_client.main
```

---

## C. Smoke checklist (end-to-end, both devices live)

Run through these in order; each maps to a real failure mode the mocks can't catch.

| # | Action | Expect | If it fails |
|---|--------|--------|-------------|
| 1 | Toy connects to `ws://pi:8080/ws?toy_id=…` | server logs a connect; `/dashboard` shows activity | check `PHILOSOPHER_SERVER_URL`, network, port 8080 |
| 2 | Stand in front of the camera | head **pans to look at you** (`look`), `/dashboard` "Present now" shows you | camera feed? `check_runtime_deps` green for dlib + haar? |
| 3 | Make a clear happy/sad face | brief **arm gesture** (happy) or **head shake** (sad/angry) on emotion change | FER+ model present? emotion stuck on `neutral` ⇒ onnxruntime/model missing |
| 4 | Say something | server transcribes at `speech_ended`; reply streams back | STT mock `"Hello world"` ⇒ faster-whisper didn't load |
| 5 | Listen to the reply | **audio plays** sentence-by-sentence | silent but text shows ⇒ `piper` binary/voice missing |
| 6 | Speak while the toy is talking | your speech is **not** transcribed (half-duplex anti-echo) | mic gate (`AudioPlayer.is_playing`) wiring |
| 7 | Two people take turns | each gets their own recent-context (per-face memory) | face identity by encoding distance — lighting? |
| 8 | Step away for >4 s | head **recenters** (`look(0)`) and idle-drifts | `RECENTER_AFTER` path in `ws_server` |

### Quick triage (from CLAUDE.md "Gotchas")
"Runs but does nothing" is almost always a missing model/binary, not a logic bug:
- emotion always `neutral` → onnxruntime or FER+ model missing
- text but no audio → `piper` binary or voice model missing
- STT returns `"Hello world"` → faster-whisper failed to load

---

## D. Performance knobs (RPi4)

Vision runs three cheap gates before dlib (see `vision/engine.py`). Tune via env:
- `PHILOSOPHER_CAMERA_DETECT_WIDTH=480` — detection downscale width (lower = faster, less range).
- `PHILOSOPHER_CAMERA_PRESENCE_GATE=true` — Haar pre-check; skips dlib on no-face frames. Set `false` if side-profile faces get dropped.
- `PHILOSOPHER_CAMERA_SKIP_SIMILAR=true` / `PHILOSOPHER_CAMERA_SKIP_THRESHOLD=2.0` — skip near-identical frames (static scene).
- `PHILOSOPHER_CAMERA_FPS=0.5` — toy capture rate; the dominant lever on server load.
- `PHILOSOPHER_STT_MODEL=tiny` — bump to `base` only if the Pi has headroom.
