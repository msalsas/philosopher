# Hardware runbook — bringing Philosopher up on real devices

> v2 architecture. The **server** (brain) runs on a Raspberry Pi 4; the **toy**
> (dumb I/O) runs on a Banana Pi BPi-M2 Zero. See `CLAUDE.md` for the design.
> This is the step-by-step for a from-scratch bring-up + a smoke checklist.

---

## 0. Repo & deployment strategy (one repo, both devices)

**Clone the whole repo onto each device — do not split it.** Reasons:

- The tracked repo is ~0.5 MB / ~100 files. The ~60 MB you see locally is
  downloaded models under `data/`, which is **git-ignored** and re-fetched per
  device by `download_models.py` — so a full clone costs nothing on either SD card.
- Each device installs **only its own subproject**, so the Banana Pi never pulls
  the server's heavy deps (faster-whisper/dlib/onnxruntime). The 512 MB limit is
  about what you `pip install` and run, not repo size.
- The WebSocket protocol is the contract between the two halves. In one repo a
  protocol change is one atomic commit touching both sides — they can't drift.

```bash
# On BOTH the RPi4 and the Banana Pi:
git clone <repo-url> ~/philosopher

# RPi4 (brain): install the server only
cd ~/philosopher/philosopher-server && python -m venv .venv && .venv/bin/pip install -e .

# Banana Pi (body): install the toy only
cd ~/philosopher/philosopher-toy   && python -m venv .venv && .venv/bin/pip install -e .
```

Updates are `git pull` on each device (re-run `pip install -e .` only if deps
changed). **Never `git add` the `data/` dir** — it holds the SQLite memory DB
with face encodings (biometric data) + transcripts; `.gitignore` already blocks
it, but don't force it.

**Autostart** with the provided systemd units (run on crash-restart):
```bash
# RPi4:
sudo cp ~/philosopher/deploy/philosopher-server.service /etc/systemd/system/
sudo systemctl enable --now philosopher-server
# Banana Pi (add the user to hardware groups first):
sudo usermod -aG gpio,audio,video $USER
sudo cp ~/philosopher/deploy/philosopher-toy.service /etc/systemd/system/
sudo systemctl enable --now philosopher-toy
```
Edit `User=`/paths in the unit files to match your install. Put per-device
settings in a `.env` beside each subproject (also git-ignored).

---

## A. Server — Raspberry Pi 4 (the brain)

### 1. System packages
```bash
sudo apt update
sudo apt install -y python3-venv python3-pip cmake build-essential \
                    libopenblas-dev liblapack-dev libjpeg-dev libatlas-base-dev
```

### 2. dlib / face_recognition (the ARM64 gotcha)
There is **no PyPI ARM64 wheel** for dlib. On **Raspberry Pi OS Bookworm (Python
3.11)**, use piwheels first; only fall back to a source build (2–4 h on a Pi) if
that fails:
```bash
pip install dlib --index-url https://www.piwheels.org/simple
pip install face_recognition
```

**On Debian Trixie / Python 3.13 / aarch64, piwheels has no dlib wheel** (it
lags new Python versions), so the command above triggers the multi-hour source
compile. Don't. Trixie ships *only* 3.13 (no apt `python3.11`), so the clean,
no-compile path is **Miniforge + conda-forge** (prebuilt aarch64 dlib) on a 3.11
env — this was the real bring-up sequence:
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh -b -p ~/miniforge3
~/miniforge3/bin/conda init bash && exec bash
conda create -y -n philosopher python=3.11
conda activate philosopher
conda install -y -c conda-forge dlib        # prebuilt — seconds, not hours
cd ~/philosopher/philosopher-server && pip install -e ".[dev]"
```
Then two stale-`face_recognition` snags (now pinned in `pyproject.toml`, but
here's why): its ~100 MB model blobs aren't always auto-pulled, and it imports
`pkg_resources`, which **setuptools≥81 removed** — so a fresh env needs:
```bash
pip install "setuptools<81"                  # restores pkg_resources
python -c "import dlib, face_recognition; print('vision stack ok')"
```
The systemd unit's `ExecStart` then points at the conda env, e.g.
`/home/<you>/miniforge3/envs/philosopher/bin/philosopher-server --server`.

### 3. piper TTS binary (a system binary, NOT a pip package)
Without it, TTS returns empty bytes → the toy gets text but no speech.
The tarball ships the binary *with bundled libs alongside it*, so move the whole
folder and symlink the binary — don't copy just the executable, or it won't find
its libs:
```bash
cd /tmp
# grab the aarch64 asset from github.com/rhasspy/piper/releases (piper_linux_aarch64.tar.gz)
tar -xzf piper_linux_aarch64.tar.gz        # extracts ./piper/ (binary + libs + espeak-ng-data)
sudo mv piper /opt/piper
sudo ln -sf /opt/piper/piper /usr/local/bin/piper
piper --help     # must succeed
```
The Piper *voice* model is separate (`data/piper_models/`, fetched by
`download_models.py`). `/health` shows `tts: degraded, voice_model: false` if the
binary is found but the voice `.onnx` isn't where the server's cwd resolves
`./data/` — pin `PHILOSOPHER_TTS_MODEL_PATH` to an absolute path, or always launch
from `philosopher-server/` (the systemd unit's `WorkingDirectory` does this).

### 4. The server itself + models
```bash
cd philosopher-server
pip install -e ".[dev]"
python scripts/download_models.py        # pre-fetch FER+/Piper/Whisper (idempotent)
python scripts/check_runtime_deps.py      # <-- readiness gate; expect all [ ok ]
```
`check_runtime_deps.py` exits non-zero and prints `[FAIL]` lines for anything
missing (dlib, piper, haar cascade). Get it fully green before going further.

Deps green only proves the engines *load*, not that they *work*. Once you have a
few captured samples, measure the real numbers (the unit suite can't — it mocks
all ML):
```bash
python scripts/validate_hardware.py \
    --faces samples/faces \       # samples/faces/<person>/*.jpg  (>=2 each: 1 enrol + probes)
    --emotions samples/emotions \ # samples/emotions/<happy|sad|angry|...>/*.jpg
    --audio samples/audio \       # samples/audio/*.wav (16kHz mono) + optional *.txt transcript
    --llm --tts
```
It reports face-identity accuracy, an emotion confusion matrix, per-stage latency,
and the composite **speech_ended → first audio** time, exiting non-zero if a
target is missed. Run with no args for a latency-only smoke check. These three
numbers — recognition accuracy, emotion accuracy, end-to-end latency — are the
ones that decide whether the product actually works; treat them as the bring-up
gate, not the green unit suite.

### 5. Run
```bash
python -m philosopher.main --server       # WebSocket + HTTP on :8080
# Sanity from another shell:
curl -s localhost:8080/health             # each engine ok/mock/degraded/unavailable; status=degraded if any isn't ok
python scripts/fake_toy.py                # drive the WS end-to-end with no hardware
# open http://<pi-ip>:8080/dashboard in a browser
```
`fake_toy.py` is a hardware-free toy: it sends a frame + audio + speech_ended and
prints the server's text/servo/WAV frames. With dummy audio + real STT you'll get
the `not_understood` fallback (correct — it's not speech); pass `--wav speech.wav
--jpeg face.jpg` for a true transcription→LLM→TTS turn.

Tip: `MOCK_MODE=true python -m philosopher.main --server` runs with stubbed
STT/vision/TTS — use it (with `fake_toy.py`) to prove the WebSocket plumbing
before models/LLM are ready. Note: `MOCK_MODE` does **not** mock the LLM — the
server always proxies the real external endpoint, so `llm: error` until
`PHILOSOPHER_LLM_BASE_URL` points at a reachable OpenAI-compatible server.

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
