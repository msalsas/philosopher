# Hardware runbook — bringing Philosopher up on real devices

> v2 architecture. The **server** (brain) runs on a **laptop**; the **toy** (dumb
> I/O) runs on a **Raspberry Pi Zero WH**. See `CLAUDE.md` for the design. This is
> the step-by-step for a from-scratch bring-up + a smoke checklist.
>
> *(History: the toy was first prototyped on a Banana Pi M2 Zero clone, abandoned
> because its onboard WiFi was unusable. That whole bring-up — gadget serial,
> Allwinner H3 SUNXI GPIO, OV5640 DVP camera, I2S amp — no longer applies and is
> not covered here. Everything below is the genuine Pi Zero WH.)*

---

## 0. Repo & deployment strategy (one repo, both devices)

**Clone the whole repo onto each device — do not split it.** Reasons:

- The tracked repo is ~0.5 MB / ~100 files. The models under `data/` are
  **git-ignored** and re-fetched per device by `download_models.py`, so a full
  clone costs nothing on either machine.
- Each device installs **only its own subproject**, so the Pi Zero never pulls the
  server's heavy deps (faster-whisper/dlib/onnxruntime). The 512 MB limit is about
  what you `pip install` and run, not repo size.
- The WebSocket protocol is the contract between the two halves. In one repo a
  protocol change is one atomic commit touching both sides — they can't drift.

```bash
# On BOTH the laptop and the Pi Zero WH:
git clone <repo-url> ~/philosopher

# Laptop (brain): install the server only
cd ~/philosopher/philosopher-server && python -m venv .venv && .venv/bin/pip install -e .

# Pi Zero WH (body): install the toy only — but NOT with a plain `pip install -e .`!
# armhf has no PyPI wheels for numpy → pip would compile it from source (slow,
# can OOM on 512 MB). Install Debian's prebuilt packages and a venv that sees
# them instead. Full sequence in §B.2:
#   sudo apt install -y python3-numpy python3-aiohttp alsa-utils
#   cd ~/philosopher/philosopher-toy && python3 -m venv --system-site-packages .venv
#   .venv/bin/pip install -e . --no-deps && .venv/bin/pip install python-dotenv
```

Updates are `git pull` on each device (re-run `pip install -e .` only if deps
changed). **Never `git add` the `data/` dir** — it holds the SQLite memory DB with
face encodings (biometric data) + transcripts; `.gitignore` already blocks it.

**Autostart.** The **server** runs on the laptop and is *not* a service — start it
on demand with `./run.sh start` (models take ~1 min to load; no point keeping it
always resident). The **toy** lives inside the plush, so it *is* a systemd service
that comes up on boot. Install it once (from the machine that runs `run.sh`):
```bash
# On the Pi: give the toy user hardware access.
sudo usermod -aG gpio,audio,video <pi-user>
# From the run.sh host: generate + enable the toy service on the Pi (SSH).
./run.sh install        # writes /etc/systemd/system/philosopher-toy.service
./run.sh uninstall      # reverts it
```
`run.sh install` fills the unit with the Pi's local paths/user (nothing
hardcoded). Per-device settings go in a `.env` beside each subproject (git-ignored).

---

## A. Server — the laptop (the brain)

The server is just a networked Python service; it never touches the host's
mic/camera/speaker (those live on the toy). **Linux x86_64 is the smoothest path.**

### 1. System packages
```bash
sudo apt update
sudo apt install -y python3-venv python3-pip cmake build-essential \
                    libopenblas-dev liblapack-dev libjpeg-dev
# System binaries the code shells out to (not pip packages):
#   espeak-ng -> only for the kokoro TTS provider (phonemizer)
#   ffmpeg    -> only for the edge TTS provider (mp3 -> wav)
sudo apt install -y espeak-ng ffmpeg
```
The `piper` binary is installed separately (§A.3). Which TTS extras/system deps you
need depends on `PHILOSOPHER_TTS_PROVIDER` (piper=default/offline, kokoro=local
neural, edge=online); install the matching pip extra: `pip install -e ".[tts-kokoro]"`
or `".[tts-edge]"`.

### 2. dlib / face_recognition
On x86_64, `pip install dlib` still **compiles from source** (there is no current
PyPI wheel), which needs the `cmake` + `build-essential` from step 1 and takes a
few minutes:
```bash
pip install dlib face_recognition   # dlib compiles (~5 min); cmake/build-essential required
```
The stale `face_recognition` wrapper needs `pkg_resources` (removed in
setuptools≥81) and its ~100 MB model blobs — both pinned in `pyproject.toml`, so a
plain `pip install -e ".[dev]"` (§A.4) pulls them. If you hit a `pkg_resources`
import error in a hand-built env: `pip install "setuptools<81"`.

> **If you run the server on ARM instead (a Pi or other ARM64 host):** there is
> **no PyPI ARM64 wheel** for dlib. On Raspberry Pi OS Bookworm (Python 3.11) use
> piwheels first (`pip install dlib --index-url https://www.piwheels.org/simple`),
> only falling back to a 2–4 h source build. On Debian Trixie / Python 3.13,
> piwheels has no dlib wheel → use **Miniforge + `conda install -c conda-forge dlib`**
> on a 3.11 env (prebuilt aarch64). The RPi4 was the original target but its
> response times were too slow — that's why the server moved to a laptop.

### 3. piper TTS binary (a system binary, NOT a pip package)
Without it, TTS returns empty bytes → the toy gets text but no speech. The tarball
ships the binary *with bundled libs alongside it*, so move the whole folder and
symlink the binary — don't copy just the executable, or it won't find its libs:
```bash
cd /tmp
# grab the x86_64 asset from github.com/rhasspy/piper/releases (piper_linux_x86_64.tar.gz)
tar -xzf piper_linux_x86_64.tar.gz         # extracts ./piper/ (binary + libs + espeak-ng-data)
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
python scripts/check_runtime_deps.py     # <-- readiness gate; expect all [ ok ]
```
`check_runtime_deps.py` exits non-zero and prints `[FAIL]` lines for anything
missing (dlib, piper, haar cascade). Get it fully green before going further.

Deps green only proves the engines *load*, not that they *work*. Once you have a
few captured samples, measure the real numbers (the unit suite can't — it mocks all
ML):
```bash
python scripts/validate_hardware.py \
    --faces samples/faces \       # samples/faces/<person>/*.jpg  (>=2 each: 1 enrol + probes)
    --emotions samples/emotions \ # samples/emotions/<happy|sad|angry|...>/*.jpg
    --audio samples/audio \       # samples/audio/*.wav (16kHz mono) + optional *.txt transcript
    --llm --tts
```
It reports face-identity accuracy, an emotion confusion matrix, per-stage latency,
and the composite **speech_ended → first audio** time, exiting non-zero if a target
is missed. Run with no args for a latency-only smoke check.

### 5. Run
```bash
python -m philosopher.main --server       # WebSocket + HTTP on :8080
# Sanity from another shell:
curl -s localhost:8080/health             # each engine ok/mock/degraded/unavailable; status=degraded if any isn't ok
python scripts/fake_toy.py                # drive the WS end-to-end with no hardware
# open http://<laptop-ip>:8080/dashboard in a browser
```
`fake_toy.py` is a hardware-free toy: it sends a frame + audio + speech_ended and
prints the server's text/servo/WAV frames. With dummy audio + real STT you'll get
the `not_understood` fallback (correct — it's not speech); pass
`--wav speech.wav --jpeg face.jpg` for a true transcription→LLM→TTS turn.

Tip: `MOCK_MODE=true python -m philosopher.main --server` runs with stubbed
STT/vision/TTS — use it (with `fake_toy.py`) to prove the WebSocket plumbing before
models/LLM are ready. Note: `MOCK_MODE` does **not** mock the LLM — the server
always proxies the real external endpoint, so `llm: error` until
`PHILOSOPHER_LLM_BASE_URL` points at a reachable OpenAI-compatible server.

**Networking.** Open TCP **8080** on the laptop (laptops ship stricter firewalls).
Give it a stable IP/hostname — DHCP changes break the toy's `PHILOSOPHER_SERVER_URL`.
For a GPU host, `PHILOSOPHER_STT_DEVICE=cuda` and switch `stt/engine.py`'s
`compute_type` from `int8` to `float16`.

---

## B. Toy — Raspberry Pi Zero WH (the body, 512 MB)

Zero ML here — only capture/playback/servos. The toy is a **Raspberry Pi Zero WH**
(Broadcom BCM2835, **armv6 / 32-bit**, 512 MB, 2.4 GHz WiFi, pre-soldered 40-pin
header, MIPI CSI camera, two micro-USB). Its proper WiFi radio is why it replaced
the Banana clone (0% packet loss vs 36–63%).

### B.0 OS + first access (headless)
- **OS:** Raspberry Pi OS (Raspbian, 32-bit) via **rpi-imager**. In the imager's
  **OS customisation**, set the hostname, your user + SSH **public key** (disable
  password auth), and the **2.4 GHz** WiFi SSID + country. ⚠️ These settings must be
  **applied at the "apply settings?" prompt** — skip it and you get a bare image
  with no WiFi/SSH/user.
- If you forgot to customise: you can fix it **without reflashing** by editing the
  boot partition (writable as your user, no sudo): write `user-data`, `network-config`,
  bump `instance_id` in `meta-data`, **and clear the cloud-init cache**
  (`sudo rm -rf <rootfs>/var/lib/cloud/*`) — bumping `instance_id` alone does *not*
  force a re-run.
- Find the Pi: `ssh <user>@<hostname>.local`, or (if `.local` won't resolve) probe
  port 22 across the DHCP range and `ssh <user>@<ip> hostname`.

### B.1 Camera — OV5647 MIPI CSI (works via rpicam-still)
The Pi Zero WH has a real **MIPI CSI-2** connector, so a Raspberry Pi camera just
works — no custom overlay. This build uses the **OV5647** module.
- Seat the **narrow Pi-Zero camera ribbon** (mini 22-pin at the board end, 15-pin at
  the camera). ⚠️ A **reversed ribbon at the board end** gives `i2c ... -5` /
  `probe failed` on boot — flip it.
- In `/boot/firmware/config.txt`:
  ```
  camera_auto_detect=0
  dtoverlay=ov5647
  ```
- The toy grabs each JPEG via the **`rpicam-still`** binary (part of `rpicam-apps`,
  usually preinstalled) — `cv2.VideoCapture` can't read the libcamera CSI stream, so
  there is **no OpenCV on the toy**. Verify: `rpicam-still -n -o /tmp/t.jpg` produces
  a real image.
- Capture rate is `PHILOSOPHER_CAMERA_FPS` (default 0.5). The plush default camera
  mode is **on-speech** (one frame per turn — the spike lands while the amp is silent,
  no brownout); `stream` mode (continuous, drives gaze) needs more power headroom.

### B.2 Install the toy (armhf — do NOT let pip compile numpy)
PyPI has no armv6 wheel for numpy → pip would build it from source (slow, can OOM on
512 MB). Use Debian's prebuilt package and a venv that sees it:
```bash
sudo apt install -y git python3-venv python3-pip alsa-utils rpicam-apps
sudo apt install -y python3-numpy python3-aiohttp        # prebuilt, system-wide
git clone https://github.com/msalsas/philosopher.git
cd philosopher/philosopher-toy
python3 -m venv --system-site-packages .venv             # venv inherits the apt packages
source .venv/bin/activate
pip install -e . --no-deps                               # don't re-pull heavy deps from PyPI
pip install python-dotenv pytest pytest-asyncio pytest-mock
# Point it at the server (base URL only — client appends /ws?toy_id=… itself):
echo "PHILOSOPHER_SERVER_URL=ws://<laptop-ip>:8080" > .env
echo "PHILOSOPHER_TOY_ID=toy_01" >> .env
PHILOSOPHER_MOCK=true python -m toy_client.main          # prove the client loop first
```

### B.3 Audio — USB mic + USB speaker (arecord/aplay)
Audio I/O uses the **`arecord`/`aplay`** binaries, not PyAudio (which pegs the CPU
enumerating devices on the 512 MB board). **Verified working end-to-end — the toy
converses.**
- The USB mic is typically a **PCM2902** (48 kHz; ALSA `plug` resamples to 16 kHz for
  us). Cards seen: `arecord -l` / `aplay -l` list the mic and USB speaker; set
  `~/.asoundrc` default, e.g.
  `pcm.!default { type asym playback.pcm "plughw:1,0" capture.pcm "plughw:0,0" }`,
  or pin `PHILOSOPHER_MIC_ALSA_DEVICE` / `PHILOSOPHER_SPEAKER_ALSA_DEVICE`.
- **VAD tuning** (`PHILOSOPHER_MIC_DEBUG=1` prints `energy=… threshold=…`): set
  `PHILOSOPHER_VAD_THRESHOLD` **above** the idle noise floor, **below** speech (cheap
  mics ~500 idle / ~2500 on speech → ~1000 works). **Critical:** turn the codec's
  **Auto Gain Control OFF** (`alsamixer -c <card>`, then `sudo alsactl store`) — AGC
  lifts the silence floor to speech level, so the VAD never sees silence and never
  emits `speech_ended`. `PHILOSOPHER_MIC_GAIN` lifts a quiet capture (scales noise
  too — it's the threshold that separates speech, not the gain).
- Speaker volume: `PHILOSOPHER_SPEAKER_VOLUME` (percent) is applied via `amixer` at
  startup.
- ⚠️ **Power / brownout:** the mic + amp draw from the Pi's 5 V rail. Running both off
  a **bus-powered USB hub** browns out the rail mid-conversation (the toy reboots).
  Fix = a **powered USB hub** and/or a solid **5 V / 2–3 A** supply.

### B.4 Head servo — BCM12 hardware PWM
The head servo runs on **hardware PWM** (PWM0 on GPIO12) via the sysfs interface —
smooth, no software-PWM tremor. Add to `/boot/firmware/config.txt`, then reboot:
```
dtparam=audio=off              # frees PWM0 (we use USB audio, not the onboard jack)
dtoverlay=pwm,pin=12,func=4    # GPIO12 -> hardware PWM0
```
- The toy **exports/permits** `/sys/class/pwm/pwmchip0/pwm0` at startup via
  passwordless `sudo -n` (see `hardware/servos.py`). Redo the config.txt step if you
  reflash the SD.
- Every move **ramps then releases** (`duty 0`): a *held* servo draws enough current
  to brown out the 5 V rail alongside the amp (the "reboots mid-conversation" saga).
- **Arm servos are disabled** here — the pin map lists BCM13/18 but there's no HW-PWM
  channel for them, and a gesture during the reply coincides with the amp and browns
  out the board (`PHILOSOPHER_SERVO_GAZE_ONLY` skips them). So the head only does
  gaze glances + idle drift. Pins are env-overridable (`PHILOSOPHER_SERVO_*_PIN`,
  0 = off).

---

## C. Smoke checklist (end-to-end, both devices live)

Run through these in order; each maps to a real failure mode the mocks can't catch.

| # | Action | Expect | If it fails |
|---|--------|--------|-------------|
| 1 | Toy connects to `ws://laptop:8080/ws?toy_id=…` | server logs a connect; `/dashboard` shows activity | check `PHILOSOPHER_SERVER_URL`, network, port 8080 |
| 2 | Stand in front of the camera | a `look` glance toward you; `/dashboard` "Present now" shows you | camera feed? `check_runtime_deps` green for dlib + haar? |
| 3 | Make a clear happy/sad face | emotion registers on `/dashboard` (histogram/chart) — gestures are gaze-only here, so no arm move | FER+ model present? emotion stuck on `neutral` ⇒ onnxruntime/model missing |
| 4 | Say something | server transcribes at `speech_ended`; reply streams back | STT mock `"Hello world"` ⇒ faster-whisper didn't load |
| 5 | Listen to the reply | **audio plays** sentence-by-sentence | silent but text shows ⇒ `piper` binary/voice missing |
| 6 | Speak while the toy is talking | your speech is **not** transcribed (half-duplex anti-echo) | mic gate (`AudioPlayer.is_playing`) wiring |
| 7 | Two people take turns | each gets their own recent-context (per-face memory) | face identity by encoding distance — lighting? |
| 8 | Hold a conversation for a minute | no reboot mid-turn | brownout — use a powered USB hub (§B.3) |

### Quick triage (from CLAUDE.md "Gotchas")
"Runs but does nothing" is almost always a missing model/binary, not a logic bug:
- emotion always `neutral` → onnxruntime or FER+ model missing
- text but no audio → `piper` binary or voice model missing
- STT returns `"Hello world"` → faster-whisper failed to load

---

## D. Performance knobs

On the laptop these are mostly optional niceties (the RPi4 they were tuned for is no
longer the host); vision runs three cheap gates before dlib (see `vision/engine.py`):
- `PHILOSOPHER_CAMERA_DETECT_WIDTH=480` — detection downscale width (lower = faster, less range).
- `PHILOSOPHER_CAMERA_PRESENCE_GATE=true` — Haar pre-check; skips dlib on no-face frames. Set `false` if side-profile faces get dropped.
- `PHILOSOPHER_CAMERA_SKIP_SIMILAR=true` / `PHILOSOPHER_CAMERA_SKIP_THRESHOLD=2.0` — skip near-identical frames (static scene).
- `PHILOSOPHER_CAMERA_FPS=0.5` — toy capture rate (per turn in on-speech mode).
- `PHILOSOPHER_STT_MODEL=base` — `tiny` is fastest; `base` is noticeably more accurate and fine on a laptop.
