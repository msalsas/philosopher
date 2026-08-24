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

# Banana Pi (body): install the toy only — but NOT with a plain `pip install -e .`!
# armhf has no PyPI wheels for numpy/opencv → pip would compile them from source
# (hours, OOM on 512 MB). Install Debian's prebuilt packages and a venv that sees
# them instead. Full sequence in §B.2:
#   sudo apt install -y python3-numpy python3-opencv python3-aiohttp alsa-utils
#   cd ~/philosopher/philosopher-toy && python3 -m venv --system-site-packages .venv
#   .venv/bin/pip install -e . --no-deps && .venv/bin/pip install python-dotenv
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

Zero ML here — only capture/playback/servos. Keep its footprint tiny. The M2 Zero
is a quirky 15 € board (Allwinner H3, **armhf/32-bit**, 512 MB, 2.4 GHz-only WiFi,
**one micro-USB OTG data port**, no Ethernet). The bring-up below is the hard-won
real sequence; skipping a step here cost us a full day.

> **STATUS (verified vs not):** B.0–B.2 were done and **verified** — the board boots,
> joins WiFi, the toy installs and reaches `[Toy] Ready!` and connects to the server
> over WebSocket (in mock and with a live mic). B.3+ (audio I/O) is **NOT yet verified
> on hardware** — see the caveats there. The board ended this session unstable / not
> booting (suspected SD corruption from forced power-cuts during hangs, B.5).

### B.0 OS + first access (headless)
- **OS:** Armbian "Debian 13 Trixie Minimal / CLI" for *Banana Pi M2 Zero*. Flash with
  rpi-imager → "Use custom" (the Armbian AppImager needs GLIBC ≥ 2.39, often too new).
- **Use a genuine SD card.** A fake/bad AliExpress card corrupts ext4 and gives endless
  `fsck`/hang loops — we burned a day on one. If boot drops to `UNEXPECTED INCONSISTENCY`,
  suspect the card before the software.
- **There is no usable network-less preseed on Trixie:** Armbian **removed
  `armbian_first_run.txt`** support, and the wizard runs on first login. So get a console:
  - **USB gadget serial (the rescue path):** the OTG port boots in *gadget* mode →
    plug a normal micro-USB **data** cable from the OTG port to a laptop → it enumerates
    as `Gadget Serial v2.4` → `sudo screen /dev/ttyACM0 115200`, Enter → `login:`.
    Default creds **`root` / `1234`** (wizard forces a change + user creation).
    *(A USB keyboard on the OTG port does NOT work in gadget mode — that confused us for
    an hour; the keyboard is fine, the port is just a device, not a host.)*
- **WiFi (networkd + netplan, NOT NetworkManager):** create
  `/etc/netplan/30-wifi.yaml` (chmod 600), 2.4 GHz SSID only:
  ```yaml
  network:
    version: 2
    renderer: networkd
    wifis:
      wlan0:
        dhcp4: true
        access-points:
          "YOUR_SSID":
            password: "YOUR_PASS"
  ```
  `netplan apply` (or reboot). Then SSH in and disable root login once your user works.

### B.1 USB OTG: gadget → host (needed for USB mic/speaker/keyboard)
The single OTG port can be **gadget** (serial console, default) **or host** (USB
peripherals) — not both. Armbian pins it to gadget by loading `g_serial` at boot.
To use USB audio you must switch to host:
```bash
sudo systemctl disable --now serial-getty@ttyGS0.service
sudo sed -i 's/^g_serial/#g_serial/' /etc/modules /etc/modules-load.d/modules.conf
printf 'blacklist g_serial\nblacklist libcomposite\n' | sudo tee /etc/modprobe.d/disable-gadget.conf
sudo reboot
```
After reboot `ls /sys/class/udc` still lists `musb-hdrc…` but plugging an OTG adapter
(micro-male → USB-A female, which grounds the ID pin) now enumerates USB devices
(`lsusb`). **You lose the USB serial console** — but a USB keyboard on HDMI now works as
the local rescue, and you have SSH. With >1 USB device (mic + speaker) you need a **hub**.

### B.2 Install the toy (armhf — do NOT let pip compile numpy/opencv)
PyPI has no armv7 wheels for numpy/opencv → pip would build them from source (hours,
likely OOM on 512 MB). Use Debian's prebuilt packages and a venv that sees them:
```bash
sudo apt install -y git python3-venv python3-pip alsa-utils
sudo apt install -y python3-numpy python3-opencv python3-aiohttp   # prebuilt, system-wide
git clone https://github.com/msalsas/philosopher.git
cd philosopher/philosopher-toy
python3 -m venv --system-site-packages .venv     # venv inherits the apt packages
source .venv/bin/activate
pip install -e . --no-deps                        # don't re-pull heavy deps from PyPI
pip install python-dotenv pytest pytest-asyncio pytest-mock
# Point it at the server (base URL only — client appends /ws?toy_id=… itself):
echo "PHILOSOPHER_SERVER_URL=ws://<pi-ip>:8080" > .env   # .env is loaded (python-dotenv)
echo "PHILOSOPHER_TOY_ID=banana_01" >> .env
PHILOSOPHER_MOCK=true python -m banana_client.main        # prove the client loop first
```

### B.3 Audio — ⚠️ UNVERIFIED on hardware
What we **observed (verified):** with the original **PyAudio** path the toy *did* reach
`[Toy] Ready!` and the live mic worked, BUT PyAudio's device enumeration takes **minutes**
and intermittently **pegs the CPU until the board hangs**. The USB mic is a **PCM2902**
that only supports **48 kHz** (rejects 16 kHz raw — let ALSA `plug` resample); cards seen:
mic = card 0, USB speaker = card 1, HDMI = card 2.

What we **changed but NEVER got to confirm working (commit `ffb75ec`):** audio I/O was
rewritten to use the **`arecord`/`aplay` binaries** instead of PyAudio, to dodge the slow
enumeration. **The first run after pulling it hung before we could verify a conversation —
so this path is unproven and is itself a suspect for the instability.** Validate it (or
revert it) once the board is stable again. The relevant env knobs (added alongside):
- `PHILOSOPHER_MIC_ALSA_DEVICE` / `PHILOSOPHER_SPEAKER_ALSA_DEVICE` (default `default` →
  follows `~/.asoundrc`: `pcm.!default { type asym playback.pcm "plughw:1,0" capture.pcm "plughw:0,0" }`).
- `PHILOSOPHER_MIC_DEBUG=1` prints `energy=… threshold=…` to calibrate the VAD.
- `PHILOSOPHER_VAD_THRESHOLD` — set **above** the idle noise floor, **below** speech. Cheap
  mics sit ~500 idle / ~2500 on speech → ~1000 works. **Critical:** turn the codec's
  **Auto Gain Control OFF** (`alsamixer -c 0`, then `sudo alsactl store`) — AGC lifts the
  silence floor to speech level, so the VAD never sees silence and never emits `speech_ended`.
- `PHILOSOPHER_MIC_GAIN` amplifies a quiet capture (scales noise too — it's the threshold
  that separates speech, not the gain).

### B.4 Not-yet-wired / known gaps on this board
- **Camera is CSI (parallel/DVP), not USB** → `cv2.VideoCapture(0, V4L2)` fails → camera
  mock for now (needs the CSI sensor driver / custom DT overlay — see **B.4.2**). The
  `/dev/video0` + `/dev/media0` that *do* exist are the **cedrus VPU decoder**, not a
  camera capture device — `media-ctl -p` shows `driver cedrus`, not a sensor.
- **Servos run on H3 SUNXI pins via bit-banged PWM** (`OPi.GPIO`). `hardware/servos.py` still
  uses RPi BCM pins (12/13/18) + `GPIO.PWM` — **must be ported to SUNXI + bit-bang**. Full
  detail in **B.4.3**.
- **Speakers are moving to GPIO/I2S** (e.g. MAX98357A DAC → a new ALSA card via DT overlay;
  `AudioPlayer` just retargets the device). I2S also stops drawing from the USB power rail.

### B.4.1 Speaker wiring — MAX98357A (I2S, no USB, no jack)
Chosen audio-out path: a **MAX98357A** I2S class-D amp on the GPIO header → one
speaker. Mono is fine for a talking toy. No USB, no analog jack, no separate PSU.
(The M2 Zero exposes **no analog line-out** on the header — only SPDIF on pin 37 —
and the H3 analog-codec output isn't broken out, so a PAM8403/analog amp would need
soldering to undocumented pads; the I2S MAX98357A is the clean route.)

5 wires + speaker. Connect the MAX98357A **by silkscreen label** (pin order varies):

| MAX98357A | Banana Pi M2 Zero (physical pin) | H3 line |
|-----------|----------------------------------|---------|
| `VIN`     | Pin 2 (5V)                       | —       |
| `GND`     | Pin 6 (GND)                      | —       |
| `BCLK`    | Pin 27                           | PA19    |
| `LRC`     | Pin 28                           | PA18    |
| `DIN`     | Pin 40                           | PA20    |
| `GAIN`    | leave floating (9 dB default)    | —       |
| `SD`      | leave floating (enabled, L+R mono) | —     |
| `+` / `–` | speaker (4–8 Ω)                  | —       |

```
 BANANA M2 ZERO header        MAX98357A            SPEAKER
   pin 2  (5V) ───────────▶ VIN
   pin 6  (GND) ──────────▶ GND
   pin 27 (PA19) ─────────▶ BCLK
   pin 28 (PA18) ─────────▶ LRC
   pin 40 (PA20) ─────────▶ DIN          OUT+ ──┐
                            GAIN (nc)            ├── 4–8 Ω speaker
                            SD   (nc)    OUT- ──┘
```

Power from the Pi 5V pin directly — a single amp for voice is a tiny load (the earlier
brownout hangs were a USB hub + mic + speaker, not this). All Dupont F-F; the header is
male pins, the MAX98357A usually ships with a male header (solder it if loose).

**Software (no ready overlay — build a custom one):** the kernel ships
`sun8i-h3-analog-codec`/`sun8i-h3-spdif-out` but **no `i2s0` overlay**. `armbian-add-overlay`
*is* present, so compile a custom overlay that enables `i2s0` + a `simple-audio-card`
bound to a `maxim,max98357a` codec, then verify the card appears in `aplay -l` and point
`PHILOSOPHER_SPEAKER_ALSA_DEVICE` at it. ⚠️ Re-enable the USB serial console first as a
recovery net (a bad overlay can break boot, and HDMI kills 2.4 GHz WiFi / there's no
serial otherwise). Overlays live in `/boot/dtb-<ver>/overlay/` + `/boot/overlay-user/`.

### B.4.2 Camera — OV5640 DVP (the ONLY CSI camera that works on H3)
⚠️ **Hardware constraint, learned the hard way:** the **Allwinner H3 has only a parallel
(DVP) camera interface — NO MIPI CSI-2 receiver.** Every Raspberry Pi camera (OV5647,
IMX219, …) is MIPI CSI-2, so **none of them work on the M2 Zero**, even though the FFC may
look similar. (We confirmed this after an OV5647 "night vision" module was bought by
mistake — it physically won't even seat: the M2 Zero CSI is a **24-pin FPC**, not the
Pi's 15-pin.) Bridging MIPI→DVP needs an FPGA — not worth it.

**Correct part:** an **OV5640 module sold *"for Banana Pi M2 Zero / M2+"*** — the DVP/parallel
variant on the 24-pin FPC, plugs straight into the CSI, **no expansion board** (e.g. OpenELAB
"Banana Pi BPI-M2+/M2 Zero Camera", ~$13; also Banana Pi store / AliExpress clones). Buy
checklist — the listing **must** say M2 Zero / M2+ and "no expansion board"; reject anything
"for Raspberry Pi" (MIPI 15-pin) or a generic 24-pin OV5640 for another board (Tinker etc.;
FPC pinout may differ). Verified-working unit (Qengineering): AliExpress item `32660117929`.

**Reference: [Qengineering/BananaPi-M2-Zero-OV5640](https://github.com/Qengineering/BananaPi-M2-Zero-OV5640).**
Proves the OV5640 works on this exact board and pins down the device specifics below.
⚠️ **Do NOT flash their prebuilt SD image** — it's Armbian 21.02 / kernel 5.10 / Buster
(ancient) and would wipe our Trixie + server setup; they also warn *"do not `apt upgrade`
or it removes the OV5640 drivers"* (fragile, baked to that kernel). We're on **6.18 mainline
where `sun6i-csi` + `ov5640` are upstream**, so build the overlay instead. Their image is
the fallback only if the overlay fights us.

**Device specifics (confirmed by the repo + our own probe):**
- Camera lands on **`/dev/video1` + `/dev/media1`** — `video0`/`media0` are the **cedrus VPU**
  (`media-ctl -p` → `driver cedrus`), which is why `cv2.VideoCapture(0)` opens nothing usable.
- OV5640 i2c entity is **`ov5640 2-003c`** → address **`0x3c` on i2c bus 2** (the CSI TWI,
  appears only once the overlay loads — *not* the header's `i2c-0` we probed earlier).
- **Mandatory before capture** (the format-set gotcha — `camera.py` must run this, or open the
  device after it's run, before cv2 reads frames):
  ```bash
  sudo media-ctl --device /dev/media1 \
    --set-v4l2 '"ov5640 2-003c":0[fmt:YUYV8_2X8/640x480@1/30]'
  ```
- Performance ceiling on H3 is **~2 fps @ 720p** — fine here: our `PHILOSOPHER_CAMERA_FPS`
  target is 0.5 fps (face/emotion frames), well within budget.

**Software (no ready overlay — build a custom one, same flow as B.4.1):**
1. Seat the 24-pin FPC in the CSI connector (mind contact orientation).
2. Compile a custom `sun8i-h3` overlay via `armbian-add-overlay` that enables `sun6i-csi`
   + an `ov5640` sensor node in **DVP 8-bit** mode (MCLK/PCLK, h/vsync, the CSI i2c bus,
   and the sensor's regulators/reset/pwdn GPIOs). The mainline `ov5640` driver supports DVP.
3. Verify: `i2cdetect -y 2` shows the OV5640 at **`0x3c`**; `media-ctl -p -d /dev/media1`
   shows an `ov5640` entity linked to the `sun6i-csi` bridge; **`/dev/video1`** appears
   (distinct from the cedrus `video0`). `v4l2-ctl -d /dev/video1 --list-formats-ext` shows
   sensor formats (e.g. UYVY/RGB), not the cedrus decode formats.
4. Run the `media-ctl --set-v4l2` format command above, then drop the mock in
   `banana_client/vision/camera.py` and point it at **`/dev/video1`** (currently it V4L2-opens
   index 0 → the cedrus; needs index 1 / `/dev/video1`, plus running the media-ctl init first).
⚠️ Re-enable the USB serial console first (a bad overlay can break boot — same caveat as B.4.1).

### B.4.3 Servos — H3 GPIO, bit-banged PWM, 5V distribution
- **The 40-pin header SHIPS UNSOLDERED** on this unit (loose strip — the "regleta"); it must
  be hand-soldered before any GPIO use. The board boots fine even with rough joints — bad
  joints simply don't conduct, so **test each pin before relying on it** (a servo that moves
  proves its signal/5V/GND joints conduct). Cold joints (grey balls not wetting the copper)
  are the #1 beginner failure → flux + heat *both* pin and pad, ~2 s, shiny cone.
- **Header is Raspberry-Pi-pin-compatible** (verified from the live DT): **5V on pins 2 & 4**,
  **GND on 6/9/14/20/25/30/34/39**, 3.3V on 1/17, I2C on 3/5, UART on 8/10. GPIO pins use
  **SUNXI names** — e.g. **PA6 = pin 7, PA7 = pin 29, PA8 = pin 31** (read `gpio-line-names`
  from the decompiled `sun8i-h2-plus-bananapi-m2-zero.dtb`; pin 1 = the square pad underside).
- **Drive via `OPi.GPIO`** (`pip install OPi.GPIO`), `GPIO.setmode(GPIO.SUNXI)`, pin names like
  `"PA6"`. ⚠️ Its **`PWM` class is HARDWARE-PWM only** (`PWM(chip,pin,freq,duty)` via sysfs
  pwmchip) and the M2 Zero exposes **no hardware PWM** on the header (`/sys/class/pwm` empty)
  → **bit-bang** the ~50 Hz signal with `GPIO.output` (jittery but moves a servo fine). See
  **`scripts/servo_test.py`** (`sudo .../python scripts/servo_test.py PA6 [PA7 PA8]`, sweeps
  each pin one at a time).
- **GPIO needs root** (`sudo`) — `manolo` isn't in a gpio group. The toy will need to run as
  root or get a udev rule; `servo_test.py` needs sudo (so it can't be launched over a
  non-interactive SSH without a NOPASSWD entry).
- **`hardware/servos.py` is NOT yet ported:** still RPi BCM pins (12/13/18) + `GPIO.PWM`.
  Port to `GPIO.SUNXI` + bit-bang (mirror `servo_test.py`), pick 3 SUNXI pins, keep the
  one-servo-at-a-time sequencing.
- **5V distribution (the catch):** the header has only **two 5V pins (2, 4)**, but 3 servos +
  the MAX98357A all want 5V. Fan one pin out with a **WAGO 221-415 (5 levers, all holes are a
  common node): pin 2 → 3 servo reds + amp VIN**. GND likewise (2nd 221-415, or just use the
  many GND pins). ⚠️ A **WAGO 221-412 = 2 levers = joins only 2 wires** (the small extra holes
  are multimeter test points, not wire slots) — use those for the **speaker leads**, not for
  distribution. WAGO clamps **bare wire**, so make stripped pigtails off Dupont jumpers.
- **Dupont gender:** header pins are male → **male-female** for servos (female on the pin, male
  into the servo's 3-pin female connector). The MAX98357A is also male after soldering its
  strip → it needs **female-female**.
- **Power headroom:** a single servo runs on the kit 5V/2A supply without browning out the
  board (servos move ONE AT A TIME by design). The combined 3-servo + audio draw on 2 A is the
  open question — test before buying a bigger PSU. If the bit-bang jitter matters, a **PCA9685**
  (I2C, 16-ch) gives clean hardware PWM. [[wasted-hardware-inventory]]

### B.5 Stability — hard hangs (UNRESOLVED)
The board hung repeatedly under sustained audio load — with PyAudio *and* on the first
(unverified) arecord/aplay run, and once it felt sluggish straight after SSH login.
**Root cause not confirmed.** Leading suspects, untested: an **unpowered USB hub + mic +
speaker browning out the 5 V rail** under simultaneous record/playback; the **sunxi musb
OTG host being weak with isochronous USB-audio transfers**; and possibly the **arecord/aplay
change itself** (never validated). To isolate next time: boot **bare (no USB)** and confirm
it stays stable; test `arecord … /dev/null` alone for 60 s; try a **powered USB hub** and a
solid **5 V/2–3 A** supply. **Never cut power to un-hang it** — yanking power mid-write
corrupts the SD (same failure mode as a fake card; this is the likely reason the board
stopped booting at the end of the session). Prefer `sudo reboot`; keep the HDMI+keyboard
console (works in host mode) as recovery.

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
