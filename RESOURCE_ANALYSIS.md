# Theoretical Resource Consumption — Philosopher v2

> Based on the WebSocket streaming architecture with configurable quality parameters.
> Hardware: Banana Pi BPi-M2 Zero (512MB) + Raspberry Pi 4 (4GB).

---

## 1. BANANA PI BPi-M2 Zero — Thin Client

### What It Does

| Task | Description |
|------|-------------|
| **Audio I/O** | Capture USB mic → energy VAD → stream PCM chunks to server |
| **Video I/O** | Capture OV5647 CSI frame → encode JPEG → stream to server |
| **Network** | Maintain WebSocket connection, heartbeat, auto-reconnect |
| **TTS** | Receive text sentences → Edge TTS download → ffplay playback |
| **Servos** | Receive emotion commands → move ONE servo at a time |
| **NO AI** | Zero ML inference. Zero face detection. Zero STT. |

### RAM Breakdown

| Component | RAM Usage | Notes |
|-----------|-----------|-------|
| OS (Armbian minimal, no desktop) | **150 MB** | Headless, trimmed services |
| Python 3.10 + interpreter | **40 MB** | Base runtime |
| asyncio + aiohttp (WebSocket) | **15 MB** | Persistent connection buffers |
| OpenCV (capture + encode only) | **80 MB** | No processing, just `VideoCapture` + `imencode` |
| PyAudio + USB audio buffers | **20 MB** | Stream buffering |
| Edge TTS (download + MP3 temp) | **40 MB** | Peaks during synthesis, then freed |
| OPi.GPIO + servo state | **5 MB** | Minimal |
| JPEG encode buffers (640x480) | **10 MB** | One frame at a time |
| **TOTAL STEADY STATE** | **~360 MB** | |
| **TOTAL PEAK (TTS + capture)** | **~400 MB** | Still 100MB headroom before swap |
| **FREE RAM** | **~100-150 MB** | Comfortable margin |

### CPU Breakdown (Single Core, ~10-40% utilization)

| Activity | CPU % | Duration | Notes |
|----------|-------|----------|-------|
| **Idle (WebSocket heartbeat)** | **5%** | Continuous | Just pings |
| **Audio capture + VAD** | **8%** | Continuous | Simple energy math on numpy arrays |
| **Camera capture (0.5 FPS)** | **12%** | Every 2s | `cap.read()` + `imencode` |
| **Camera capture (1.0 FPS)** | **20%** | Every 1s | Doubled encode load |
| **Camera capture (2.0 FPS)** | **35%** | Every 0.5s | Significant load |
| **Edge TTS synthesis** | **40%** | ~2-3s | Downloads MP3, decodes, streams to ffplay |
| **ffplay playback** | **15%** | During speech | Audio decoding |
| **Servo movement** | **3%** | ~0.5-1.2s | GPIO PWM, negligible |
| **PEAK (TTS + 1.0 FPS capture)** | **~75%** | Brief spikes | Still responsive |
| **PEAK (TTS + 2.0 FPS capture)** | **~90%** | Brief spikes | Risk of stutter |

### Critical Bottlenecks on Banana Pi

1. **Edge TTS + ffplay**: The biggest CPU spike. When a sentence arrives, the toy downloads an MP3 from Microsoft servers, saves to temp, then launches ffplay. This takes 1-3 seconds and hogs 40% CPU. During this time, camera capture may stutter.
2. **Camera encode at 2.0 FPS**: Encoding 640x480 JPEGs twice per second pushes the H3 to its limit. Not recommended unless CPU is monitored.
3. **USB Hub bandwidth**: Mic + speaker + WiFi dongle (if used) share USB 2.0 bandwidth (~480 Mbps). Audio is only ~256 kbps, so plenty of headroom. But if the hub is low-quality, latency jitter may occur.

---

## 2. RASPBERRY PI 4 — Server (Brain)

### What It Does

| Task | Description |
|------|-------------|
| **STT** | Receive PCM chunks → accumulate → Whisper transcribe → yield text |
| **Vision** | Receive JPEG → decode → face detection → face recognition → emotion analysis |
| **LLM** | Build prompt → stream tokens from external LLM API → sentence buffer |
| **Memory** | SQLite lookups, keyword search, recency scoring, store interaction |
| **Personality** | Load YAML, build system prompt, format response |
| **LangGraph** | Orchestrate 6-node pipeline (perception → memory → prompt → think → format → store) |
| **WebSocket** | Manage toy connection, route audio/vision/text streams |
| **NO TTS** | Toy handles speech synthesis. Server only sends text. |
| **NO LOCAL LLM** | External API only (LM Studio, OpenAI, etc.). Server is just a proxy. |

### RAM Breakdown (Steady State)

| Component | `tiny` STT | `base` STT | Notes |
|-----------|------------|------------|-------|
| OS (Raspberry Pi OS Lite) | **300 MB** | **300 MB** | Headless, no GUI |
| Python + FastAPI + websockets | **80 MB** | **80 MB** | Async server runtime |
| **Whisper model (loaded once)** | **100 MB** | **300 MB** | `faster-whisper` keeps model in RAM |
| Face recognition (face_recognition lib) | **80 MB** | **80 MB** | dlib model loaded at startup |
| DeepFace (emotion model) | **150 MB** | **150 MB** | TensorFlow lite model |
| SQLite + memory cache | **30 MB** | **30 MB** | WAL mode, small DB |
| LangGraph + state machine | **20 MB** | **20 MB** | Lightweight graph |
| Personality YAML (cached) | **5 MB** | **5 MB** | Parsed prompts |
| HTTP client (httpx + OpenAI) | **15 MB** | **15 MB** | Connection pool |
| **TOTAL with `tiny` STT** | **~780 MB** | — | **~3.2 GB free** |
| **TOTAL with `base` STT** | — | **~980 MB** | **~3.0 GB free** |

### CPU Breakdown (4 Cores, utilization per core)

| Activity | CPU Cores | Duration | Notes |
|----------|-----------|----------|-------|
| **Idle (WebSocket, heartbeat)** | **5% total** | Continuous | Negligible |
| **Whisper STT (`tiny`)** | **~150%** (1.5 cores) | **~1-2s** | After user stops speaking. Transcribes accumulated audio. |
| **Whisper STT (`base`)** | **~300%** (3 cores) | **~2-4s** | Slower but more accurate. May starve other tasks briefly. |
| **Face detection** | **~100%** (1 core) | **~0.5-1s** | Per JPEG frame. `face_recognition.face_locations()` is CPU-heavy. |
| **Face encoding + matching** | **~50%** (0.5 core) | **~0.3s** | If face found: compute encoding, check known faces. |
| **DeepFace emotion** | **~150%** (1.5 cores) | **~1-2s** | Per detected face. Heavy TF model. |
| **LLM streaming (proxy)** | **~10%** (0.1 core) | **Continuous** | Just forwards tokens from external API. Zero local compute. |
| **SQLite write** | **~20%** (0.2 core) | **~0.1s** | Store interaction. WAL mode, non-blocking. |
| **PEAK (STT + vision + emotion)** | **~400%** (all 4 cores) | **~2-3s** | When user finishes speaking + frame arrives simultaneously. |

### Critical Bottlenecks on Raspberry Pi 4

1. **Concurrent STT + Vision + Emotion**: When the user finishes speaking, the server must:
   - Run Whisper on accumulated audio (1.5-3 cores for 2-4s)
   - Process the latest JPEG frame (1 core for 1s)
   - Run DeepFace on detected face (1.5 cores for 1-2s)
   
   **Total: 4-5.5 cores worth of work, but only 4 cores available.** This means:
   - With `tiny` STT: Doable, but DeepFace may queue for 1-2 seconds. Acceptable.
   - With `base` STT: DeepFace will be delayed by 2-3 seconds. Emotion detection arrives **after** the LLM response starts. This is a problem — the toy may speak with the wrong emotion initially.

2. **DeepFace is the heaviest vision task**: On CPU, DeepFace takes 1-2 seconds per face. If the user is the only face in the frame, this is fine. If there are 2-3 people, it scales linearly (2-3x slower).

3. **No GPU acceleration**: Raspberry Pi 4 has no CUDA. Everything runs on ARM Cortex-A72 cores. Whisper `int8` quantization helps, but it's still slow compared to a desktop CPU.

---

## 3. COMPARISON TABLE: `tiny` vs `base` STT

| Metric | `tiny` (default) | `base` (higher quality) | Impact |
|--------|------------------|------------------------|--------|
| **Server RAM** | ~780 MB total | ~980 MB total | Pi 4 has 4GB. Both fine. |
| **Server CPU during STT** | 1.5 cores × 1-2s | 3 cores × 2-4s | `base` starves DeepFace |
| **STT accuracy** | ~85% WER (word error rate) | ~75% WER | `base` significantly better |
| **STT latency** | 1-2s after speech end | 2-4s after speech end | `base` delays response start |
| **Total pipeline latency** | ~2.5s (STT 1.5s + LLM 1s) | ~4.5s (STT 3s + LLM 1.5s) | Target is < 1.5s |
| **Recommendation** | **Start here** | Use if room is noisy | `tiny` may miss accented words |

**Key insight**: The Raspberry Pi 4 can handle either `base` STT OR real-time DeepFace emotion, but not both simultaneously without delays. With `tiny` STT, there's enough CPU headroom for vision. With `base`, vision lags.

---

## 4. CAMERA FPS TRADE-OFF ON BANANA PI

| FPS | Interval | CPU Load | Server Vision Load | Emotion Freshness | Recommendation |
|-----|----------|----------|-------------------|-------------------|----------------|
| **0.5** (default) | 2.0s | ~12% | 0.5 frames/sec to process | Up to 2s stale | Start here. Lowest risk. |
| **1.0** | 1.0s | ~20% | 1 frame/sec to process | Up to 1s stale | If H3 CPU < 50% idle. Better responsiveness. |
| **2.0** | 0.5s | ~35% | 2 frames/sec to process | Up to 0.5s stale | If H3 CPU < 30% idle AND server handles load. |

**Server-side impact of higher toy FPS**: The Pi 4 must process more JPEGs. At 2.0 FPS, DeepFace runs twice as often, consuming 2-4 seconds of CPU time per second of real time. This can backlog when STT is also running.

**Recommendation**: Keep toy at 0.5 FPS. The Pi 4 processes one frame every 2 seconds, leaving plenty of CPU for STT. Emotion feels "fresh enough" for a philosopher toy's slow, contemplative demeanor.

---

## 5. SERVO POWER REALITY CHECK

| Scenario | Power Draw | Banana Pi Stability |
|----------|------------|---------------------|
| **One servo moving** | ~200-300mA | ✅ Stable |
| **Two servos simultaneously** | ~400-600mA | ⚠️ Risk of brownout |
| **Three servos simultaneously** | ~600-900mA | ❌ Reboot likely |
| **Servo + TTS playback** | ~300mA + 100mA | ✅ Stable |
| **Servo + TTS + camera** | ~300mA + 100mA + 150mA | ⚠️ Marginal if all peak |

**The Banana Pi's MicroUSB input**: Assuming a 5V/2A supply (10W), the board itself uses ~2-3W (400-600mA). One servo at peak draws ~1.5W (300mA). Total: ~4-5W. Within the 10W budget, but **transient spikes** (servo stall current) can briefly exceed 2A and trigger the polyfuse or voltage regulator shutdown.

**Why sequential servos are mandatory**: Even though the average draw is within budget, the instantaneous current when a servo starts moving can spike to 1A for 50ms. Two servos starting simultaneously = 2A spike + board's 600mA = 2.6A. The polyfuse trips or voltage sags below 4.5V, causing the SoC to reset.

---

## 6. NETWORK BANDWIDTH

| Stream | Data Rate | Direction | Notes |
|--------|-----------|-----------|-------|
| **Audio (PCM 16kHz mono)** | ~256 kbps | Toy → Server | Continuous during speech |
| **Video (JPEG 640x480@60%)** | ~20-50 kbps | Toy → Server | Per frame. 0.5 FPS = negligible. |
| **Text (JSON sentences)** | ~1-5 kbps | Server → Toy | Sporadic, tiny |
| **Servo commands** | ~0.5 kbps | Server → Toy | One command per sentence |
| **Heartbeat** | ~0.1 kbps | Bidirectional | Every 5s |
| **TOTAL PEAK** | **~300 kbps** | Mixed | Well within WiFi capacity (even 802.11n = 50+ Mbps). |

**Latency**: On a local LAN/WiFi, WebSocket round-trip is ~5-15ms. Negligible compared to STT/LLM latency.

---

## 7. SUMMARY: WHAT EACH MACHINE DOES

### Banana Pi BPi-M2 Zero (512MB)
**Role**: Dumb I/O node. Zero intelligence.
- Captures microphone audio and sends raw PCM to server
- Captures camera frames and sends JPEG to server
- Receives text sentences, downloads Edge TTS audio, plays via speaker
- Receives servo commands, moves ONE servo at a time with delays
- Runs at ~60% average CPU, ~400MB RAM. Comfortable but not idle.

### Raspberry Pi 4 (4GB)
**Role**: AI brain. All computation.
- Transcribes audio chunks into text (Whisper)
- Detects faces, recognizes identities, analyzes emotions (face_recognition + DeepFace)
- Generates responses via external LLM API (OpenAI-compatible)
- Manages conversation memory (short-term deque + SQLite)
- Applies personality styles from YAML
- Streams sentences back to toy as they're completed
- Runs at ~30-80% CPU during interaction, ~800MB-1GB RAM. Headroom for `base` STT if needed.

---

## 8. REAL-WORLD PERFORMANCE EXPECTATIONS

### Best Case (quiet room, good WiFi, `tiny` STT, 0.5 FPS)
- User speaks for 3 seconds
- Toy streams audio to server
- User stops speaking
- Server: STT completes in 1.5s → face detected from last frame → memory lookup → LLM starts streaming
- **First sentence arrives at toy: ~2.0s after speech ends**
- Toy starts speaking: ~2.5s after speech ends
- **Perceived as: "slightly thoughtful pause, then responds"** ✅

### Worst Case (noisy room, `base` STT, 2.0 FPS, multiple faces)
- User speaks for 3 seconds
- Toy streams audio, but VAD may trigger early/late
- Server: STT takes 3s → 2 faces detected, DeepFace takes 3s → emotion arrives AFTER LLM response
- First sentence arrives at toy: ~4.5s after speech ends
- Toy speaks with stale emotion (or no emotion initially)
- **Perceived as: "long awkward pause, then responds robotically"** ❌

### Mitigation for Worst Case
- Use `tiny` STT (faster)
- Lower camera to 0.5 FPS (less server load)
- Add text-based emotion inference: if user says "I'm angry", override camera emotion
- Skip DeepFace if face_recognition detects no known face (save CPU)

---

## 9. UPGRADE PATHS

If the system feels sluggish in practice:

| Upgrade | Cost | Impact | Effort |
|---------|------|--------|--------|
| **Switch STT to `base`** | Free | +10-15% accuracy, +1-2s latency | Edit `.env`, restart |
| **Bump camera to 1.0 FPS** | Free | 2× fresher emotion, +8% toy CPU | Edit `.env`, restart |
| **External 5V servo supply** | ~$5 | Multi-servo animations, instant poses | Hardware mod |
| **Raspberry Pi 5 (8GB)** | ~$80 | 2-3× faster server, room for `small` STT | Swap board |
| **Coral USB TPU** | ~$60 | Offload face detection + emotion from CPU | USB plug-in |
| **Upgrade Banana Pi to M5 (4GB)** | ~$50 | Run Whisper locally, eliminate server dependency | Swap board |

The cheapest and highest-impact upgrade: **external 5V servo supply**. It unlocks instant, simultaneous servo poses for $5 in parts.
