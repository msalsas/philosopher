# AGENTS.md — Philosopher Toy

> Reference for AI agents working on the Philosopher toy client (the "body").
> Last reviewed: 2026-06-06

---

## 1. Overview

The **philosopher-toy** is the physical client layer of the Philosopher conversational AI ecosystem. It runs on a **Raspberry Pi Zero WH** embedded inside a plush toy and handles all hardware interactions:

- **Camera**: Face detection and emotion recognition
- **Microphone**: Speech-to-text
- **Speaker**: Text-to-speech
- **Servos**: Head and arm movements

```
Camera/Mic -> Raspberry Pi Zero WH -> HTTP -> Philosopher Server -> LLM
                <- Response <-
Servos/Speaker <-
```

**Key Design Decisions:**
- Hardware-abstracted: mock mode works without physical hardware
- Simple HTTP REST: POSTs to server, receives JSON response
- No WebSocket: polling only
- Emotion-driven animations: servo positions based on detected emotion

---

## 2. Architecture

### 2.1 Main Loop (`banana_client/main.py`)

```
1. Capture frame from camera
2. Detect faces + emotions (OpenCV + face_recognition + DeepFace)
3. Listen to microphone (VAD - voice activity detection)
4. Transcribe speech to text (Google Speech Recognition)
5. POST to server: {message, face_id, face_name, emotion}
6. Receive text response
7. Synthesize voice (Edge TTS) and play via ffplay
8. Animate servos based on detected emotion
9. Repeat
```

### 2.2 Module Reference

| Module | Files | Hardware |
|--------|-------|----------|
| `protocol/` | `client.py` | HTTP client to philosopher-server |
| `vision/` | `camera.py` | USB camera (OpenCV + face_recognition + DeepFace) |
| `audio/` | `stt.py`, `tts.py` | Microphone (PyAudio + SpeechRecognition), Speaker (Edge TTS) |
| `hardware/` | `servos.py` | Head servo (GPIO 12), Left arm (GPIO 13), Right arm (GPIO 18) |

---

## 3. Configuration

Settings are loaded from `.env`:

```bash
PHILOSOPHER_SERVER_URL=http://192.168.1.50:8080
PHILOSOPHER_MOCK=false    # Set true for dev without hardware
```

---

## 4. Hardware Details

### 4.1 Servos (`hardware/servos.py`)
- Pins: head=GPIO 12, left_arm=GPIO 13, right_arm=GPIO 18
- Library: OPi.GPIO (falls back to mock mode on import error)
- Emotion animations:
  - `happy`: head +10°, right_arm +45°
  - `sad`: head -10°, left_arm -20°
  - `surprised`: head +20°, both arms +30°
  - default: head +5°

### 4.2 Camera (`vision/camera.py`)
- Resolution: 640x480 @ 5 FPS
- `face_recognition` for face detection + encoding
- `DeepFace.analyze(..., actions=["emotion"])` for emotion recognition
- Face ID: SHA256 hash of first 8 bytes of face encoding
- Known faces matched by distance threshold (0.6)
- Emotion mapping: `happy→happy`, `sad→sad`, `angry→angry`, `surprise→surprised`, `fear→fear`, `neutral→neutral`

### 4.3 STT (`audio/stt.py`)
- VAD: energy-based threshold (default 300)
- Silence timeout: 2 seconds
- Uses `speech_recognition.Recognizer().recognize_google(..., language="es-ES")`
- **Requires internet** (Google Speech Recognition)

### 4.4 TTS (`audio/tts.py`)
- Uses `edge_tts` (Microsoft Edge, free)
- Default voice: `es-ES-AlvaroNeural`
- Splits text into 400-char segments
- Plays via `ffplay -nodisp -autoexit`
- Strips `*...*` (asterisk actions) before speaking

---

## 5. Critical Implementation Details

### Face Recognition Hash (`vision/camera.py:48`)

Face IDs are deterministic hashes: `SHA256(first_8_bytes_of_encoding)[:12]`. This means the same face gets the same ID across sessions, but only if the encoding is bit-for-bit identical. Slight camera/lighting variations may produce different encodings.

### TTS Segmentation (`audio/tts.py:25`)

Edge TTS has length limits. Text is split into 400-character chunks. Each chunk is synthesized and played separately. Asterisk-wrapped actions (e.g., `*smiles*`) are stripped before TTS.

### Mock Mode

Set `PHILOSOPHER_MOCK=true` to run without physical hardware. Servos print to stdout instead of moving. Camera and microphone gracefully degrade to unavailable.

---

## 6. Testing

```bash
cd philosopher-toy
pytest tests/ -v
```

- 3 tests: protocol creation, servo mock movement, servo animation
- All tests run in mock mode (no hardware required)

---

## 7. Dependencies

```toml
dependencies = [
    "aiohttp>=3.9.0",
    "numpy>=1.26.0",
    "edge-tts>=6.1.0",
]

optional-dependencies = {
    "audio": ["pyaudio>=0.2.14", "speechrecognition>=3.10.0"],
    "vision": ["opencv-python>=4.9.0", "face-recognition>=1.3.0", "deepface>=0.0.89"],
}
```

Dev: `pytest`, `pytest-asyncio`, `pytest-mock`

No linter configured yet.

---

## 8. Quick Reference

| Task | Command |
|------|---------|
| Install | `cd philosopher-toy && pip install -e ".[all]"` |
| Run | `python -m banana_client.main` |
| Run (mock) | `PHILOSOPHER_MOCK=true python -m banana_client.main` |
| Test | `pytest tests/ -v` |

---

## 9. Known Limitations & Future Work

- **STT requires internet**: Uses Google Speech Recognition. No offline STT yet.
- **No wake word**: Currently requires button press or continuous listening. No "Hey Philosopher" detection.
- **No LED implementation**: `hardware/leds.py` does not exist yet.
- **TTS depends on `ffplay`**: Requires ffmpeg installed on the Raspberry Pi Zero WH.
- **Camera emotion detection is slow**: DeepFace on CPU can be sluggish. Consider lighter models.
- **No WebSocket**: HTTP polling only. Could add WebSocket for real-time bidirectional communication.
