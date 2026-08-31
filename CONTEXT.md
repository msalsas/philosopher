# Philosopher - Contexto del Proyecto para Retomar

> ### ⚠️ Este documento describe la v1. El proyecto está en **v2**.
> En v2 **toda la IA corre en el servidor (Raspberry Pi 4, 4GB)**: STT (`faster-whisper`), visión (`face_recognition` + `DeepFace`) y TTS (**Piper** vía subprocess). El peluche (**Raspberry Pi Zero WH, 512MB**) es un **cliente de E/S sin ML**: captura mic/cámara y los envía por **WebSocket**, reproduce WAV y mueve servos. La comunicación es **WebSocket** (`/ws`), no HTTP POST (el `/chat` HTTP solo queda para diagnóstico). Donde el texto de abajo diga "No WebSocket", STT con Google, o DeepFace/STT/Edge-TTS *en el peluche*, es **v1 obsoleta**. La fuente de verdad es `MIGRATION_GUIDE.md` + el código + `/CLAUDE.md`.

## Vision General

Philosopher es un agente conversacional con personalidad, memoria dual (corto y largo plazo), reconocimiento facial y deteccion de emociones. Esta embebido en un peluche inteligente que funciona con una Raspberry Pi Zero WH. El proyecto se divide en **dos proyectos Python independientes**.

## Arquitectura de 3 Capas

```
+-------------------+  WiFi/LAN  +--------------------+  HTTP REST  +-------------------+
| Ordenador LLM     |            | philosopher-server  |             | philosopher-toy    |
| (LM Studio,       | <------->  | (Cualquier PC)     | <---------> | (Raspberry Pi Zero WH en el  |
|  OpenAI, Ollama)  |   HTTP     |                    |   WiFi      |  peluche)         |
+-------------------+            +--------------------+             +-------------------+
  Modelo 8B                            Cerebro                           Cuerpo
  Genera texto                         LangGraph + Memoria               Camara, micro, 
                                       Personalidad + API                servos, LEDs
```

### Capa 1: Ordenador LLM
- **Hardware**: Cualquier ordenador con GPU o CPU suficiente
- **Software**: un servidor LLM OpenAI-compatible (LM Studio, Ollama, vLLM, llama.cpp…), OpenAI API, Ollama, o cualquier endpoint compatible con OpenAI API
- **Modelo**: 8B parametros (ej: Llama 3.1 8B, Mistral 7B, Phi-3)
- **Rol**: Genera respuestas de texto basadas en los prompts que el servidor construye
- **Conexion**: HTTP local (misma red WiFi que el servidor)

### Capa 2: philosopher-server (Cualquier ordenador con Python 3.10+)
- **Hardware**: Cualquier PC, portatil, o Raspberry Pi 4
- **Rol**: El cerebro. Coordina toda la inteligencia del sistema
- **Proyecto Python**: `philosopher-server/` con su propio `pyproject.toml`

### Capa 3: philosopher-toy (Raspberry Pi Zero WH dentro del peluche)
- **Hardware**: Raspberry Pi Zero WH BPI-M2 Zero / BPI-M64 + camara USB + microfono + altavoz + servos + LEDs
- **Rol**: Los sentidos y el cuerpo. Captura el mundo fisico y actua sobre el
- **Proyecto Python**: `philosopher-toy/` con su propio `pyproject.toml`

## Proyecto 1: philosopher-server

### Pipeline LangGraph (flujo de procesamiento)

Cada mensaje del usuario recorre 6 nodos en secuencia:

```
[User Input] --> perception --> memory --> prompt --> think --> format --> store
                    |            |          |         |        |        |
                    v            v          v         v        v        v
               Busca rostro  Recupera   Construye  LLM     Aplica   Guarda en
               conocido     contexto   prompt     genera  estilo   memoria
               en SQLite    relevante  con pers.  texto   persona
```

1. **perception**: Recibe `face_id` del cliente, busca en SQLite si es conocido, obtiene nombre y numero de encuentros
2. **memory**: Recupera memoria corta (ultimos 10 mensajes) + memoria larga (busqueda por keywords en SQLite)
3. **prompt**: Construye el system prompt combinando: personalidad YAML + nombre persona + emocion + recuerdos relevantes
4. **think**: Llama al LLM (OpenAI-compatible API) con el prompt construido + historial de conversacion
5. **format**: Aplica estilo de personalidad (trunca si es necesario, limpia formato)
6. **store**: Guarda la interaccion en memoria corta (buffer circular) y larga (SQLite con keywords)

### Modulos del servidor

| Modulo | Archivos | Funcion |
|--------|----------|---------|
| `config/` | `settings.py`, `personalities/<lang>/*.yaml` | Configuracion Pydantic + prompts de personalidad |
| `core/` | `state.py`, `nodes.py`, `graph.py`, `orchestrator.py` | Pipeline LangGraph completo |
| `llm/` | `client.py` | Cliente generico OpenAI (local/cloud) |
| `memory/` | `short_term.py`, `long_term.py`, `manager.py` | Buffer circular + SQLite semantico |
| `personality/` | `engine.py` | Motor YAML multi-idioma |
| `api/` | `app.py` | FastAPI REST para el cliente |
| `events/` | `bus.py` | Bus de eventos pub/sub (17 tipos) |
| `utils/` | `logger.py` | Logging estructurado con structlog |

### Memoria

- **Corta plazo**: `deque` circular en RAM (max 10 mensajes). Se pierde al reiniciar.
- **Larga plazo**: SQLite persistente con busqueda por keywords. Tablas:
  - `memories`: id, timestamp, type, content, summary, emotion, face_id, face_name, keywords
  - `known_faces`: face_id, name, first_seen, last_seen, encounters
- Busqueda semantica simple: keywords extraidas del texto + factor de recencia

### Personalidades

**Estructura**: Solo archivos YAML en `config/personalities/<idioma>/`. Sin codigo.

| Nombre | Rol | Traits clave |
|--------|-----|-------------|
| `filosofo` | Acompañante sabio | empathy:9, energy:4, ask_questions, metaphors |
| `curioso` | Explorador entusiasta | curiosity:10, energy:9, siempre pregunta |
| `poetico` | Alma artistica | empathy:8, energy:3, metaphors, images |
| `amigo` | Amigo cercano | empathy:9, humor:8, formalidad:0, directo |
| `sabio` | Mentor anciano | formality:4, energy:2, max 2 frases |

Cada YAML contiene: `system_prompt`, `traits`, `speech_patterns` (greetings, farewells), `emotion_responses` (6 emociones), `style` (max_length, use_metaphors, etc.)

**Multi-idioma**: La variable `PHILOSOPHER_LANGUAGE` selecciona la carpeta (`es`, `en`, etc.). Para añadir un idioma nuevo, solo crear carpeta con YAMLs.

### API REST

| Endpoint | Metodo | Body/Params | Response |
|----------|--------|-------------|----------|
| `/health` | GET | - | status, llm_health, memory_stats |
| `/chat` | POST | `message`, `face_id?`, `face_name?`, `emotion?` | `response`, `emotion`, `turn` |
| `/personalities` | GET | - | Lista de personalidades disponibles |
| `/personality` | POST | `name` | Confirmacion de cambio |
| `/memory/stats` | GET | - | total_memories, known_faces, by_type |

## Proyecto 2: philosopher-toy

### Hardware controlado

| Componente | Modulo | GPIO pins | Libreria |
|------------|--------|-----------|----------|
| Cabeza (servo) | `hardware/servos.py` | GPIO 12 | OPi.GPIO / lgpio |
| Brazo izq (servo) | `hardware/servos.py` | GPIO 13 | OPi.GPIO / lgpio |
| Brazo der (servo) | `hardware/servos.py` | GPIO 18 | OPi.GPIO / lgpio |
| Ojos LED | `hardware/leds.py` | GPIO 19 | OPi.GPIO / lgpio |
| Boton | (events) | GPIO 20 | OPi.GPIO / lgpio |
| Sensor tactil | (events) | GPIO 16 | OPi.GPIO / lgpio |
| Camara USB | `vision/camera.py` | USB | OpenCV (solo captura + JPEG; visión va en el server) |
| Microfono USB | `audio/capture.py` | USB | PyAudio + VAD (STT va en el server) |
| Altavoz | `audio/player.py` | Jack/USB | PyAudio (reproduce WAV del server; TTS va en el server) |

> **v2:** El peluche no hace ML. `audio/stt.py` y `audio/tts.py` son legado v1.

### Bucle principal del cliente — v2 (WebSocket)

```
Tres tareas asyncio concurrentes sobre un WebSocket:
- audio:  mic + VAD -> speech_started/PCM(0x01)/speech_ended al server
- video:  captura frame -> JPEG(0x02) al server a PHILOSOPHER_CAMERA_FPS
- recibe: {type:text} -> imprime
          {type:servo} -> servos.animate(emotion)   (un servo a la vez)
          WAV(0x03) -> AudioPlayer.play_wav()
```

### Animaciones por emocion

| Emocion | Cabeza | Brazo izq | Brazo der | LED |
|---------|--------|-----------|-----------|-----|
| happy | +10° | - | +45° | verde |
| sad | -10° | -20° | - | azul |
| surprised | +20° | +30° | +30° | amarillo |
| angry | -5° | - | - | rojo |
| neutral | +5° | - | - | blanco |

## Flujo de datos completo (interaccion tipica)

```
Usuario habla al peluche
    |
    v
[Toy] Microfono captura audio
[Toy] STT transcribe: "Hola, como estas?"
[Toy] Camara detecta rostro: face_id="abc", emotion="happy"
    |
    v HTTP POST /chat
[Toy --> Server] {message:"Hola...", face_id:"abc", emotion:"happy"}
    |
    v
[Server] perception: face_id "abc" -> name="Juan", encounters=5
[Server] memory: recupera 3 recuerdos relevantes de Juan
[Server] prompt: construye system prompt con personalidad + contexto Juan
[Server] think: LLM genera "Hola Juan! Me alegra verte sonreir..."
[Server] format: aplica estilo personalidad
[Server] store: guarda interaccion en SQLite
    |
    v HTTP Response
[Server --> Toy] {response:"Hola Juan! Me alegra verte sonreir..."}
    |
    v
[Toy] TTS convierte a voz y reproduce
[Toy] Servos animan "happy" (cabeza +10, brazo der +45)
[Toy] LEDs cambian a verde
```

## Decisiones de diseno clave

1. **Dos proyectos separados**: Server y toy tienen sus propios `pyproject.toml`, dependencias, y tests. Se despliegan en hardware diferente.

2. **Server generico**: No esta atado a Raspberry Pi. Funciona en cualquier ordenador con Python 3.10+.

3. **Prompts separados del codigo**: Las personalidades son solo archivos YAML en carpetas por idioma. No hay codigo Python dentro de las carpetas de idioma.

4. **Comunicacion WebSocket (v2)**: El toy mantiene una conexion `/ws` y envia frames binarios (audio/JPEG) + JSON de control; el server devuelve WAV + comandos. El POST HTTP `/chat` solo queda como fallback de diagnostico. *(v1 decia "No WebSocket, solo HTTP POST" — obsoleto.)*

5. **LLM flexible**: Cualquier endpoint compatible con OpenAI API. Por defecto apunta a LM Studio en localhost pero se configura via `.env`.

6. **Memoria semantica simple**: En lugar de embeddings vectoriales (que requieren mucha RAM), usa busqueda por keywords + factor de recencia. Optimizado para 8B models y hardware limitado.

7. **Modo mock**: Ambos proyectos soportan `MOCK_MODE=true` para desarrollo sin hardware fisico.

## Archivos fuente (sin __pycache__ ni .egg-info)

### philosopher-server (24 archivos .py + 6 YAML + 8 test files)
```
philosopher-server/
├── pyproject.toml
├── .env.example
├── README.md
├── philosopher/
│   ├── __init__.py
│   ├── main.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   └── personalities/
│   │       ├── es/{filosofo,curioso,poetico,amigo,sabio}.yaml
│   │       └── en/philosopher.yaml
│   ├── core/
│   │   ├── __init__.py, state.py, nodes.py, graph.py, orchestrator.py
│   ├── llm/
│   │   ├── __init__.py, client.py
│   ├── memory/
│   │   ├── __init__.py, short_term.py, long_term.py, manager.py
│   ├── personality/
│   │   ├── __init__.py, engine.py
│   ├── api/
│   │   ├── __init__.py, app.py
│   ├── events/
│   │   ├── __init__.py, bus.py
│   └── utils/
│       ├── __init__.py, logger.py
└── tests/
    ├── conftest.py
    ├── unit/{test_config,test_memory,test_personality,test_state,test_events}.py
    └── integration/test_nodes.py
```

### philosopher-toy (12 archivos .py + 2 test files)
```
philosopher-toy/
├── pyproject.toml
├── .env.example
├── README.md
├── toy_client/
│   ├── __init__.py
│   ├── main.py
│   ├── protocol/
│   │   ├── __init__.py, client.py
│   ├── vision/
│   │   ├── __init__.py, camera.py
│   ├── audio/
│   │   ├── __init__.py, stt.py, tts.py
│   ├── hardware/
│   │   ├── __init__.py, servos.py, leds.py
│   └── utils/
│       └── __init__.py
└── tests/
    ├── test_protocol.py
    └── test_hardware.py
```

## Tests

- **Server**: 28 tests (22 unitarios + 6 integracion) cubriendo config, memoria, personalidad, estado, eventos, y nodos LangGraph
- **Toy**: 3 tests (protocolo, servos, LEDs) en modo mock
- **Ejecucion**: `pytest tests/ -v` en cada proyecto

## Pendientes conocidos / mejoras futuras

- WebSocket para comunicacion bidireccional en tiempo real
- Embeddings vectoriales reales (sentence-transformers) para memoria semantica mas precisa
- Registro de nuevos rostros via API (actualmente hay que hacerlo manualmente)
- Dashboard web para ver estadisticas de memoria y conversaciones
- Wake word detection (por ejemplo "Hey Philosopher") en vez de boton fisico
- Animaciones mas complejas con secuencias de servo predefinidas
- Soporte offline para STT (actualmente usa Google Speech Recognition que requiere internet)
