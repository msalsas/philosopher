# Philosopher - Flujo End-to-End del Producto

## Que es Philosopher

Philosopher es un peluche inteligente con el que puedes hablar. Tiene personalidad propia, recuerda a las personas que le visitan, detecta emociones por la cara, mueve la cabeza con un servo, y responde con voz natural en espanol. No es un simple altavoz con comandos: es un companero conversacional que evoluciona con cada interaccion.

---

## Arquitectura de 3 Capas (donde corre cada cosa)

```
+------------------------------------------------------------------+
|  CAPA 1: Servidor LLM externo (OpenAI-compatible)                |
|  (Ollama, llama.cpp, vLLM, LM Studio... cualquier PC)            |
|  Corre el modelo de IA (LLM) - ej: Llama 3.1 8B                  |
|  Solo recibe texto y devuelve texto. No tiene memoria ni         |
|  personalidad propia.                                            |
+------------------------------------------------------------------+
                              ^
                              | HTTP (red local)
                              v
+------------------------------------------------------------------+
|  CAPA 2: Servidor Philosopher (cualquier ordenador)               |
|  El "cerebro" del sistema. Corre donde quieras (PC viejo,       |
|  laptop, mini PC, etc).                                          |
|  - Gestiona la personalidad (como responde)                      |
|  - Gestiona la memoria a corto y largo plazo                     |
|  - Orquesta el pipeline conversacional con LangGraph             |
|  - Expone un WebSocket (tiempo real) para el peluche; HTTP para   |
|    diagnostico                                                   |
+------------------------------------------------------------------+
                              ^
                              | WebSocket (WiFi)
                              v
+------------------------------------------------------------------+
|  CAPA 3: Raspberry Pi Zero WH dentro del peluche                            |
|  El "cuerpo" del sistema. El hardware fisico del juguete.        |
|  - Camara CSI: captura imagen (el server ve cara y emocion)       |
|  - Microfono USB: captura voz (el server la transcribe: STT)      |
|  - Altavoz: reproduce el audio WAV que sintetiza el server (TTS)  |
|  - Servo de cabeza (los brazos existen en el mapa de pines pero  |
|    estan deshabilitados en este montaje)                         |
+------------------------------------------------------------------+
```

La razon de esta separacion es flexibilidad: puedes cambiar el modelo de IA sin tocar el peluche, puedes mover el servidor a otro ordenador, y el peluche solo se encarga de entrada/salida fisica.

---

## Flujo Completo: Una Conversacion Tipica

### Paso 0: Encendido y Conexion

1. Enciendes el peluche (se alimenta por USB-C o bateria)
2. La Raspberry Pi Zero WH arranca, se conecta al WiFi
3. Se conecta al servidor Philosopher por la red
4. El servidor verifica que puede hablar con el LLM
5. El peluche hace un movimiento de cabeza como gesto de saludo
6. Todo listo - Philosopher "esta vivo"

---

### Paso 1: Deteccion Facial (la camara ve)

**El usuario se acerca al peluche.**

- La camara del peluche captura una imagen por turno (modo on-speech, el que
  ahorra energia en el montaje del peluche; existe un modo continuo pero pide
  mas potencia) y la envia al servidor
- El servidor detecta si hay una cara en el encuadre
- Si es una cara nueva: la aprende, genera un ID unico, y lo guarda
- Si es una cara conocida: la reconoce y recupera su nombre
- Ademas, analiza la expresion facial para detectar la emocion:
  - Feliz, triste, enfadado, sorprendido, asustado, neutral

**Que envia al servidor:**
```
- ID del rostro (ej: "face_a3f7d2")
- Nombre (si lo conoce, ej: "Maria" o null si es nuevo)
- Emocion detectada (ej: "happy")
```

**Memoria:** Si Maria ya hablo antes con el peluche, el servidor busca en su memoria a largo plazo y encuentra datos como: "Maria tiene un perro llamado Toby", "Maria le gusta hablar de musica", "la ultima vez Maria estaba triste por el trabajo".

---

### Paso 1b: Que pasa si te oye pero no te ve?

**Si hablas desde otra habitacion, de espaldas, o con poca luz:**

- El microfono te oye igualmente y transcribe tu voz
- Como la camara no ve tu cara, no sabe quien eres ni que emocion tienes
- **Pero te responde de todas formas** - la conversacion fluye normalmente
- La unica diferencia: no recupera recuerdos personales asociados a tu cara, y la personalidad no adapta el tono a una emocion especifica
- Si luego te acercas y te ve, en el siguiente mensaje ya te reconoce y recupera todo el contexto

**Resumen:** la vision mejora la experiencia (reconoce quien eres y como te sientes) pero no es obligatoria. Philosopher siempre te escuchara y respondara mientras le hables.

---

### Paso 2: Escucha de Voz (STT - Speech to Text)

**El usuario habla al peluche.**

- El microfono esta siempre escuchando en segundo plano
- Usa deteccion de actividad de voz (VAD): detecta cuando empiezas a hablar y cuando paras
- No necesitas pulsar ningun boton para hablar - es conversacion natural
- Convierte el audio a texto usando reconocimiento de voz en espanol
- Si no entiende bien, pide amablemente que repitas

**Ejemplo:**
- Usuario dice: "Hola Philosopher, como estas hoy?"
- El sistema transcribe: "Hola Philosopher, como estas hoy?"

---

### Paso 3: El Servidor Procesa (el cerebro trabaja)

**El Raspberry Pi Zero WH envia al servidor:**
```json
{
  "message": "Hola Philosopher, como estas hoy?",
  "face_id": "face_a3f7d2",
  "face_name": "Maria",
  "emotion": "happy"
}
```

**El servidor ejecuta 6 pasos secuenciales (pipeline LangGraph):**

#### Paso 3.1 - Percepcion (quien habla?)
- Recupera de la memoria a largo plazo todo lo que sabe sobre Maria
- Es la primera vez? Es alguien nuevo? Es alguien habitual?
- Si Maria ha venido 10 veces, lo sabe. Si es la primera, tambien.

#### Paso 3.2 - Memoria (que recuerdo relevante?)
- Busca en la memoria a corto plazo: la conversacion actual (ultimos 10 mensajes)
- Busca en la memoria a largo plazo: recuerdos relevantes a lo que Maria acaba de decir
  - Si Maria pregunta "como estas", busca recuerdos recientes de Maria
  - Si Maria dice "mi perro", busca recuerdos sobre "Toby"
- Puntua los recuerdos por relevancia y recencia (lo mas reciente y relevante gana)

#### Paso 3.3 - Construccion del Prompt (la personalidad entra en juego)
- Carga la personalidad activa (ej: "filosofo", "curioso", "poetico"...)
- Construye el prompt del sistema con:
  - La personalidad completa (como debe comportarse)
  - El nombre de la persona ("estas hablando con Maria")
  - La emocion detectada ("Maria parece feliz")
  - Los recuerdos relevantes recuperados
- Todo optimizado para que quepa en el contexto de un modelo 8B (es decir, conciso)

**Ejemplo de prompt que se envia al LLM:**
```
Eres Philosopher, un peluche sabio y tierno. Hablas con calma y serenidad.
No das sermones: plantas preguntas suaves que invitan a reflexionar.
Maximo 3 frases por respuesta. Espanol natural.
Estas hablando con Maria.
La persona parece feliz.
Recuerdo: Maria tiene un perro llamado Toby.
Recuerdo: A Maria le gusta hablar de musica.
```

#### Paso 3.4 - Pensamiento (llamada al LLM)
- El prompt del sistema + el historial reciente de la conversacion + el mensaje actual se envian al LLM
- El LLM genera una respuesta en texto plano
- Timeout de 30 segundos - si tarda mas, se maneja el error

**Respuesta del LLM:** "Hola Maria! Me alegra verte con esa sonrisa. Toby ya te ha sacado a pasear hoy?"

#### Paso 3.5 - Formato (pulir la respuesta)
- Aplica restricciones de la personalidad (maximo X caracteres)
- Si el LLM fallo o devolvio vacio, usa un mensaje de fallback amable

#### Paso 3.6 - Almacenamiento (guardar para el futuro)
- Guarda la interaccion en memoria a corto plazo (ultimos 10 mensajes)
- Genera un resumen y lo guarda en memoria a largo plazo (SQLite)
- Si habia un rostro, actualiza el contador de encuentros
- Extrae palabras clave del mensaje para futuras busquedas

---

### Paso 4: Respuesta por Voz (TTS - Text to Speech)

**El servidor responde al Raspberry Pi Zero WH:**
```json
{
  "response": "Hola Maria! Me alegra verte con esa sonrisa. Toby ya te ha sacado a pasear hoy?",
  "emotion": "happy",
  "turn": 3
}
```

- El Raspberry Pi Zero WH recibe el texto
- Limpia marcadores de accion (asteriscos, etc.)
- Divide en trozos si es muy largo (mas de 400 caracteres)
- Sintetiza voz en espanol usando TTS (Edge TTS - voz masculina "Alvaro")
- Reproduce por el altavoz conectado

---

### Paso 5: Animacion Fisica (el peluche "cobra vida")

**Simultaneamente a la voz, el peluche se mueve:**

- **Servo de cabeza:**
  - Sigue la cara con la mirada (gaze) y hace un leve "mirar alrededor" en reposo
  - En negativos (triste/enfadado) niega ligeramente con la cabeza
  - Los brazos estan en el mapa de pines pero deshabilitados en este montaje
    (evitan el brownout del rail de 5V durante la respuesta)

**El usuario ve y oye al peluche responder de forma natural.**

---

### Paso 6: La Conversacion Continua

El ciclo se repite:

1. La camara vuelve a capturar la cara de Maria en el siguiente turno
2. El microfono sigue escuchando
3. Cada nuevo mensaje de Maria pasa por todo el pipeline
4. La memoria a corto plazo mantiene el contexto de la conversacion actual
5. La memoria a largo plazo recupera recuerdos relevantes

**Ejemplo de conversacion continua:**

```
Maria: "Hola Philosopher, como estas hoy?"
Peluche: "Hola Maria! Me alegra verte con esa sonrisa. Toby ya te ha sacado a pasear hoy?"
         [sigue a Maria con la mirada]

Maria: "Si, fuimos al parque esta manana."
Peluche: "El parque... que buen lugar para despejar la mente. Toby disfruta corriendo?"
         [inclina cabeza con curiosidad]

Maria: "Le encanta, no para de correr detras de las palomas."
Peluche: "Ja, los perros y las palomas... una historia sin fin. Y tu, como te sentiste paseando?"
         [gesto curioso con la cabeza]
```

Fijate como:
- Recuerda el nombre de Maria
- Recuerda que tiene un perro llamado Toby
- La conversacion fluye naturalmente
- Responde con maximo 2-3 frases (optimizado para 8B)
- La emocion afecta el tono y los gestos

---

## Memoria: Como Funciona

### Memoria a Corto Plazo
- Guarda los ultimos 10 mensajes de la conversacion actual
- Se pierde al reiniciar (esta en RAM)
- Sirve para mantener el contexto de "que estabamos hablando"

### Memoria a Largo Plazo
- Guardada en una base de datos SQLite persistente
- **Recuerda a las personas:** cada cara tiene un ID, nombre, y contador de encuentros
- **Recuerda conversaciones:** cada interaccion se resume y se guarda con palabras clave
- **Busqueda inteligente:** cuando alguien habla, busca recuerdos relevantes por palabras clave + recencia
- **No usa embeddings vectoriales** (demasiado pesados para 8B): usa keywords + recency scoring

### Ejemplo practico de memoria

Maria visita el peluche por primera vez:
```
Maria: "Me llamo Maria y tengo un perro llamado Toby."
[Guardado en memoria largo plazo: "Maria tiene un perro llamado Toby"]
```

Una semana despues, Maria vuelve:
```
[El sistema reconoce la cara de Maria]
[Recupera de memoria: "Maria tiene un perro llamado Toby", "le gusta el parque"]

Peluche: "Hola de nuevo, Maria! Como esta Toby?"
```

---

## Personalidades

Philosopher tiene varias personalidades que puedes cambiar. Cada una responde de forma diferente:

| Personalidad | Estilo |
|-------------|--------|
| **Filosofo** (por defecto) | Sabio, tranquilo, planta preguntas reflexivas, usa metaforas |
| **Curioso** | Inquisitivo, pregunta mucho, anima a explorar |
| **Poetico** | Usa imagenes poeticas, habla con mas fluidez artistica |
| **Amigo** | Cercano, coloquial, como un amigo de confianza |
| **Sabio** | Mas sereno y contemplativo, tono de anciano sabio |

Todas estan en espanol pero el sistema soporta multi-idioma. Para anadir un idioma nuevo, solo hace falta crear una carpeta (ej: `fr/` para frances) con los archivos YAML.

---

## Modos de Operacion

### Modo Conversacion (por defecto)
- El peluche inicia la conversacion cuando detecta una cara
- Escucha continuamente
- Responde por voz con animaciones

### Modo Servidor API
- El peluche habla por WebSocket (el camino en tiempo real); ademas hay una
  API HTTP `/chat` para diagnostico
- Cualquier cliente puede enviar mensajes por HTTP para pruebas
- Permite integrar Philosopher con apps web, movil, etc.

### Modo Mock
- Para desarrollo sin hardware
- Simula servos, camara y microfono
- Permite programar y probar sin tener el peluche fisico

### Modo Texto (fallback)
- Si el microfono no funciona, puedes escribir por teclado
- Util para pruebas y depuracion

---

## Eventos del Sistema (17 tipos)

El sistema emite eventos para que diferentes partes se comuniquen sin acoplarse:

- `CONVERSATION_STARTED` - Empieza una conversacion nueva
- `CONVERSATION_ENDED` - La conversacion termina (no detecta cara tras X tiempo)
- `FACE_DETECTED` / `FACE_RECOGNIZED` / `FACE_NEW` - Eventos de reconocimiento facial
- `EMOTION_CHANGED` - Cambia la emocion detectada
- `USER_MESSAGE` / `ASSISTANT_MESSAGE` - Mensajes enviados/recibidos
- `MEMORY_STORED_SHORT` / `MEMORY_STORED_LONG` - Memoria guardada
- `LLM_REQUEST` / `LLM_RESPONSE` / `LLM_ERROR` - Comunicacion con el LLM
- `TTS_STARTED` / `TTS_FINISHED` - Sintesis de voz
- `SERVO_ANIMATION` - Movimientos fisicos
- `ERROR` - Errores del sistema

---

## Que pasa sin conexion?

- Si el peluche pierde la conexion con el servidor: intenta reconectar automaticamente
- Si el servidor no puede hablar con el LLM: devuelve un mensaje de error amable
- Si el microfono falla: el peluche entra en modo texto (escribe por teclado)
- Si la camara falla: sigue funcionando sin reconocimiento facial

El sistema esta disenado para degradarse gracefulmente - si algo falla, lo demas sigue funcionando.

---

## Resumen del Flujo en una Imagen

```
[Usuario se acerca] 
      |
      v
[Camara] --> detecta cara + emocion
      |
      v
[Microfono] --> escucha voz --> transcribe a texto
      |
      v
[Raspberry Pi Zero WH] --> envia por WiFi al servidor
      |
      v
[Servidor Philosopher]
  1. Recupera memoria de esa persona
  2. Busca recuerdos relevantes
  3. Construye prompt con personalidad + contexto
  4. Pregunta al LLM
  5. Formatea la respuesta
  6. Guarda todo en memoria
      |
      v
[Raspberry Pi Zero WH] --> recibe respuesta de texto
      |
      +-->[TTS] --> reproduce voz por altavoz
      +-->[Servo] --> mueve la cabeza (gaze + idle)
      |
      v
[Usuario oye y ve la respuesta]
      |
      v
[Repite desde el paso 1]
```

---

## Caracteristicas Clave del Producto

1. **Privacidad**: todo corre en tu red local. El LLM esta en tu ordenador, el servidor en tu red, solo el texto viaja entre ellos. Nada va a la nube.

2. **Personalidad persistente**: el peluche siempre tiene la misma "personalidad" hasta que tu la cambies.

3. **Memoria real**: recuerda conversaciones pasadas, nombres, detalles. No es un chatbot de "cada conversacion es nueva".

4. **Deteccion emocional**: responde de forma diferente si estas feliz, triste, enfadado o sorprendido.

5. **Multi-idioma**: disenado para anadir idiomas nuevos facilmente.

6. **Hardware flexible**: si algun componente falla o no esta disponible (camara, microfono, algun servo), el resto sigue funcionando.

7. **Modelo ligero**: optimizado para LLMs de 8B parametros, funciona en hardware modesto.
