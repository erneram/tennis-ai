<!-- PDF (docs/informe.pdf), from the repository root:
pandoc docs/informe.md -s --embed-resources --resource-path=docs --css docs/informe.css --metadata pagetitle="Serve Lab: informe escrito" -o /tmp/informe.html
"/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" --headless --no-pdf-header-footer --print-to-pdf=docs/informe.pdf /tmp/informe.html
-->

# Serve Lab: informe escrito

CC3092 Deep Learning y Sistemas Inteligentes, Proyecto 3: Casos de Uso de LLM. Universidad del Valle de Guatemala.

Ernesto David Ascencio Ramírez (23009) y Hugo Daniel Barillas Ajín (23556). 6 de noviembre de 2026.

Repositorio: https://github.com/erneram/tennis-ai/tree/proyecto-3

## 2.1 Definición del problema y usuario objetivo

### Usuario y tarea

El usuario es un jugador amateur de tenis que entrena solo, sin entrenador. Graba su saque de lado con el teléfono y quiere saber **qué corregir primero**. Sin Serve Lab, solo puede mirar su propio video y adivinar, o pagar a un entrenador.

Serve Lab recibe un video corto de un saque y devuelve:

- la fase con la principal oportunidad de mejora;
- los cuadros que respaldan la observación;
- un ejercicio para practicar;
- desde el segundo intento, una comparación con el intento anterior;
- al terminar la sesión, un resumen del progreso y el enfoque de la siguiente práctica.

Se eligió el saque porque es el único golpe que el jugador controla por completo: no depende de la pelota del rival, empieza desde una posición estática y se ejecuta igual en cualquier situación de juego. Lo que se corrige en la práctica pasa directo al partido.

### Por qué un LLM

La tarea es mirar cuadros de un video y responder en lenguaje natural sobre posiciones del cuerpo. Un modelo de visión y lenguaje hace las dos cosas sin datos de entrenamiento: reconoce a una persona, su brazo y su raqueta en una imagen, sigue instrucciones escritas y redacta la observación en español.

Alternativas consideradas:

- **Estimación de pose con reglas fijas** (MediaPipe Pose, YOLO Pose, ángulos medidos). Es más precisa para ángulos, pero no es un producto de LLM. Se movió al Proyecto Final (rama `proyecto-final`).
- **Un clasificador entrenado.** Con 39 videos de un amateur y tres profesionales no hay datos suficientes para entrenar sin sobreajustar.
- **Una sola llamada con todos los cuadros** (el prompt simple). Se midió como línea base: marca a los 11 profesionales y saca macro-F1 0.00 (sección 2.4).

### Criterios de éxito

Regla del producto: **una corrección equivocada es peor que una omitida.** Decirle a un jugador que corrija algo que estaba bien le hace daño; si el modelo no ve algo con claridad, debe decirlo. Por eso se mide:

- **Macro-F1 de la categoría** a mejorar, contra las respuestas de referencia.
- **Falsas alarmas en profesionales:** cuántos saques buenos reciben una corrección.
- **Problemas detectados:** cuántas caídas tempranas del brazo de JC encuentra.
- **Videos sin respuesta:** análisis válidos que terminan en error.
- **Solicitudes de nueva grabación:** si pide otra grabación cuando el video no sirve.
- **Latencia:** como máximo 45 s por video, lo que un visitante de la feria puede esperar.

## 2.2 Datos y pipeline del sistema

### Fuentes de datos

Los datos son videos propios, grabados con teléfono, de lado, siguiendo la guía `docs/grabacion.md`:

| Jugador | Videos | Analizables | Respuesta de referencia |
| --- | --- | --- | --- |
| JC, amateur | 28 | 20 | caída temprana del brazo de lanzamiento (`toss_arm_drops_early`) |
| JC, amateur | (de los 28) | 8 no | pedir otra grabación |
| pro01, pro02 y pro03, profesionales | 11 (2, 1 y 8) | 11 | sin problema claro (`no_clear_problem`) |

- **Formato:** `.MOV` y `.mp4` de unos segundos, a la resolución y cuadros por segundo del teléfono.
- **Calidad:** 8 de los 28 videos de JC no se pueden analizar. En 7 la cámara está detrás del jugador o en diagonal; en 1 el video termina antes del aterrizaje. Se quedan en el conjunto a propósito, para medir si el sistema pide otra grabación.
- **Privacidad:** los videos nunca entran a Git; viven en la carpeta ignorada `data/`. Solo JC dio permiso de usar su imagen, así que solo sus cuadros aparecen en el póster y en este informe.
- **Otras entradas del usuario:** el video que sube el visitante y, opcionalmente, un ID de jugador para guardar su historial.

### Preprocesamiento

No hay texto que limpiar ni fragmentar: la entrada es un video. No se usa RAG ni embeddings, porque la información que el modelo necesita (fases, criterios, ejercicios) es poca y va completa en cada prompt.

1. **Revisión del video.** OpenCV lee los cuadros por segundo y la duración, y rechaza videos muy cortos, muy largos, con pocos cuadros por segundo o archivos que no son video.
2. **Muestreo.** Se toman 32 cuadros espaciados uniformemente. Cada cuadro lleva su número dibujado en la esquina para que el modelo pueda nombrarlo.
3. **Recorte al jugador.** El modelo da un recuadro del jugador en tres cuadros. El código une los recuadros, los agranda hasta el alcance del saque y vuelve a decodificar los cuadros recortados. Así un jugador lejos de la cámara llena el cuadro.
4. **Ventana del saque.** El modelo dice dónde empieza y termina el saque, y se toman 32 cuadros nuevos solo dentro de esa ventana.
5. **Tamaños.** Los 32 cuadros para ubicar fases van a 512 px de lado. Los cuadros clave que se juzgan van a 1 024 px. Para el brazo de lanzamiento, la parte superior del cuerpo se recorta y se amplía.
6. **Respuestas guardadas.** En modo demo, una respuesta guardada solo se reutiliza para el mismo video, modelo, versión de prompt y versión de criterios.

### Arquitectura

![Arquitectura del sistema: cada caja azul es una llamada al modelo, cada caja blanca es código y cada caja verde es lo que ve el jugador.](images/informe/arquitectura.png)

Todas las respuestas del modelo son JSON restringido por un esquema (llguidance dentro de MLX-VLM). La app web (React) llama a una API local en Python (`app/api.py`), que corre el pipeline y el modelo en la misma Mac.

### Ventana de contexto

- Un análisis de JC consume unos **29 000 tokens** en todas sus llamadas; uno de profesional, unos 23 700 (`experiments/iterations.csv`).
- **Cómo se controla:**
  - El modelo nunca recibe el video completo, solo cuadros muestreados.
  - Los 32 cuadros para ubicar fases van a 512 px, unos 150 tokens por cuadro. Solo los cuadros clave van a 1 024 px.
  - El trabajo se divide en llamadas pequeñas. Cada una recibe solo los cuadros y las instrucciones de su pregunta, así ninguna se acerca al límite de contexto del modelo.
  - `max_tokens` es 600 por respuesta.

## 2.3 Selección del modelo y técnica de adaptación

### Modelos evaluados

El producto usa **Qwen3-VL-4B-Instruct**, cuantizado a 4 bits para MLX y ejecutado en la Mac del stand (Mac M5 Pro, 24 GB de memoria unificada).

Es un modelo de visión y lenguaje abierto, denso (no MoE) y en versión *instruct*: está ajustado para seguir instrucciones. Eso hace falta porque cada respuesta tiene que ser un JSON con un formato fijo, y un modelo base necesitaría ejemplos o ajuste fino para lograrlo. Tiene licencia Apache-2.0, pesa unos 3 GB y lee varias imágenes en una sola solicitud.

Modelos considerados (licencias revisadas en `docs/licenses.md`):

| Modelo | Licencia | Decisión |
| --- | --- | --- |
| Qwen3-VL-4B-Instruct, 4 bits | Apache-2.0 | Modelo principal. |
| Qwen3-VL-2B-Instruct, 4 bits | Apache-2.0 | Línea base: la mitad del tamaño. |
| Qwen3-VL-8B-Instruct, 4 bits | Apache-2.0 | Probado y descartado (tabla siguiente). |
| Qwen3.5-4B | Apache-2.0 | Candidato, no probado. |
| Gemma 4 E4B | Apache-2.0 | No había versión MLX de 4 bits. |
| Qwen2.5-VL-3B | Qwen Research License | Descartado: solo uso no comercial. |
| Gemma 3 | Gemma Terms of Use | Descartado: política de usos prohibidos y descarga restringida. |
| Llama 3.2 Vision | Llama Community License | Descartado: los modelos multimodales tienen restricciones en la UE. |
| GPT, Claude, Gemini (vía API) | Comerciales | No se usan en el producto (ver abajo). Claude se usó solo como juez para etiquetar el conjunto de evaluación. |

### Comparación de tamaños

Mismo prompt (v17) y misma semilla (0) en los tres tamaños, sobre los mismos 31 videos analizables:

| Modelo | Macro-F1 | Caídas de JC detectadas (de 20) | Falsas alarmas en profesionales (de 11) | Segundos por video de JC | Memoria máxima |
| --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B | 0.052 | 1 | 9 | 18 | 3.7 GB |
| **Qwen3-VL-4B** | **0.574** | **7** | **0** | 30 | 5.1 GB |
| Qwen3-VL-8B | 0.135 | 0 | 1 | 48 | 7.9 GB |

- **2B:** marca problemas a casi todos, incluidos 9 de los 11 profesionales.
- **8B:** tarda 1.6 veces más y usa 1.5 veces la memoria, y no fue mejor. Con los prompts v6 y v7 también siguió la redacción de la pregunta: marcó a 9 y luego a 11 de los 11 profesionales. El 4B con v6 marcó a 3. Un modelo más grande no corrigió ese sesgo; lo corrigieron los cambios de prompt (v8 en adelante).
- **4B:** es el único que encontró caídas del brazo sin dar falsas alarmas, con una espera aceptable para la feria.

Corridas en `eval/serve/runs/`, cada una con su fila en `experiments/iterations.csv`:

- 2B: `20261102-032818-full` y `20261102-033105-full`.
- 4B: `20261102-023129-full` y `20261102-023612-full`.
- 8B: `20261102-031416-full` y `20261102-032147-full`.
- 8B con v6 y v7: `20261101-pro-v6-8b`, `20261101-pro-v7-8b`, `20261101-jc-crop-v6-8b` y `20261101-jc-crop-v7-8b`.

Límites de esta comparación:

- Los prompts se escribieron con el 4B, así que el 8B podría mejorar con prompts propios.
- Se hizo con una sola semilla. Las versiones de prompt se compararon con 4 semillas porque el puntaje cambia mucho entre semillas (sección 2.5). La diferencia entre tamaños es grande, pero no está medida con esa misma precisión.

### Por qué un modelo abierto en la Mac y no uno de pago por internet

El enunciado permite modelos cerrados vía API (GPT, Claude, Gemini) o abiertos. Elegimos uno abierto que corre en la computadora por cinco razones:

1. **Privacidad.** Los videos muestran a personas: JC, los profesionales y los visitantes que suban su saque en la feria. Con un modelo local, ningún cuadro sale de la computadora. La app solo escucha en `127.0.0.1`, y el video de un visitante solo se copia a una carpeta temporal del sistema para analizarlo. Con una API, cada cuadro se enviaría a un tercero.
2. **Funciona sin internet.** Después de la primera descarga, todo corre sin conexión. El enunciado recomienda tener un plan de respaldo para la feria porque «la conexión a internet o una API externa pueden fallar» (recomendación 4). Aquí no hay API que pueda fallar.
3. **Costo cero por análisis.** Un análisis procesa unos 29 000 tokens, con 32 cuadros para las fases, cuadros clave y varias preguntas. Con una API se pagarían esos tokens en cada saque, y en la feria se analizan muchos. En la Mac el costo es $0. La energía no se midió; se estima en 0.3 a 0.5 Wh por análisis (unos 30 s a 40–60 W).
4. **Reproducibilidad.** El modelo es un archivo fijo: no cambia sin aviso, y la semilla y la temperatura están bajo nuestro control. Por eso se pueden repetir las 4 semillas de cada versión y comparar versiones con justicia. Un modelo por API puede actualizarse entre una corrida y otra.
5. **Cabe en el equipo.** Necesita unos 5 GB de los 24 GB de la Mac. Analiza un video de JC en 44 s en promedio y uno de profesional en 29 s, dentro de la espera razonable para un visitante.

Lo que se pierde: un modelo cerrado grande probablemente distinguiría mejor ángulos finos del cuerpo, que es la principal limitación del 4B. Esto no se midió: ningún modelo cerrado se evaluó como sistema. Además, el producto depende de una Mac con Apple Silicon.

### Dónde está el modelo

Los pesos del modelo **no están en el repositorio**: pesan unos 3 GB y GitHub no está hecho para archivos de ese tamaño. Tampoco hace falta subirlos, porque se descargan de Hugging Face (`mlx-community/Qwen3-VL-4B-Instruct-4bit`) la primera vez que se ejecuta la app y quedan en la caché local. El repositorio guarda todo lo demás: el nombre exacto del modelo en `configs/default.yaml`, los prompts, el código, las corridas y las métricas.

El modelo tampoco está desplegado en un servidor: la app corre en la Mac y se abre en el navegador de esa misma computadora. El enunciado pide el producto «funcionando en vivo en al menos un dispositivo», y ese dispositivo es la Mac del stand.

### Técnica de adaptación

Prompting *zero-shot*: los prompts no incluyen ejemplos resueltos. El modelo se usa tal como se descarga, sin cambiar sus pesos. Es transferencia de aprendizaje a gran escala: un modelo preentrenado con muchísimas imágenes y textos resuelve una tarea nueva solo con instrucciones.

Se combina con tres técnicas:

- **Salida estructurada.** Cada respuesta es un JSON restringido por un esquema (llguidance, dentro de MLX-VLM). El modelo solo puede elegir cuadros, fases, categorías y ejercicios que existen, así que no puede inventar opciones.
- **Varias llamadas pequeñas en vez de una grande.** Cada llamada responde una sola pregunta sobre pocos cuadros. El código toma las decisiones que el modelo hacía mal: ubica el impulso y el aterrizaje a partir del impacto, y elige el punto a mejorar como la fase más temprana con un problema confirmado.
- **Preguntas que se verifican.** Por cada categoría se hacen dos preguntas opuestas, y el chequeo solo cuenta si coinciden. Para el brazo de lanzamiento se hace una pregunta neutral con opciones, porque con sí / no el modelo daba la razón a la pregunta.

No hubo ajuste fino (ni completo ni LoRA). Hay 31 videos analizables de un solo amateur y tres profesionales: son muy pocos para entrenar sin sobreajustar, y las etiquetas sirven solo como clave de evaluación. El ajuste fino con cuadros etiquetados de más jugadores queda como trabajo futuro.

### Diseño de los prompts finales (v22)

El prompt de cada versión es un archivo, `strokes/serve/prompts/v22.yaml`. Su encabezado explica qué cambió respecto a la versión anterior, por qué, y qué resultó. Las categorías que se revisan y sus criterios están en `strokes/serve/criteria.yaml`, y los ejercicios en `strokes/serve/drills.yaml`. Las llamadas de un análisis son:

| Llamada | Qué recibe | Qué devuelve |
| --- | --- | --- |
| `player` | Primer cuadro, el del medio y el último | Recuadro del jugador, para recortar el video |
| `window` | 32 cuadros del video | Si se puede analizar (o el motivo para grabar de nuevo) y dónde empieza y termina el saque |
| `phases` | 32 cuadros dentro del saque, de 512 px, numerados | Un cuadro por cada una de las seis fases. Si parecen adivinadas, se repite con 1 de cada 2, 3 y 4 cuadros |
| `refine` | 12 cuadros cercanos al impacto | El cuadro exacto del impacto |
| `offsets` | (código, sin modelo) | Impulso 0.1 s antes del impacto, aterrizaje 0.6 s después |
| `checks` | Un cuadro clave de 1 024 px por pregunta | Para cada categoría, un campo `seen` con lo que ve, y luego sí / no / no está claro. Para el brazo de lanzamiento, una opción sobre la parte superior del cuerpo |
| `observation` | El cuadro del punto a mejorar | Hasta 3 frases en español, en tú, solo sobre ese punto |
| `summary` | Los intentos de la sesión | Resumen y enfoque de la siguiente práctica, sin inventar resultados |

Parámetros de generación (`configs/default.yaml`):

| Parámetro | Valor |
| --- | --- |
| `temperature` | 0.2 |
| `top_p` | 0.9 |
| `max_tokens` | 600 (el modelo escribe primero lo que ve y después juzga) |
| `seed` | 0 |

Con temperatura 0 las respuestas eran iguales para cualquier semilla, pero el puntaje fue menor (0.115 con v17 frente a 0.574).

## 2.4 Evaluación y medición de respuestas

### Conjunto de evaluación

- **39 casos** (el enunciado recomienda de 30 a 50): los 39 videos de la sección 2.2.
  - 31 analizables: 20 de JC con la caída temprana del brazo y 11 de profesionales sin problema claro.
  - 8 de JC que deben terminar en una solicitud de nueva grabación.
- **Cómo se construyó:** cada video se grabó de lado siguiendo `docs/grabacion.md`. Los videos de JC que no siguen la guía se quedaron como casos de nueva grabación.
- **Quién validó las referencias:** Claude, otro modelo de lenguaje, como juez (LLM como juez). Miró 20 cuadros alrededor del impacto de cada video y escribió la categoría, la fase y una nota (`eval/serve/claude_labels.csv`).
- **Pendiente:** el enunciado pide validar el juez contra evaluación humana en una muestra. Las plantillas `eval/serve/labels_ernesto.csv` y `eval/serve/labels_hugo.csv` todavía están vacías. Cuando se llenen, `uv run python -m eval.run_eval --agreement eval/serve/labels_ernesto.csv eval/serve/labels_hugo.csv` calcula el acuerdo (κ de Cohen).

### Métricas de calidad

- **Precisión:** cuando la app dice «esto es un problema», qué tan seguido acierta. Una precisión baja significa falsas alarmas.
- **Recall:** de los problemas reales, cuántos encuentra la app. Un recall bajo significa problemas que se escapan.
- **F1:** combina las dos, de 0 a 1. Solo es alto cuando ambas son altas.
- **Macro-F1 de la categoría** (la métrica principal): el F1 de cada respuesta posible, promediado con el mismo peso.
  - Un sistema que siempre dice «el brazo baja» saca 0.39, y uno que siempre dice «sin problema claro» saca 0.26.
  - Un video sin respuesta cuenta como una respuesta más que siempre está mal, así que un solo error cuesta cerca de un tercio del macro-F1.
  - Como guía para este proyecto: menos de 0.4 no es mejor que dar siempre la misma respuesta; alrededor de 0.7 es útil pero con errores visibles; más de 0.8 es bueno.
- **Falsas alarmas en profesionales** y **caídas de JC detectadas:** los dos tipos de error, contados directamente.
- **Videos sin respuesta:** análisis de un video válido que terminan sin una respuesta válida.
- **Solicitudes correctas de nueva grabación.**
- **Fallas inventadas:** observaciones que mencionan un problema que el sistema no confirmó (revisadas a mano, sección 2.5).

### Varianza y mitades

El modelo muestrea sus respuestas (temperatura 0.2). Al cambiar la semilla, el cuadro del impacto se movió hasta 18 cuadros y respuestas sobre el mismo cuadro cambiaron. Por eso:

- cada versión se corre con las semillas 0 a 3 y se reporta la media ± la desviación estándar (`uv run python -m experiments.seeds`);
- los videos se dividen en una mitad de desarrollo y una de prueba, y un cambio que solo ayuda en una mitad es sospechoso. Las dos mitades se vieron al elegir versiones, así que los puntajes son optimistas.

### Métricas operativas

Promedio por video de JC con v22 (4 semillas):

| Medida | Valor |
| --- | --- |
| Latencia | 44 s (29 s por video profesional) |
| Tokens | unos 29 000 (23 700 por video profesional) |
| Memoria máxima | 5.1 GB |
| Costo por análisis | $0 de API; energía estimada de 0.3 a 0.5 Wh |

### Líneas base

Con la semilla 0, sobre los 31 videos analizables:

| Sistema | Macro-F1 | Falsas alarmas en profesionales (de 11) | Caídas de JC detectadas (de 20) |
| --- | --- | --- | --- |
| Prompt simple: los cuadros clave sin criterios, una sola llamada | 0.00 | 11 | 0 |
| Modelo 2B con el mismo pipeline (v8) | 0.00 | 11 | 0 |
| Siempre «el brazo baja» (sin LLM) | 0.39 | 11 | 20 |
| Siempre «sin problema claro» (sin LLM) | 0.26 | 0 | 0 |
| **Sistema final (v22)** | **0.93** | **1** | **20** |

## 2.5 Resultados de iteraciones

### Tabla de iteraciones

Cada versión de prompt es un archivo (`strokes/serve/prompts/v*.yaml`) que nunca se edita después de tener resultados; su encabezado dice qué cambió, por qué y qué resultó. Cada corrida es una fila de `experiments/iterations.csv` (84 corridas). El modelo, los 32 cuadros y los parámetros de generación se mantuvieron fijos, salvo en las corridas de las líneas base y de la comparación de tamaños.

Con la semilla 0:

| Versión | Cambio respecto a la anterior | Macro-F1 | Falsas alarmas (de 11) | Caídas de JC (de 20) | Latencia | Tokens |
| --- | --- | --- | --- | --- | --- | --- |
| v8 | dos preguntas opuestas que deben coincidir | 0.15 | 0 | 0 | 31 s | 25 k |
| v9 | mirada cercana al impacto | 0.19 | 0 | 0 | 31 s | 30 k |
| v12 | pregunta neutral sobre el torso, en el cuadro del impulso | 0.28 | 0 | 3 | 36 s | 30 k |
| v13 | el código pone el impulso 0.1 s antes del impacto | 0.39 | 0 | 7 | 30 s | 28 k |
| v18 | tercer intento si las fases parecen adivinadas | 0.57 | 0 | 7 | 30 s | 28 k |
| v20 | hacia dónde apunta el brazo de lanzamiento | 0.80 | 1 | 16 | 34 s | 29 k |
| **v22** | **diagonal casi horizontal es caída y cuarto intento** | **0.93** | **1** | **20** | **44 s** | **29 k** |

Con las semillas 0 a 3 (4 corridas por versión):

| Versión | Cambio | Macro-F1 | Caídas de JC (de 80) | Falsas alarmas (de 44) | Videos sin respuesta (de 128) |
| --- | --- | --- | --- | --- | --- |
| v12 | pregunta neutral sobre el torso | 0.210 ± 0.052 | 8 | 3 | 6 |
| v17 | aterrizaje 0.6 s después del impacto | 0.336 ± 0.168 | 15 | 2 | 6 |
| v18 | tercer intento si las fases parecen adivinadas | 0.372 ± 0.144 | 16 | 2 | 2 |
| v19 | mayoría de tres respuestas (no adoptada) | 0.319 ± 0.084 | 10 | 2 | 2 |
| v20 | hacia dónde apunta el brazo de lanzamiento | 0.652 ± 0.178 | 58 | 4 | 2 |
| v21 | una diagonal más cerca de horizontal es caída | 0.714 ± 0.233 | 66 | 3 | 2 |
| **v22** | **cuarto intento** | **0.842 ± 0.084** | **67** | **3** | **0** |

En la mitad de prueba, v22 saca 0.879 de media (v21: 0.819).

### Regla de adopción

Una versión pasa a ser la versión por defecto solo si cumple todo esto:

- su media de macro-F1 es mayor que la de la versión actual;
- su mitad de prueba no es peor;
- en profesionales da 0 falsas alarmas con la semilla 0 y como máximo 2 de 44 en total;
- tarda como máximo 45 s por video de JC;
- las pruebas automáticas pasan;
- la app sigue pidiendo otra grabación con el video malo de la demo.

v19 variaba menos entre semillas pero sacó menos, así que no se adoptó. v20 rompió el límite de falsas alarmas (4 de 44), pero subió la media de 0.372 a 0.652; el equipo la adoptó por su recall y dejó la excepción escrita en su encabezado. v21 y v22 mejoraron a su versión anterior en todas las medidas.

### Ejemplos de fallas y cómo se abordaron

1. **La respuesta seguía la redacción, no la imagen (v5 a v7).** Primero la respuesta seguía el orden de la lista de opciones. Luego el modelo nunca dijo «no» (0 de 114 respuestas). Luego marcó a 11 de 11 profesionales. Pasó también con el modelo 8B. **v8:** dos preguntas opuestas que deben coincidir; si no coinciden, el chequeo queda como «no se ve claro».
2. **Fases adivinadas.** Con el jugador lejos de la cámara, el modelo devolvía seis cuadros seguidos: contaba en vez de mirar. **v8:** recorte al jugador (18 de 20 videos analizados). **v18 y v22:** si las fases parecen adivinadas, se intenta de nuevo con 1 de cada 2, 3 y 4 cuadros. Los videos sin respuesta bajaron de 6 a 2 y luego a 0 de 128.
3. **Impacto mal ubicado.** El impacto dura centésimas de segundo y cae entre dos cuadros muestreados. **v9:** una segunda mirada a 12 cuadros cercanos; el impacto quedó bien en 16 de 18 videos (antes 12).
4. **Cuadros que el modelo no encontraba.** El modelo elegía el aterrizaje con el jugador todavía en el aire. **v13 y v17:** el código pone el impulso 0.1 s antes del impacto y el aterrizaje 0.6 s después. El equilibrio al aterrizar se vio en 11 de 21 videos (antes 3).
5. **Fallas inventadas en la observación (alucinación).** La observación mencionaba problemas que el sistema no había confirmado. **v16:** la observación solo habla del punto a mejorar y nunca dice que otra cosa está mal. Las fallas inventadas bajaron de 4 a 1 de 7 observaciones revisadas.
6. **El modelo miraba el brazo equivocado (v10 a v20).** Es la falla de la figura siguiente.

![JC, 0.1 s antes del impacto: el brazo de lanzamiento (a la izquierda) ya bajó al hombro, mientras la mano de la raqueta sigue arriba.](images/informe/falla.jpg){width=60%}

En este cuadro, el brazo de lanzamiento de JC ya bajó al hombro. Con v18 el modelo describía el brazo como «extendido hacia afuera», pero respondía «arriba de la cabeza» en 61 de 82 respuestas: miraba la mano de la raqueta, que sí está arriba. **v20** le pide ubicar primero la mano de la raqueta y luego decir hacia dónde apunta el otro brazo, con opciones neutrales. La caída se vio en 58 de 80 saques (antes 16). **v21** cuenta como caída una diagonal más cerca de horizontal que de vertical: 66 de 80.

Lo que v22 todavía falla:

- sus 3 falsas alarmas vienen todas de un mismo saque, pro03_s03, cuyo cuadro de impacto se ubica tarde;
- no detecta 13 de las 80 caídas de JC en las 4 semillas.

## 2.6 Producto final y experiencia de usuario

### Descripción y flujo de uso

Serve Lab es una app web local. La página está hecha en React con componentes de The Gridcn, y `app/api.py` la sirve junto con su API en `http://127.0.0.1:7860`. La versión anterior en Gradio se guarda como respaldo y da las mismas respuestas.

![La app después de analizar un saque de JC.](images/app/result.jpg)

1. El jugador sube un saque grabado de lado o elige un video de la demo, y pulsa «Analizar saque».
2. El progreso se muestra en cuatro pasos numerados.
3. La app muestra:
   - la fase con la principal oportunidad de mejora, la observación y un ejercicio;
   - «Qué tan seguro está»: la confianza y cuántos chequeos pudo juzgar el modelo;
   - el cuadro clave de cada una de las seis fases, con su veredicto en palabras y color. «★ Aquí» marca el cuadro del que habla la observación, y un cuadro se abre en grande al tocarlo;
   - el video en cámara lenta con la fase actual;
   - cada chequeo con lo que vio el modelo («Qué revisamos en cada fase»);
   - desde el segundo intento, una comparación con el intento anterior.
4. «Terminar sesión» resume la sesión y propone el enfoque de la siguiente práctica. Con un ID de jugador, la sesión se agrega a su historial.
5. «Empezar de nuevo» borra el video, las respuestas y la sesión para el siguiente jugador.

### Decisiones de utilizabilidad

- **Incertidumbre.**
  - Cada fase se marca como se ve bien (verde), a mejorar (rojo), no se ve claro (gris) o sin revisar. El veredicto va también en palabras, no solo en color.
  - La app dice cuántos puntos pudo revisar («Pudimos revisar 5 de 6 puntos») y la confianza: alta, media o baja.
  - Un chequeo cuenta solo si dos preguntas opuestas coinciden; si no, se muestra «no se ve claro» en vez de una corrección.
  - Si no hay ningún problema confirmado, la respuesta es «sin problema claro», no una corrección forzada.
- **Errores.**
  - Si el video no sirve (ángulo incorrecto, jugador fuera de cuadro, poca luz, saque incompleto), la app muestra el motivo y cómo grabar de nuevo, en vez de la retroalimentación.
  - Videos muy cortos o muy largos, con pocos cuadros por segundo, de más de 200 MB o archivos que no son video se rechazan con un mensaje claro, sin errores técnicos.
  - Si el modelo no da una respuesta válida, la app lo dice y pide intentar de nuevo.
- **Tiempos de espera.**
  - Un análisis tarda de 15 a 45 s, y el progreso por pasos le muestra al jugador qué está pasando.
  - En modo demo (`uv run python -m app.api --demo`), los videos de la demo se analizan una vez al iniciar y después se muestran al instante, marcados como «respuesta guardada».
- **Sin internet.** La página incluye sus fuentes y no carga nada de internet.

## 2.7 Limitaciones, riesgos y consideraciones éticas

### Limitaciones técnicas

- **Alucinaciones.** El modelo describe una cosa y responde otra: describía el brazo de JC «extendido hacia afuera» y respondía «arriba de la cabeza». Hasta v16, las observaciones mencionaban problemas no confirmados. El esquema JSON impide inventar fases, categorías o ejercicios, pero no impide que el modelo juzgue mal un cuadro.
- **Ángulos finos.** El 4B no distingue bien ángulos del cuerpo, como cuánto se dobla un codo o una rodilla.
- **Varianza.** La respuesta cambia con la semilla: el cuadro del impacto se movió hasta 18 cuadros entre semillas. v22 tiene ± 0.084 de macro-F1 entre semillas.
- **Fecha de corte del modelo.** No afecta: el sistema no usa conocimiento del modelo sobre hechos, y los criterios vienen de `strokes/serve/criteria.yaml`.
- **Ventana de contexto.** Unos 29 000 tokens por análisis, repartidos en llamadas pequeñas; el modelo nunca recibe el video completo.
- **Costo y latencia.** $0 por análisis, pero 44 s por video de JC, y solo corre en una Mac con Apple Silicon.

### Limitaciones de la evaluación

- **Un solo juez.** Las referencias son de Claude y todavía no se validaron contra etiquetas del equipo (κ pendiente).
- **Puntajes optimistas.** Las dos mitades se vieron al elegir versiones.
- **Tarea casi binaria.** Todos los videos analizables de JC tienen la misma falla y ningún profesional tiene falla. El modelo podría distinguir a JC de los profesionales por otras cosas (ropa, cancha, cámara) y no por el brazo.
- **Pocos jugadores.** Un amateur y tres profesionales, todos grabados de lado. Los resultados no se pueden generalizar a otros jugadores o ángulos.

### Riesgos y medidas

- **Sesgo.** Con un solo amateur, el sistema se ajustó a la falla de JC. Medida: la regla de adopción limita las falsas alarmas en profesionales, y la app dice «sin problema claro» cuando no confirma nada.
- **Privacidad.**
  - El modelo corre en la Mac y ningún cuadro sale de la computadora.
  - La app escucha solo en `127.0.0.1`, y el respaldo en Gradio tiene la analítica desactivada.
  - Los videos del proyecto y los historiales quedan en carpetas ignoradas por Git. El video de un visitante solo se copia a una carpeta temporal del sistema, y su respuesta no se guarda en el caché de la demo.
  - Solo se publican imágenes de JC, que dio su consentimiento.
- **Prompt injection.**
  - El usuario no escribe texto libre que llegue a los prompts: sube un video, y el modelo solo responde JSON restringido por esquema.
  - Lo único que vuelve de la página al prompt del resumen son los intentos de la sesión. `app/api.py` (`valid_attempt`) los valida: hora con formato fijo, y fase, categoría y ejercicio de las listas conocidas.
- **Uso indebido.** El producto no es un diagnóstico médico ni reemplaza a un entrenador, y no mide números (ángulos, alturas, velocidades). La observación nunca da números.

### Cuándo no debería usarse

- Para evaluar lesiones o dolor, o para decidir algo médico.
- Para otros golpes que no sean el saque, o para analizar partidos completos.
- Con videos de frente o de espaldas, o con varios jugadores en cuadro.
- Como juicio de nivel profesional: la precisión no está validada con expertos.

## 2.8 Conclusiones

### Desempeño final

El sistema final usa la versión de prompt v22 con Qwen3-VL-4B en la Mac:

- macro-F1 de **0.842 ± 0.084** en 4 semillas (0.93 con la semilla 0); en la mitad de prueba, 0.879;
- detecta la caída temprana del brazo de JC en **67 de 80** saques;
- da **3 falsas alarmas en 44** saques profesionales, todas en el mismo saque;
- **ningún** video válido sin respuesta en 128 análisis;
- 44 s y unos 29 000 tokens por video de JC, 5.1 GB de memoria y $0 por análisis.

En el contexto del problema, el sistema separa el saque de JC de los saques profesionales y casi nunca le pide a un profesional que corrija algo. Pero el conjunto es pequeño y casi binario, y las referencias son de un solo juez. Por eso los números muestran que el método funciona en estos videos, no que funcione con cualquier jugador.

### Aprendizajes

- **La redacción del prompt decide la respuesta.** Con sí / no, el modelo daba la razón a la pregunta; un modelo más grande no lo corrigió. Lo corrigieron dos preguntas opuestas y preguntas neutrales con opciones.
- **El código debe hacer lo que el modelo hace mal.** Ubicar el aterrizaje y elegir el punto a mejorar funcionan mejor como reglas fijas; el modelo se queda con lo que solo él puede hacer: mirar.
- **Leer lo que el modelo describe.** El campo `seen` mostró que el modelo veía el brazo bien pero respondía sobre la mano de la raqueta. Esa diferencia llevó al cambio con más efecto (v20).
- **Una semilla no basta.** El mismo prompt saca de 0.21 a 0.57 según la semilla (v17). Las versiones se comparan con la media de varias semillas.
- **Un cambio a la vez, y las versiones como archivos.** Cada versión con su encabezado y cada corrida en `iterations.csv` permitieron saber qué cambio causó cada mejora.

### Trabajo futuro

- Validar las referencias con las etiquetas del equipo y reportar κ.
- Grabar más jugadores amateurs, con otras fallas y otros ángulos, para romper la tarea casi binaria.
- Separar de verdad una mitad de prueba que no se mire al elegir versiones.
- Ajuste fino (LoRA) con cuadros etiquetados de más jugadores.
- Estimación de pose para medir ángulos (Proyecto Final, rama `proyecto-final`).
- Probar el modelo 8B con prompts escritos para él.

## 2.9 Enlace al repositorio

https://github.com/erneram/tennis-ai/tree/proyecto-3 (público, rama `proyecto-3`).

El repositorio contiene:

- el código completo: muestreo y preprocesamiento de video (`core/`), pipeline del saque (`strokes/serve/`), app (`app/`, `web/`);
- los prompts versionados (`strokes/serve/prompts/`);
- el conjunto de evaluación, sus referencias y los scripts de evaluación (`eval/`);
- el registro de iteraciones (`experiments/iterations.csv`) y el póster (`docs/poster/`).

El README explica cómo instalar y ejecutar el producto y cómo reproducir las métricas. El proyecto no usa llaves de API; la configuración opcional va en un archivo `.env` excluido por `.gitignore`.
