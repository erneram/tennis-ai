<!-- PDF (docs/informe.pdf), from the repository root:
pandoc docs/informe.md -s --embed-resources --resource-path=docs --css docs/informe.css --metadata pagetitle="Serve Lab: informe escrito" -o /tmp/informe.html
"/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" --headless --no-pdf-header-footer --print-to-pdf=docs/informe.pdf /tmp/informe.html
-->

<div class="cover">
<div class="top">
<p class="school">Universidad del Valle de Guatemala</p>
<p>Facultad de Ingeniería</p>
<p>CC3092 Deep Learning y Sistemas Inteligentes</p>
</div>
<div class="middle">
<p class="title">Serve Lab: análisis del saque de tenis con un modelo de visión y lenguaje</p>
<p>Proyecto 3: Casos de Uso de LLM</p>
<p>Informe escrito</p>
</div>
<div class="authors">
<p>Ernesto David Ascencio Ramírez, 23009</p>
<p>Hugo Daniel Barillas Ajín, 23556</p>
</div>
<p class="date">Guatemala, 6 de noviembre de 2026</p>
</div>

## 2.1 Definición del problema y usuario objetivo

**Usuario y tarea.** El usuario es un jugador amateur de tenis que entrena solo, sin entrenador. Graba su saque de lado con el teléfono y quiere saber qué corregir primero. Sin Serve Lab, solo puede mirar su video y adivinar, o pagar a un entrenador. Serve Lab devuelve la fase con la principal oportunidad de mejora, los cuadros que la muestran y un ejercicio. Desde el segundo intento compara con el anterior, y al final de la sesión resume el progreso. Se eligió el saque porque es el único golpe que el jugador controla por completo: no depende de la pelota del rival.

**Por qué un LLM.** La tarea es mirar cuadros de un video y responder en lenguaje natural sobre posiciones del cuerpo. Un modelo de visión y lenguaje lo hace sin datos de entrenamiento y redacta la observación en español. Alternativas consideradas:

- Estimación de pose con reglas fijas: es más precisa para ángulos, pero no es un producto de LLM. Pasó al Proyecto Final.
- Un clasificador entrenado: con 39 videos no hay datos suficientes.
- Una sola llamada con todos los cuadros: se midió como línea base y saca macro-F1 0.00 (sección 2.4).

**Criterios de éxito.** Regla del producto: una corrección equivocada es peor que una omitida. Se mide el macro-F1 de la categoría a mejorar, las falsas alarmas en saques profesionales, las caídas del brazo detectadas en el amateur, los videos sin respuesta, las solicitudes de nueva grabación y la latencia (como máximo 45 s).

## 2.2 Datos y pipeline del sistema

**Datos.** Videos propios, grabados con teléfono, de lado, en formato .MOV y .mp4 de unos segundos:

| Jugador | Videos | Analizables | Respuesta de referencia |
| --- | --- | --- | --- |
| JC, amateur | 28 | 20 | caída temprana del brazo de lanzamiento |
| JC, amateur (los mismos 28) | | 8 no | pedir otra grabación (7 de espaldas o en diagonal, 1 incompleto) |
| Tres profesionales | 11 | 11 | sin problema claro |

Los videos nunca se suben al repositorio. Solo JC dio permiso de usar su imagen, así que solo sus cuadros aparecen en el póster y en este informe.

**Preprocesamiento.** No se usa RAG ni embeddings: la información que el modelo necesita (fases, criterios, ejercicios) es poca y va completa en cada prompt.

1. OpenCV revisa los cuadros por segundo y la duración, y rechaza videos muy cortos, muy largos o que no son video.
2. Se toman 32 cuadros espaciados uniformemente, cada uno con su número dibujado.
3. El modelo da un recuadro del jugador y los cuadros se recortan a ese recuadro.
4. El modelo dice dónde empieza y termina el saque, y se toman 32 cuadros nuevos solo dentro de esa ventana.
5. Los cuadros para ubicar fases van a 512 px; los cuadros que se juzgan, a 1 024 px.

![Arquitectura: cada caja azul es una llamada al modelo, cada caja blanca es código y cada caja verde es lo que ve el jugador.](images/informe/arquitectura.png)

**Ventana de contexto.** Un análisis consume unos 29 000 tokens en todas sus llamadas (23 700 en un video profesional). El modelo nunca recibe el video completo: cada llamada recibe solo los cuadros de su pregunta, cada cuadro de 512 px cuesta unos 150 tokens, y cada respuesta tiene un máximo de 600 tokens.

## 2.3 Selección del modelo y técnica de adaptación

**Modelo.** Qwen3-VL-4B-Instruct, cuantizado a 4 bits con MLX, en una Mac M5 Pro de 24 GB. Es abierto, denso (no MoE), de 4 000 millones de parámetros y en versión instruct, necesaria para responder siempre en JSON sin ejemplos. Tiene licencia Apache-2.0, pesa unos 3 GB y lee varias imágenes por solicitud. Se descartaron Qwen2.5-VL-3B (uso no comercial), Gemma 3 y Llama 3.2 Vision por sus licencias, y Gemma 4 E4B porque no tenía versión MLX de 4 bits.

Mismo prompt (v17) y misma semilla en tres tamaños:

| Modelo | Macro-F1 | Caídas de JC (de 20) | Falsas alarmas (de 11) | Segundos por video | Memoria |
| --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B | 0.052 | 1 | 9 | 18 | 3.7 GB |
| **Qwen3-VL-4B** | **0.574** | **7** | **0** | **30** | **5.1 GB** |
| Qwen3-VL-8B | 0.135 | 0 | 1 | 48 | 7.9 GB |

El 2B marca a casi todos. El 8B es más lento, usa más memoria y no fue mejor; con prompts anteriores también marcó a 11 de 11 profesionales. El 4B es el único que encontró caídas sin falsas alarmas. Límite: los prompts se escribieron para el 4B y la comparación usa una sola semilla.

**Por qué abierto y local, y no GPT, Claude o Gemini por API:**

- Privacidad: ningún cuadro de un jugador sale de la computadora.
- Funciona sin internet, como recomienda el enunciado para la feria.
- Costo de $0 por análisis; con una API se pagarían 29 000 tokens por saque.
- Reproducible: el modelo no cambia sin aviso, y la semilla y la temperatura están bajo control.
- Lo que se pierde: un modelo cerrado grande probablemente vería mejor los ángulos finos. No se midió.

Los pesos no están en el repositorio: se descargan de Hugging Face (mlx-community/Qwen3-VL-4B-Instruct-4bit) en la primera ejecución. La app no está en un servidor: corre en la Mac del stand.

**Técnica.** Prompting zero-shot, sin ejemplos y sin cambiar los pesos: transferencia de aprendizaje a gran escala. Se combina con tres técnicas:

- Salida JSON restringida por un esquema: el modelo solo puede elegir cuadros, fases, categorías y ejercicios que existen.
- Varias llamadas pequeñas: cada una responde una sola pregunta, y el código toma las decisiones que el modelo hacía mal.
- Dos preguntas opuestas por categoría, que deben coincidir. Para el brazo de lanzamiento, una pregunta neutral con opciones.

No hubo ajuste fino: 31 videos analizables son muy pocos para entrenar sin sobreajustar.

**Prompts finales (versión v22).** Un análisis hace estas llamadas, todas con instrucciones de sistema en inglés y respuesta en JSON:

| Llamada | Qué devuelve |
| --- | --- |
| Jugador | Recuadro del jugador en tres cuadros |
| Ventana | Si el video sirve (o el motivo para grabar de nuevo), y el inicio y el fin del saque |
| Fases | El cuadro de cada una de las seis fases; si parecen adivinadas, se repite con 1 de cada 2, 3 y 4 cuadros |
| Impacto | El cuadro exacto del impacto, entre 12 cuadros cercanos. El impulso (−0.1 s) y el aterrizaje (+0.6 s) los pone el código |
| Chequeos | Por categoría, lo que ve el modelo y luego sí, no o no está claro |
| Observación | Hasta 3 frases en español sobre el punto a mejorar |
| Resumen | Progreso de la sesión y enfoque de la siguiente práctica |

Parámetros: temperature 0.2, top_p 0.9, max_tokens 600, semilla 0. Con temperatura 0 el puntaje fue menor (0.115 frente a 0.574 con v17).

## 2.4 Evaluación y medición de respuestas

**Conjunto de evaluación.** 39 videos: 31 analizables (20 de JC con la caída del brazo y 11 profesionales sin problema claro) y 8 que deben terminar en una solicitud de nueva grabación. Las respuestas de referencia son de Claude como juez (LLM como juez), que miró 20 cuadros alrededor del impacto de cada video. **Pendiente:** validarlas contra las etiquetas del equipo y reportar el acuerdo (κ de Cohen).

**Métricas.**

- Macro-F1 de la categoría: el F1 de cada respuesta posible, con el mismo peso. Un video sin respuesta cuenta como error y cuesta cerca de un tercio del puntaje. Como guía: menos de 0.4 no es mejor que dar siempre la misma respuesta, y más de 0.8 es bueno.
- Falsas alarmas en profesionales (precisión) y caídas de JC detectadas (recall).
- Videos sin respuesta y solicitudes correctas de nueva grabación.
- Media ± desviación estándar en 4 semillas, porque cambiar la semilla movió el cuadro del impacto hasta 18 cuadros. Los videos se dividen en una mitad de desarrollo y una de prueba.
- Operativas, con v22: 44 s por video de JC y 29 s por video profesional, unos 29 000 tokens, 5.1 GB de memoria y $0 por análisis.

**Líneas base** (semilla 0, 31 videos analizables):

| Sistema | Macro-F1 | Falsas alarmas (de 11) | Caídas de JC (de 20) |
| --- | --- | --- | --- |
| Prompt simple, una sola llamada sin criterios | 0.00 | 11 | 0 |
| Modelo 2B con el mismo pipeline | 0.00 | 11 | 0 |
| Siempre «el brazo baja» (sin LLM) | 0.39 | 11 | 20 |
| Siempre «sin problema claro» (sin LLM) | 0.26 | 0 | 0 |
| **Sistema final (v22)** | **0.93** | **1** | **20** |

## 2.5 Resultados de iteraciones

Cada versión de prompt es un archivo que no se edita después de tener resultados, y cada corrida queda registrada (84 corridas). Resultados con las semillas 0 a 3:

| Versión | Cambio | Macro-F1 | Caídas de JC (de 80) | Falsas alarmas (de 44) | Sin respuesta (de 128) |
| --- | --- | --- | --- | --- | --- |
| v12 | pregunta neutral sobre el torso | 0.210 ± 0.052 | 8 | 3 | 6 |
| v17 | el código pone impulso y aterrizaje | 0.336 ± 0.168 | 15 | 2 | 6 |
| v18 | tercer intento si las fases parecen adivinadas | 0.372 ± 0.144 | 16 | 2 | 2 |
| v19 | mayoría de tres respuestas (no adoptada) | 0.319 ± 0.084 | 10 | 2 | 2 |
| v20 | hacia dónde apunta el brazo de lanzamiento | 0.652 ± 0.178 | 58 | 4 | 2 |
| v21 | una diagonal casi horizontal es caída | 0.714 ± 0.233 | 66 | 3 | 2 |
| **v22** | **cuarto intento** | **0.842 ± 0.084** | **67** | **3** | **0** |

Una versión se adopta solo si sube la media de macro-F1, no empeora la mitad de prueba, da como máximo 2 falsas alarmas de 44, tarda como máximo 45 s y pasa las pruebas. v19 no se adoptó. v20 rompió el límite de falsas alarmas (4 de 44), pero el equipo la adoptó por su recall. v21 y v22 mejoraron en todas las medidas.

**Fallas y cómo se abordaron.**

1. **La redacción decidía la respuesta (v5 a v7).** El modelo seguía el orden de las opciones, luego nunca dijo «no» (0 de 114) y luego marcó a 11 de 11 profesionales. Solución en v8: dos preguntas opuestas que deben coincidir.
2. **Fases adivinadas.** El modelo devolvía seis cuadros seguidos. Solución: recorte al jugador (v8) y nuevos intentos con menos cuadros (v18, v22). Los videos sin respuesta bajaron de 6 a 0.
3. **Impacto mal ubicado.** Solución en v9: una segunda mirada a 12 cuadros cercanos; bien en 16 de 18 videos (antes 12).
4. **Fallas inventadas (alucinación).** La observación mencionaba problemas no confirmados. Solución en v16: hablar solo del punto a mejorar; bajaron de 4 a 1 de 7.
5. **El brazo equivocado (v10 a v20).** Es la falla de la figura.

![JC, 0.1 s antes del impacto: el brazo de lanzamiento (a la izquierda) ya bajó al hombro; la mano de la raqueta sigue arriba.](images/informe/falla.jpg){width=40%}

Con v18 el modelo describía el brazo como «extendido hacia afuera», pero respondía «arriba de la cabeza» en 61 de 82 respuestas: miraba la mano de la raqueta. v20 le pide ubicar primero la raqueta y decir hacia dónde apunta el otro brazo: la caída se vio en 58 de 80 saques (antes 16). Lo que v22 todavía falla: sus 3 falsas alarmas vienen de un mismo saque, y no detecta 13 de las 80 caídas.

## 2.6 Producto final y experiencia de usuario

Serve Lab es una app web local (React) servida por una API en Python en la misma Mac. El jugador sube un saque o elige uno de la demo y pulsa «Analizar saque»; el progreso se muestra en cuatro pasos. La app muestra la fase a mejorar con la observación y el ejercicio, los cuadros de las seis fases con su veredicto, el video en cámara lenta y lo que el modelo vio en cada chequeo. «Terminar sesión» resume el progreso; «Empezar de nuevo» limpia todo para el siguiente jugador.

![La app después de analizar un saque de JC.](images/app/result.jpg)

- **Incertidumbre:** cada fase se marca como se ve bien (verde), a mejorar (rojo) o no se ve claro (gris), también en palabras. La app dice cuántos puntos pudo revisar («Pudimos revisar 5 de 6 puntos») y la confianza. Si no confirma nada, responde «sin problema claro».
- **Errores:** si el video no sirve, explica el motivo y cómo grabar de nuevo. Si el modelo no da una respuesta válida, lo dice y pide intentar otra vez.
- **Espera:** un análisis tarda de 15 a 45 s. En modo demo, las respuestas de los videos de la demo se guardan y se muestran al instante.

## 2.7 Limitaciones, riesgos y consideraciones éticas

**Limitaciones técnicas.**

- Alucinaciones: el modelo describe una cosa y responde otra. El esquema JSON impide inventar opciones, pero no juzgar mal un cuadro.
- El 4B no distingue bien ángulos finos, como cuánto se dobla un codo.
- La respuesta cambia con la semilla (± 0.084 en v22).
- La fecha de corte no afecta: los criterios vienen de archivos del proyecto, no del conocimiento del modelo.
- Solo corre en una Mac con Apple Silicon, y tarda 44 s por video.

**Limitaciones de la evaluación.** Las referencias son de un solo juez (Claude) y faltan las del equipo. Las dos mitades se vieron al elegir versiones, así que los puntajes son optimistas. La tarea es casi binaria: todo JC tiene la misma falla y ningún profesional tiene falla, y el modelo podría distinguirlos por otras cosas. Hay un solo amateur, y todos los videos son de lado.

**Riesgos y medidas.**

- Sesgo: el sistema se ajustó a la falla de JC; la regla de adopción limita las falsas alarmas.
- Privacidad: modelo local, la app solo escucha en la propia computadora, los videos no se suben al repositorio y solo se publican imágenes de JC, con su consentimiento.
- Prompt injection: el usuario no escribe texto que llegue a los prompts, y los intentos que vuelven de la página al resumen se validan contra las listas conocidas.
- No debe usarse para evaluar lesiones, para otros golpes, con videos de frente o de espaldas, ni como juicio de nivel profesional.

## 2.8 Conclusiones

El sistema final (v22) saca macro-F1 0.842 ± 0.084 en 4 semillas (0.879 en la mitad de prueba). Detecta la caída del brazo de JC en 67 de 80 saques, da 3 falsas alarmas en 44 saques profesionales y no deja ningún video sin respuesta. Separa el saque de JC de los profesionales, pero el conjunto es pequeño y casi binario: el método funciona en estos videos, no está probado con cualquier jugador.

**Aprendizajes.**

- La redacción del prompt decide la respuesta, y un modelo más grande no lo corrige.
- El código debe hacer lo que el modelo hace mal; el modelo se queda con mirar.
- Leer lo que el modelo describe reveló el cambio con más efecto (v20).
- Una semilla no basta: v17 saca de 0.21 a 0.57 según la semilla.

**Trabajo futuro.** Validar las referencias con el equipo, grabar más jugadores y ángulos, separar una mitad de prueba que no se mire, ajuste fino (LoRA) con cuadros etiquetados y estimación de pose en el Proyecto Final.

## 2.9 Enlace al repositorio

https://github.com/erneram/tennis-ai/tree/proyecto-3

El repositorio contiene el código, los prompts versionados, el conjunto de evaluación, los scripts de evaluación y el registro de iteraciones. El README explica cómo instalar, ejecutar y reproducir las métricas. No usa llaves de API.
