// Mock backend for VITE_MOCK=1: fixtures plus a fake 4-step job of ~4 s.
import type {
  AttemptRecord,
  Compared,
  Config,
  ImprovementPoint,
  Job,
  Moment,
  Occlusion,
  OtherPart,
  Result,
  Source,
  Summary,
  TossModel,
} from "@/api"

export const config: Config = {
  model: "MediaPipe + M1C",
  version: "modelo propio",
  demo: [
    { id: "jc-s15.mp4", label: "Amateur · jc-s15", kind: "amateur", video_url: "/api/demo/jc-s15.mp4" },
    { id: "jc-s4.mp4", label: "Amateur · jc-s4", kind: "amateur", video_url: "/api/demo/jc-s4.mp4" },
    { id: "pro02_s01.mp4", label: "Profesional · pro02_s01", kind: "pro", video_url: "/api/demo/pro02_s01.mp4" },
    { id: "pro03_s07.mp4", label: "Profesional · pro03_s07", kind: "pro", video_url: "/api/demo/pro03_s07.mp4" },
  ],
  tips: [
    "Teléfono horizontal, a la altura del pecho y recto, a unos 6 m, lente 1x sin zoom.",
    "De lado: la cámara mira tu pecho, sobre la prolongación de la línea de fondo.",
    "Todo el cuerpo y la raqueta arriba dentro del cuadro, sin el sol detrás de ti.",
    "Un solo saque por video, de 4 a 8 segundos, a 1080p y 60 fps.",
  ],
}

// Arm angles (degrees) per moment so each placeholder frame looks different.
const POSE: Record<string, [number, number, number]> = {
  bend: [200, 160, 0.78],
  arm_up: [100, 200, 0.3],
  arm_down: [150, 250, 0.16],
  contact: [180, 80, 0.12],
  landing: [230, 300, 0.9],
}
const MOMENTS = [
  ["bend", "Piernas flexionadas", "las piernas", "legs"],
  ["arm_up", "Brazo que lanza arriba", "el brazo que lanza", ""],
  ["arm_down", "La muñeca baja del hombro", "el brazo que lanza", "toss"],
  ["contact", "Impacto", "el brazo de la raqueta", "arm_at_contact"],
  ["landing", "Aterrizaje", "las piernas", "landing"],
] as const

function frame(id: string, tint: string): string {
  const [left, right, ball] = POSE[id]
  const arm = (deg: number) => {
    const r = (deg * Math.PI) / 180
    return `${(60 + 34 * Math.cos(r)).toFixed(1)},${(52 - 34 * Math.sin(r)).toFixed(1)}`
  }
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 160">
<rect width="120" height="160" fill="${tint}"/><path d="M0 132h120" stroke="#9fb7c2" stroke-width="2"/>
<g stroke="#29404d" stroke-width="5" stroke-linecap="round" fill="none">
<circle cx="60" cy="34" r="9" fill="#29404d"/><path d="M60 46v42M60 88l-14 42M60 88l14 42"/>
<path d="M60 52L${arm(right)}"/><path d="M60 52L${arm(left)}" stroke="#ff8c00"/></g>
<circle cx="46" cy="${(160 * ball).toFixed(0)}" r="4.5" fill="#c8d84a"/></svg>`
  return `data:image/svg+xml,${encodeURIComponent(svg)}`
}

/** The key moments; those of the aspects in `said` show a point to improve. */
function moments(said: string[], tint: string): Moment[] {
  return MOMENTS.filter(([id]) => id !== "arm_down" || said.includes("toss")).map(([id, label, part, aspect]) => ({
    id,
    label,
    note: `En naranja: ${part}.`,
    evidence: said.includes(aspect),
    image_url: frame(id, tint),
  }))
}

const PARTS = ["Hombro (lanza)", "Codo (lanza)", "Muñeca (lanza)", "Hombro (raqueta)", "Codo (raqueta)", "Muñeca (raqueta)"]
const STARTS = [-1.6, -1.33, -1.07, -0.8, -0.53, -0.27, 0, 0.27]

function occlusion(sign: number): Occlusion {
  const values = PARTS.map((_, j) =>
    STARTS.map((s, k) => sign * (j === 2 ? 0.06 : j === 1 ? 0.02 : 0.004) * Math.exp(-((k - 4) ** 2) / 3) + (s > 0 ? 0.002 : 0)),
  )
  return { parts: PARTS, starts: STARTS, values }
}

const compare = (height: number, fall: number, reach: number): Compared[] => [
  {
    aspect: "toss_height",
    label: "Altura de la mano que lanza",
    caption: "Qué tan arriba sigue tu mano de lanzamiento cuando la raqueta está atrás.",
    value: height,
    pros_median: 1.05,
    pros_floor: 0.73,
    in_range: height >= 0.73,
  },
  {
    aspect: "racket_drop",
    label: "Caída de la raqueta",
    caption: "Cuánto cae tu raqueta detrás de la espalda antes de subir a pegarle.",
    value: fall,
    pros_median: 0.21,
    pros_floor: 0.08,
    in_range: fall >= 0.08,
  },
  {
    aspect: "arm_at_contact",
    label: "Brazo al impacto",
    caption: "Qué tan arriba le pegas a la pelota.",
    value: reach,
    pros_median: 1.02,
    pros_floor: 0.89,
    in_range: reach >= 0.89,
  },
]

const points: ImprovementPoint[] = [
  {
    aspect: "arm_at_contact",
    label: "Brazo al impacto",
    level: "measure",
    level_label: "Medida frente a profesionales",
    precision: 0.983,
    sure: "high",
    sure_label: "Muy probable",
    seen: "Le pegas a la pelota más abajo de lo que alcanzas: al pegarle, tu brazo no llega estirado hacia arriba.",
    goal: "Lo ideal: el brazo estirado, casi vertical, pegándole lo más alto que alcanzas.",
    trust:
      "Cuando la app marca esto, casi siempre tiene razón. Sale de comparar tu video con saques de profesionales; no es un diagnóstico.",
    detail:
      "Medida frente a profesionales. Al impacto, la muñeca de la raqueta llega 0.57 torsos sobre su hombro. En los profesionales, la mediana es 1.02 y 9 de cada 10 llegan a 0.89 o más. Las distancias van en largos de torso. Es una medida de la pose, sin entrenar: quedas fuera del rango de 9 de cada 10 profesionales. De los saques amateur fuera de ese rango, 98% tenían este problema según quien los etiquetó. No es un diagnóstico.",
    drill: {
      name: "Alcanzar el punto más alto",
      instructions:
        "Pide a un compañero que sostenga una pelota en alto, por encima de tu alcance con el brazo extendido. Haz 10 golpes de sombra tocándola con la raqueta y el brazo totalmente estirado.",
    },
  },
  {
    aspect: "landing",
    label: "Aterrizaje",
    level: "trial",
    level_label: "Modelo en prueba",
    precision: 0.732,
    sure: "medium",
    sure_label: "Probable",
    seen: "Después de pegarle avanzas poco: caes casi en el mismo lugar.",
    goal: "Lo ideal: caer hacia adelante, dentro de la cancha, sobre el pie de adelante y en equilibrio.",
    trust: "Cuando la app marca esto, suele tener razón, pero no siempre. Todavía lo estamos probando con más jugadores.",
    detail:
      "Modelo en prueba. Después del impacto, la cadera avanza 0.10 torsos. En los profesionales, la mediana es 0.46 y 9 de cada 10 llegan a 0.26 o más. Las distancias van en largos de torso. 5 de 5 versiones del modelo lo marcan. En validación acertó 73% de las veces que lo marcó y lo marcó en 7% de los saques profesionales. Está en prueba: falta probarlo con más jugadores nuevos.",
    drill: {
      name: "Aterriza y sostén",
      instructions:
        "Después de cada saque, aterriza sobre el pie adelantado dentro de la cancha y mantén el equilibrio 2 segundos. 3 series de 8.",
    },
  },
  {
    aspect: "toss",
    label: "Brazo que lanza",
    level: "value",
    level_label: "Modelo elegido por valor",
    precision: 0.2,
    sure: "low",
    sure_label: "En duda",
    seen: "Bajas muy pronto el brazo con el que lanzas la pelota: cuando la raqueta va a subir a pegarle, esa mano ya está a la altura del hombro o más abajo.",
    goal: "Lo ideal: que siga arriba, apuntando a la pelota, hasta que la raqueta empiece a subir.",
    trust: "Cuando la app marca esto, se equivoca más veces de las que acierta. Tómalo como una sugerencia, no como un hecho.",
    detail:
      "Modelo elegido por valor. Tu muñeca de lanzamiento baja del hombro 0.37 s antes del impacto. 5 de 5 versiones del modelo coinciden. En validación acertó 20% de las veces que lo marcó y lo marcó en 0% de los saques profesionales. En los videos de prueba encontró el problema en 94% de los saques que lo tenían y lo marcó de más en 56% de los que no. El modelo se eligió por su valor; con las reglas de los demás puntos no tendría nivel.",
    drill: {
      name: "Brazo arriba hasta el golpe",
      instructions:
        "Lanza y deja el brazo de lanzamiento apuntando a la pelota hasta que la raqueta empiece a subir. Haz 3 series de 8 saques suaves concentrándote solo en esa pausa.",
    },
  },
]

const others = (toss: boolean, fall: number): OtherPart[] => [
  ...(toss
    ? [
        {
          aspect: "toss",
          label: "Brazo que lanza",
          status: "fine" as const,
          status_label: "Sin problema",
          seen: "Mantienes arriba el brazo con el que lanzas la pelota hasta que la raqueta sube a pegarle.",
        },
      ]
    : []),
  {
    aspect: "legs",
    label: "Piernas",
    status: "fine",
    status_label: "Sin problema",
    seen: "Doblas las rodillas antes de subir a pegarle.",
  },
  fall >= 0.08
    ? {
        aspect: "racket_drop",
        label: "Caída de la raqueta",
        status: "inside",
        status_label: "Parecido a los profesionales",
        seen: "Antes de subir a pegarle, tu raqueta cae detrás de la espalda.",
        note: "Es solo una observación.",
      }
    : {
        aspect: "racket_drop",
        label: "Caída de la raqueta",
        status: "outside",
        status_label: "Distinto a los profesionales",
        seen: "Antes de subir a pegarle, tu raqueta casi no cae detrás de la espalda.",
        goal: "Lo ideal: que caiga detrás de la espalda, como si te rascaras, y luego suba hacia la pelota.",
        note: "Es solo una observación, sin ejercicio. Cuando la app marca esto, se equivoca seguido.",
      },
]

const UNCHECKED = ["los hombros", "dónde está la raqueta cuando la pelota llega arriba"]

const toss = (drops: boolean): TossModel => ({
  drops_early: drops,
  reason: drops
    ? "Lo que más pesa hacia «baja antes de tiempo»: muñeca (lanza) entre 0.53 y 0.27 s antes del impacto."
    : "Lo que más se parece a un saque sin el problema: muñeca (lanza) entre 1.33 y 1.07 s antes del impacto.",
  probability: drops ? 0.64 : 0.06,
  threshold: 0.198,
  seeds: drops ? [0.58, 0.61, 0.64, 0.66, 0.71] : [0.03, 0.05, 0.06, 0.07, 0.09],
  confidence: drops ? "high" : "medium",
  confidence_label: drops ? "alta" : "media",
  certainty:
    "5 de 5 versiones del modelo coinciden y MediaPipe vio el 94% de los puntos de tus brazos. En los videos de prueba, el modelo encontró el problema en 94% de los saques que lo tenían y recomendó el ejercicio de más en 56% de los que no.",
  occlusion: occlusion(drops ? 1 : -1),
})

const dropResult = (tint: string): Result => ({
  kind: "feedback",
  seconds: 4.8,
  cached: true,
  title: "A mejorar: brazo al impacto, aterrizaje y brazo que lanza",
  points,
  others: others(false, 0.02),
  unchecked: UNCHECKED,
  drill: points[0].drill,
  moments: moments(["arm_at_contact", "landing", "toss"], tint),
  compare: compare(0.38, 0.02, 0.57),
  toss: toss(true),
  slow_url: "/api/slow/mock.mp4",
  record: {
    time: "10:00",
    category: "toss_arm_drops_early",
    probability: 0.64,
    drill_id: "high_reach",
    points: ["arm_at_contact", "landing", "toss"],
  },
})

const proResult: Result = {
  kind: "feedback",
  seconds: 3.1,
  cached: true,
  title: "No vemos un problema claro",
  points: [],
  others: [
    ...others(true, 0.21),
    {
      aspect: "arm_at_contact",
      label: "Brazo al impacto",
      status: "fine",
      status_label: "Sin problema",
      seen: "Le pegas a la pelota arriba, con el brazo estirado.",
    },
    {
      aspect: "landing",
      label: "Aterrizaje",
      status: "fine",
      status_label: "Sin problema",
      seen: "Después de pegarle caes hacia adelante, dentro de la cancha.",
    },
  ],
  unchecked: UNCHECKED,
  drill: {
    name: "Repetir el mismo saque",
    instructions:
      "Haz 3 series de 10 saques al 70 % de velocidad buscando repetir siempre el mismo movimiento. Al terminar, graba otro saque de lado para revisarlo de nuevo.",
  },
  moments: moments([], "#e6f1f4"),
  compare: compare(1.05, 0.21, 1.02),
  toss: toss(false),
  slow_url: "/api/slow/mock-pro.mp4",
  record: { time: "10:05", category: "no_clear_problem", probability: 0.06, drill_id: "steady_serves", points: [] },
}

const retakeResult: Result = {
  kind: "retake",
  title: "Necesitamos una nueva grabación",
  message:
    "No vemos el lanzamiento de la pelota: graba el saque completo, con el brazo que lanza y la raqueta arriba dentro del cuadro.",
}

const MESSAGES = [
  "En cola…",
  "Paso 1 de 4: revisando el video…",
  "Paso 2 de 4: buscando tu cuerpo en cada cuadro…",
  "Paso 3 de 4: midiendo tu saque…",
  "Paso 4 de 4: preparando la cámara lenta…",
]

const jobs = new Map<string, { start: number; result: Result }>()
let uploads = 0

export function mockStart(source: Source): string {
  const id = "demo" in source ? source.demo : `upload-${++uploads}`
  const result = id.startsWith("jc-s4")
    ? retakeResult
    : id.startsWith("pro")
      ? proResult
      : { ...dropResult(uploads % 2 ? "#eef5f7" : "#e6f1f4"), cached: "demo" in source }
  const job = `job-${jobs.size + 1}`
  jobs.set(job, { start: Date.now(), result })
  return job
}

export function mockJob(id: string): Job {
  const job = jobs.get(id)
  if (!job)
    return {
      status: "done",
      step: 4,
      progress: 1,
      message: "",
      result: { kind: "error", title: "No pudimos analizar el video" },
    }
  const elapsed = Date.now() - job.start
  if (elapsed >= 4000) return { status: "done", step: 4, progress: 1, message: "Listo", result: job.result }
  const step = Math.min(4, Math.floor(elapsed / 1000) + 1)
  return { status: "running", step, progress: elapsed / 4000, message: MESSAGES[step] }
}

export function mockSummary(attempts: AttemptRecord[]): Summary {
  const said = attempts.filter((a) => a.points.length).length
  return {
    text: said
      ? `En ${attempts.length} intento(s) salió: brazo al impacto en ${said}, aterrizaje en ${said}, brazo que lanza en ${said}.`
      : `En ${attempts.length} intento(s) no vimos un problema claro.`,
    focus_label: "Alcanzar el punto más alto",
    next_focus:
      "Pide a un compañero que sostenga una pelota en alto, por encima de tu alcance con el brazo extendido. Haz 10 golpes de sombra tocándola con la raqueta y el brazo totalmente estirado.",
    saved: true,
  }
}
