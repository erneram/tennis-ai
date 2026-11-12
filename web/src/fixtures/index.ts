// Mock backend for VITE_MOCK=1: fixtures plus a fake 4-step job of ~4 s.
import type {
  AttemptRecord,
  BodyMeasure,
  Config,
  Job,
  Moment,
  Occlusion,
  Result,
  Source,
  Summary,
  TossProgress,
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
  start: [200, 160, 0.78],
  arm_up: [100, 200, 0.3],
  arm_down: [150, 250, 0.16],
  contact: [180, 80, 0.12],
}
const MOMENTS = [
  ["start", "Inicio"],
  ["arm_up", "Brazo más arriba"],
  ["arm_down", "La muñeca baja del hombro"],
  ["contact", "Impacto"],
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

function moments(evidence: boolean, tint: string): Moment[] {
  return MOMENTS.map(([id, label]) => ({
    id,
    label,
    evidence: evidence && id === "arm_down",
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

const progress = (height: number): TossProgress => ({
  height,
  pros_median: 1.05,
  pros_floor: 0.73,
  in_range: height >= 0.73,
  text: `Cuando la raqueta está más baja, tu mano de lanzamiento queda ${height.toFixed(2)} torsos sobre el hombro. En los profesionales, la mediana es 1.05 y 9 de cada 10 llegan a 0.73 o más.`,
})

const body = (peak: number, reach: number): BodyMeasure[] => [
  {
    aspect: "racket_at_peak",
    label: "Raqueta en el pico",
    mine: `De la raqueta más baja al impacto pasan ${peak.toFixed(2)} s.`,
    pros: "Profesionales (mediana): de la raqueta más baja al impacto pasan 0.20 s.",
  },
  {
    aspect: "arm_at_contact",
    label: "Brazo al impacto",
    mine: `Al impacto, la muñeca de la raqueta llega ${reach.toFixed(2)} torsos sobre su hombro.`,
    pros: "Profesionales (mediana): al impacto, la muñeca de la raqueta llega 1.02 torsos sobre su hombro.",
  },
]

const dropResult = (tint: string): Result => ({
  kind: "feedback",
  seconds: 4.8,
  cached: true,
  feedback: {
    drops_early: true,
    title: "El brazo que lanza baja antes de tiempo",
    phase_label: "3. Posición de trofeo",
    observation: "Tu muñeca de lanzamiento baja del hombro 0.37 s antes del impacto.",
    reason: "Lo que más pesa hacia «baja antes de tiempo»: muñeca (lanza) entre 0.53 y 0.27 s antes del impacto.",
    drill: {
      name: "Brazo arriba hasta el golpe",
      instructions:
        "Lanza y deja el brazo de lanzamiento apuntando a la pelota hasta que la raqueta empiece a subir. Haz 3 series de 8 saques suaves concentrándote solo en esa pausa.",
    },
    probability: 0.64,
    threshold: 0.198,
    seeds: [0.58, 0.61, 0.64, 0.66, 0.71],
    confidence: "high",
    confidence_label: "alta",
    certainty:
      "5 de 5 versiones del modelo coinciden y MediaPipe vio el 94% de los puntos de tus brazos. En los videos de prueba, el modelo encontró el problema en 100% de los saques que lo tenían y recomendó el ejercicio de más en 64% de los que no.",
  },
  moments: moments(true, tint),
  occlusion: occlusion(1),
  progress: progress(0.38),
  body: body(0.78, 0.57),
  slow_url: "/api/slow/mock.mp4",
  record: { time: "10:00", category: "toss_arm_drops_early", probability: 0.64, drill_id: "toss_hold" },
})

const proResult: Result = {
  kind: "feedback",
  seconds: 3.1,
  cached: true,
  feedback: {
    drops_early: false,
    title: "No vemos que el brazo que lanza baje antes de tiempo",
    phase_label: "3. Posición de trofeo",
    observation: "Tu muñeca de lanzamiento baja del hombro 0.10 s antes del impacto.",
    reason: "Lo que más se parece a un saque sin el problema: muñeca (lanza) entre 1.33 y 1.07 s antes del impacto.",
    drill: {
      name: "Repetir el mismo saque",
      instructions:
        "Haz 3 series de 10 saques al 70 % de velocidad buscando repetir siempre el mismo movimiento. Al terminar, graba otro saque de lado para revisarlo de nuevo.",
    },
    probability: 0.06,
    threshold: 0.198,
    seeds: [0.03, 0.05, 0.06, 0.07, 0.09],
    confidence: "medium",
    confidence_label: "media",
    certainty:
      "5 de 5 versiones del modelo coinciden y MediaPipe vio el 98% de los puntos de tus brazos. En los videos de prueba, el modelo encontró el problema en 100% de los saques que lo tenían y recomendó el ejercicio de más en 64% de los que no.",
  },
  moments: moments(false, "#e6f1f4"),
  occlusion: occlusion(-1),
  progress: progress(1.05),
  body: body(0.2, 1.02),
  slow_url: "/api/slow/mock-pro.mp4",
  record: { time: "10:05", category: "no_clear_problem", probability: 0.06, drill_id: "steady_serves" },
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
  "Paso 3 de 4: midiendo el brazo que lanza…",
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
  const drops = attempts.filter((a) => a.category === "toss_arm_drops_early").length
  return {
    text: `En ${attempts.length} intento(s), el brazo que lanza bajó antes de tiempo en ${drops}.`,
    focus_label: "Brazo arriba hasta el golpe",
    next_focus:
      "Lanza y deja el brazo de lanzamiento apuntando a la pelota hasta que la raqueta empiece a subir. Haz 3 series de 8 saques suaves concentrándote solo en esa pausa.",
    saved: true,
  }
}
