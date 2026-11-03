// Mock backend for VITE_MOCK=1: fixtures plus a fake 4-step job of ~4 s.
import type { AttemptRecord, Config, Job, Phase, Result, Source, Summary, Verdict } from "@/api"

const PHASES = [
  ["stance", "1. Posición inicial"],
  ["toss", "2. Lanzamiento"],
  ["trophy", "3. Posición de trofeo"],
  ["leg_drive", "4. Impulso de piernas"],
  ["contact", "5. Impacto"],
  ["follow_through", "6. Terminación"],
] as const

export const config: Config = {
  model: "Qwen3-VL-4B",
  version: "v18",
  phases: PHASES.map(([id, label]) => ({ id, label })),
  demo: [
    { id: "jc-s15.mp4", label: "Amateur · jc-s15", kind: "amateur", video_url: "/api/demo/jc-s15.mp4" },
    { id: "jc-s4.MOV", label: "Amateur · jc-s4", kind: "amateur", video_url: "/api/demo/jc-s4.MOV" },
    { id: "pro02_s01.mp4", label: "Pro · pro02_s01", kind: "pro", video_url: "/api/demo/pro02_s01.mp4" },
    { id: "pro03_s07.mp4", label: "Pro · pro03_s07", kind: "pro", video_url: "/api/demo/pro03_s07.mp4" },
  ],
  tips: [
    "Graba de lado, a la altura de la cintura y a unos 4 a 6 metros.",
    "Que se vea el cuerpo completo y la pelota durante todo el saque.",
    "Deja el teléfono quieto: apoyado o en un trípode, sin zoom.",
    "Un solo saque por video, de 3 a 10 segundos, con buena luz.",
  ],
}

// Arm angles (degrees) per phase so each placeholder frame looks different.
const POSE: Record<string, [number, number, number]> = {
  stance: [200, 160, 0.78],
  toss: [100, 200, 0.3],
  trophy: [80, 230, 0.12],
  leg_drive: [150, 250, 0.16],
  contact: [180, 80, 0.12],
  follow_through: [250, 300, 0.7],
}

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
<path d="M60 52L${arm(left)}M60 52L${arm(right)}"/></g>
<circle cx="${id === "toss" ? 52 : 46}" cy="${(160 * ball).toFixed(0)}" r="4.5" fill="#c8d84a"/></svg>`
  return `data:image/svg+xml,${encodeURIComponent(svg)}`
}

function phases(verdicts: (Verdict | null)[], evidence: string | null, tint: string): Phase[] {
  return PHASES.map(([id, label], i) => ({
    id,
    label,
    verdict: verdicts[i],
    evidence: id === evidence,
    image_url: frame(id, tint),
  }))
}

const trophyResult = (tint: string): Result => ({
  kind: "feedback",
  seconds: 31.4,
  cached: true,
  feedback: {
    phase: "trophy",
    phase_label: "3. Posición de trofeo",
    category: "toss_arm",
    observation:
      "Tu brazo de lanzamiento baja demasiado pronto y deja de apuntar a la pelota mientras la raqueta baja por detrás. Esto afecta la precisión del servicio porque el punto de impacto se desplaza. La posición del brazo de lanzamiento debe mantenerse firme hasta el momento del golpe.",
    seen_in: "leg_drive",
    seen_in_label: "4. Impulso de piernas",
    drill: {
      name: "Brazo arriba hasta el golpe",
      instructions:
        "Lanza y deja el brazo de lanzamiento apuntando a la pelota hasta que la raqueta empiece a subir. Haz 3 series de 8 saques suaves concentrándote solo en esa pausa.",
    },
    confidence: "medium",
    confidence_label: "media",
    certainty: "Pudimos revisar 5 de 6 puntos. No se ve con claridad: 6. Terminación.",
    checked: 5,
    total: 6,
  },
  phases: phases([null, null, "yes", "no", "yes", "unclear"], "leg_drive", tint),
  checks: [
    {
      category: "toss_arm",
      phase: "trophy",
      phase_label: "3. Posición de trofeo",
      part: "Brazo de lanzamiento",
      state: "recto",
      answer: "yes",
      seen: "El brazo izquierdo está extendido hacia arriba apuntando a la pelota.",
    },
    {
      category: "toss_arm",
      phase: "leg_drive",
      phase_label: "4. Impulso de piernas",
      part: "Brazo de lanzamiento con la raqueta atrás",
      state: "ya bajó",
      answer: "no",
      seen: "El brazo izquierdo ya está a la altura del pecho mientras la raqueta baja por la espalda.",
    },
    {
      category: "legs",
      phase: "leg_drive",
      phase_label: "4. Impulso de piernas",
      part: "Rodillas",
      state: "flexionadas",
      answer: "yes",
      seen: "Las rodillas están dobladas antes de empujar hacia arriba.",
    },
    {
      category: "contact",
      phase: "contact",
      phase_label: "5. Impacto",
      part: "Brazo de la raqueta en el impacto",
      state: "casi recto",
      answer: "yes",
      seen: "El brazo de la raqueta está casi completamente extendido al golpear.",
    },
    {
      category: "contact",
      phase: "contact",
      phase_label: "5. Impacto",
      part: "Altura del impacto",
      state: "sobre la cabeza",
      answer: "yes",
      seen: "La pelota se golpea por encima de la cabeza, delante del cuerpo.",
    },
    {
      category: "balance",
      phase: "follow_through",
      phase_label: "6. Terminación",
      part: "Equilibrio al terminar",
      state: "no se ve claro",
      answer: "unclear",
      seen: "El jugador sale del cuadro antes de terminar el movimiento.",
    },
  ],
  slow_url: "/api/slow/mock.mp4",
  record: { time: new Date().toISOString(), phase: "trophy", category: "toss_arm", drill_id: "toss_arm_hold" },
})

const proResult: Result = {
  kind: "feedback",
  seconds: 28.9,
  cached: true,
  feedback: {
    phase: null,
    phase_label: null,
    category: "none",
    observation:
      "Las seis fases del saque se ven sólidas: el brazo de lanzamiento se mantiene arriba, las piernas empujan antes del golpe y el impacto ocurre con el brazo extendido sobre la cabeza.",
    seen_in: null,
    seen_in_label: null,
    drill: {
      name: "Mantén la rutina",
      instructions: "Repite 2 series de 10 saques a ritmo de partido buscando el mismo lanzamiento cada vez.",
    },
    confidence: "high",
    confidence_label: "alta",
    certainty: "Pudimos revisar 6 de 6 puntos.",
    checked: 6,
    total: 6,
  },
  phases: phases(["yes", "yes", "yes", "yes", "yes", "yes"], null, "#e6f1f4"),
  checks: [],
  slow_url: "/api/slow/mock-pro.mp4",
  record: { time: new Date().toISOString(), phase: "none", category: "none", drill_id: "keep_routine" },
}

const retakeResult: Result = {
  kind: "retake",
  title: "Necesitamos una nueva grabación",
  message:
    "No se ve el cuerpo completo durante el saque: la cámara está demasiado cerca y corta los pies y la raqueta.",
}

const MESSAGES = [
  "En cola…",
  "Paso 1 de 4: revisando el video…",
  "Paso 2 de 4: buscando las fases del saque…",
  "Paso 3 de 4: revisando cada fase…",
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
      : { ...trophyResult(uploads % 2 ? "#eef5f7" : "#e6f1f4"), cached: "demo" in source }
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
  return {
    text: `En ${attempts.length} ${attempts.length === 1 ? "intento" : "intentos"} el punto que más se repitió fue la posición de trofeo: el brazo de lanzamiento baja antes de tiempo.`,
    focus_label: "3. Posición de trofeo",
    next_focus: "Brazo arriba hasta el golpe: 3 series de 8 saques suaves.",
    saved: true,
  }
}
