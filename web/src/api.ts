// Typed client for the Serve Lab API (same origin, under /api).
// VITE_MOCK=1 swaps every call for fixtures in src/fixtures with a fake ~4 s job.

export interface Config {
  model: string
  version: string
  demo: { id: string; label: string; kind: "amateur" | "pro"; video_url: string }[]
  tips: string[]
}

export interface Feedback {
  drops_early: boolean
  title: string
  phase_label: string
  observation: string
  reason: string
  drill: { name: string; instructions: string }
  probability: number
  threshold: number
  seeds: number[]
  confidence: "high" | "medium" | "low"
  confidence_label: "alta" | "media" | "baja"
  certainty: string
}

/** A key frame around contact, with the skeleton drawn. */
export interface Moment {
  id: string
  label: string
  evidence: boolean
  image_url: string
}

/** Fall in the probability when one arm joint moves like a clean serve during one stretch. */
export interface Occlusion {
  parts: string[]
  starts: number[]
  values: number[][]
}

/** The toss height (torso lengths above the shoulder) against the professionals'. */
export interface TossProgress {
  height: number
  pros_median: number
  pros_floor: number
  in_range: boolean
  text: string
}

/** A pose measure that separates amateurs from professionals, next to the professionals'. */
export interface BodyMeasure {
  aspect: string
  label: string
  mine: string
  pros: string
  in_range: boolean
}

/** One point to improve, with how far it can be trusted and its drill. */
export interface ImprovementPoint {
  aspect: string
  label: string
  level: "validated" | "trial" | "measure"
  level_label: string
  seen: string
  evidence: string
  drill: { name: string; instructions: string }
}

export interface AttemptRecord {
  time: string
  category: string
  probability: number
  drill_id: string
}

export interface Result {
  kind: "feedback" | "retake" | "error"
  title?: string
  message?: string
  seconds?: number
  cached?: boolean
  feedback?: Feedback
  moments?: Moment[]
  occlusion?: Occlusion
  points?: ImprovementPoint[]
  progress?: TossProgress
  body?: BodyMeasure[]
  slow_url?: string
  record?: AttemptRecord
}

export interface Job {
  status: "queued" | "running" | "done"
  step: number
  progress: number
  message: string
  result?: Result
}

export interface Summary {
  text: string
  focus_label: string
  next_focus: string
  saved: boolean
}

export type Source = { file: File } | { demo: string }

export const MOCK = import.meta.env.VITE_MOCK === "1"

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<T>
}

export async function getConfig(): Promise<Config> {
  if (MOCK) return (await import("./fixtures")).config
  return json(await fetch("/api/config"))
}

export async function startAnalysis(source: Source): Promise<string> {
  if (MOCK) return (await import("./fixtures")).mockStart(source)
  const init: RequestInit =
    "file" in source
      ? { method: "POST", body: formData(source.file) }
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ demo: source.demo }),
        }
  const { job } = await json<{ job: string }>(await fetch("/api/analyze", init))
  return job
}

function formData(file: File) {
  const data = new FormData()
  data.append("file", file)
  return data
}

export async function getJob(job: string): Promise<Job> {
  if (MOCK) return (await import("./fixtures")).mockJob(job)
  return json(await fetch(`/api/jobs/${encodeURIComponent(job)}`))
}

/** Polls every 500 ms until the job is done; reports each update. */
export async function waitForJob(job: string, onUpdate: (job: Job) => void, signal: AbortSignal): Promise<Result> {
  for (;;) {
    const state = await getJob(job)
    if (signal.aborted) throw new DOMException("aborted", "AbortError")
    onUpdate(state)
    if (state.status === "done") {
      return state.result ?? { kind: "error", title: "No pudimos analizar el video" }
    }
    await new Promise((resolve) => setTimeout(resolve, 500))
  }
}

export async function getSummary(player: string, attempts: AttemptRecord[]): Promise<Summary> {
  if (MOCK) return (await import("./fixtures")).mockSummary(attempts)
  const res = await fetch("/api/summary", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ player, attempts }),
  })
  const body = await json<Summary | { error: string }>(res)
  if ("error" in body) throw new Error(body.error)
  return body
}
