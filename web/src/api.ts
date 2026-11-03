// Typed client for the Serve Lab API (same origin, under /api).
// VITE_MOCK=1 swaps every call for fixtures in src/fixtures with a fake ~4 s job.

export type Verdict = "yes" | "no" | "unclear"

export interface Config {
  model: string
  version: string
  phases: { id: string; label: string }[]
  demo: { id: string; label: string; kind: "amateur" | "pro"; video_url: string }[]
  tips: string[]
}

export interface Feedback {
  phase: string | null
  phase_label: string | null
  category: string
  observation: string
  seen_in: string | null
  seen_in_label: string | null
  drill: { name: string; instructions: string }
  confidence: "high" | "medium" | "low"
  confidence_label: "alta" | "media" | "baja"
  certainty: string
  checked: number
  total: number
}

export interface Phase {
  id: string
  label: string
  verdict: Verdict | null
  evidence: boolean
  image_url: string
}

export interface Check {
  category: string
  phase: string
  phase_label: string
  part: string
  state: string
  answer: Verdict
  seen: string
}

export interface AttemptRecord {
  time: string
  phase: string
  category: string
  drill_id: string
}

export interface Result {
  kind: "feedback" | "retake" | "error"
  title?: string
  message?: string
  seconds?: number
  cached?: boolean
  feedback?: Feedback
  phases?: Phase[]
  checks?: Check[]
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
