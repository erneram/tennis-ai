import { useEffect, useRef, useState } from "react"
import { toast } from "sonner"
import {
  MOCK,
  getConfig,
  getSummary,
  startAnalysis,
  waitForJob,
  type Config,
  type Job,
  type Result,
  type Source,
  type Summary,
} from "@/api"
import { InputPanel, SessionPanel, type Chosen } from "@/components/InputPanel"
import { ResultPanel, type Attempt } from "@/components/ResultPanel"
import { UplinkHeader } from "@/components/thegridcn/uplink-header"
import { Toaster } from "@/components/ui/sonner"

export type Run =
  | { status: "idle" }
  | { status: "running"; job: Job | null }
  | { status: "done"; result: Result; attempt: number | null }

const OFFLINE: Result = {
  kind: "error",
  title: "No pudimos analizar el video",
  message: "No hay conexión con el servidor local. Revisa que Serve Lab esté abierto y vuelve a intentarlo.",
}

export default function App() {
  const [config, setConfig] = useState<Config | null>(null)
  const [chosen, setChosen] = useState<Chosen | null>(null)
  const [run, setRun] = useState<Run>({ status: "idle" })
  const [attempts, setAttempts] = useState<Attempt[]>([])
  const [player, setPlayer] = useState("")
  const [summary, setSummary] = useState<Summary | null>(null)
  const [summarizing, setSummarizing] = useState(false)
  const [resetKey, setResetKey] = useState(0)
  const abort = useRef<AbortController | null>(null)
  const count = useRef(0)
  const results = useRef<HTMLDivElement>(null)

  useEffect(() => {
    getConfig()
      .then(setConfig)
      .catch(() => toast.error("No hay conexión con el servidor local."))
  }, [])

  async function analyze(source: Source) {
    abort.current?.abort()
    const controller = new AbortController()
    abort.current = controller
    setRun({ status: "running", job: null })
    if (matchMedia("(max-width: 1023px)").matches) {
      const smooth = !matchMedia("(prefers-reduced-motion: reduce)").matches
      results.current?.scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "start" })
    }
    try {
      const job = await startAnalysis(source)
      const result = await waitForJob(job, (j) => setRun({ status: "running", job: j }), controller.signal)
      if (result.kind === "feedback") {
        const n = ++count.current
        setAttempts((prev) => [...prev, { n, result }])
        setRun({ status: "done", result, attempt: n })
      } else {
        setRun({ status: "done", result, attempt: null })
      }
    } catch (err) {
      if (controller.signal.aborted) return
      setRun({ status: "done", result: OFFLINE, attempt: null })
      console.error(err)
    }
  }

  function sourceOf(c: Chosen): Source {
    return c.kind === "file" ? { file: c.file } : { demo: c.demo.id }
  }

  function choose(next: Chosen | null) {
    if (chosen?.kind === "file") URL.revokeObjectURL(chosen.url)
    setChosen(next)
  }

  function reset() {
    abort.current?.abort()
    choose(null)
    setRun({ status: "idle" })
    setAttempts([])
    count.current = 0
    setSummary(null)
    setPlayer("")
    setResetKey((k) => k + 1)
  }

  async function finish() {
    setSummarizing(true)
    try {
      const records = attempts.flatMap((a) => (a.result.record ? [a.result.record] : []))
      setSummary(await getSummary(player.trim(), records))
    } catch (err) {
      toast.error(err instanceof Error && err.message ? err.message : "No pudimos preparar el resumen.")
    } finally {
      setSummarizing(false)
    }
  }

  // Mock-only: ?auto=jc-s15.mp4,jc-s4.mp4 runs those demos in order (for screenshots).
  const autoRan = useRef(false)
  useEffect(() => {
    const auto = new URLSearchParams(location.search).get("auto")
    if (!MOCK || !config || !auto || autoRan.current) return
    autoRan.current = true
    ;(async () => {
      for (const id of auto.split(",")) {
        const demo = config.demo.find((d) => d.id === id)
        if (!demo) continue
        setChosen({ kind: "demo", demo })
        await analyze({ demo: id })
      }
    })()
  }, [config])

  const running = run.status === "running"
  const current = run.status === "done" && run.attempt ? attempts[run.attempt - 1] : undefined
  const previous = current ? attempts[current.n - 2] : undefined

  return (
    <div className="min-h-dvh">
      <UplinkHeader
        leftText={
          <span className="flex items-center gap-2">
            <span aria-hidden="true" className="size-2 rounded-full bg-success" />
            Local · sin internet
          </span>
        }
        rightText={config ? `${config.model} · ${config.version}` : "Cargando modelo…"}
        className="bg-card"
      />

      <header className="mx-auto flex max-w-[1440px] flex-wrap items-end justify-between gap-x-6 gap-y-1 px-4 pt-6 pb-6 sm:px-6 lg:px-10 lg:pt-8">
        <div>
          <h1 className="font-display font-bold text-3xl tracking-[0.08em] sm:text-4xl">
            SERVE <span className="text-ink">LAB</span>
          </h1>
          <p className="mt-1 text-lg text-muted-foreground">Análisis de saque con MediaPipe y redes entrenadas por el equipo</p>
        </div>
      </header>

      <main className="mx-auto grid max-w-[1440px] gap-8 px-4 pb-10 sm:px-6 lg:grid-cols-[380px_minmax(0,1fr)] lg:grid-rows-[auto_1fr] lg:gap-x-10 lg:px-10">
        <InputPanel
          key={resetKey}
          config={config}
          chosen={chosen}
          onChoose={choose}
          running={running}
          onAnalyze={() => chosen && analyze(sourceOf(chosen))}
        />
        <div ref={results} className="min-w-0 scroll-mt-4 lg:col-start-2 lg:row-span-2 lg:row-start-1">
          <ResultPanel config={config} run={run} current={current} previous={previous} summary={summary} />
        </div>
        <SessionPanel
          running={running}
          onReset={reset}
          onFinish={finish}
          attempts={attempts.length}
          player={player}
          onPlayer={setPlayer}
          summarizing={summarizing}
        />
      </main>

      <footer className="mx-auto max-w-[1440px] px-4 pb-8 text-muted-foreground sm:px-6 lg:px-10">
        No reemplaza a un entrenador.
      </footer>
      <Toaster position="top-center" />
    </div>
  )
}
