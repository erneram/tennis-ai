import { TriangleAlertIcon } from "lucide-react"
import type { ReactNode } from "react"
import type {
  BodyMeasure,
  Config,
  Feedback as Answer,
  ImprovementPoint,
  Occlusion,
  Result,
  Summary,
  TossProgress,
} from "@/api"
import type { Run } from "@/App"
import { Tips } from "@/components/InputPanel"
import { MomentTiles } from "@/components/MomentTiles"
import { ProgressPanel } from "@/components/ProgressPanel"
import { AlertBanner } from "@/components/thegridcn/alert"
import { Badge } from "@/components/thegridcn/badge"
import { DataCard } from "@/components/thegridcn/data-card"
import { EmptyState } from "@/components/thegridcn/empty-state"
import { HUDFrame } from "@/components/thegridcn/hud-frame"
import { cn } from "@/lib/utils"

export interface Attempt {
  n: number
  result: Result
}

interface Props {
  config: Config | null
  run: Run
  current?: Attempt
  previous?: Attempt
  summary: Summary | null
}

const CONFIDENCE = { high: "success", medium: "warning", low: "outline" } as const
const LEVEL = { validated: "success", trial: "warning", measure: "outline" } as const

export function ResultPanel({ config, run, current, previous, summary }: Props) {
  return (
    <div className="flex flex-col gap-8 @container">
      {summary ? <SummaryPanel summary={summary} /> : null}
      {run.status === "idle" ? <Idle /> : null}
      {run.status === "running" ? <ProgressPanel job={run.job} /> : null}
      {run.status === "done" && run.result.kind !== "feedback" ? (
        <Retake result={run.result} tips={config?.tips ?? []} />
      ) : null}
      {run.status === "done" && current ? <Feedback key={current.n} attempt={current} previous={previous} /> : null}
    </div>
  )
}

function Idle() {
  const steps = [
    "Elige un video de la demo o sube el tuyo, grabado de lado.",
    "Pulsa Analizar saque: tarda unos 10 s, todo en esta computadora.",
    "Mira hasta 3 puntos a mejorar, cada uno con su ejercicio y con qué tan seguro es.",
  ]
  return (
    <EmptyState title="Cómo funciona" className="animate-enter py-12">
      <ol className="mt-6 grid w-full max-w-3xl gap-4 text-left @2xl:grid-cols-3">
        {steps.map((s, i) => (
          <li key={s} className="flex gap-3 rounded-sm border border-primary/30 bg-background p-4 @2xl:flex-col">
            <span className="font-display font-bold text-2xl text-ink leading-none">{i + 1}</span>
            <span className="text-lg leading-snug">{s}</span>
          </li>
        ))}
      </ol>
    </EmptyState>
  )
}

function Retake({ result, tips }: { result: Result; tips: string[] }) {
  return (
    <div className="flex animate-enter flex-col gap-6">
      <AlertBanner
        title={
          result.title ??
          (result.kind === "retake" ? "Necesitamos una nueva grabación" : "No pudimos analizar el video")
        }
        icon={<TriangleAlertIcon aria-hidden="true" className="size-6" />}
      >
        {result.message}
      </AlertBanner>
      <HUDFrame label="Cómo grabar el saque">
        <Tips tips={tips} />
      </HUDFrame>
    </div>
  )
}

function Feedback({ attempt, previous }: { attempt: Attempt; previous?: Attempt }) {
  const r = attempt.result
  const f = r.feedback
  if (!f) return null
  const evidence = r.moments?.find((m) => m.evidence) ?? r.moments?.find((m) => m.id === "contact")

  return (
    <HUDFrame label="Resultado" className="animate-enter">
      <div className="flex flex-col gap-8">
        <header className="flex flex-col gap-2">
          <p className="label-hud font-mono text-muted-foreground">
            Intento {attempt.n}
            {r.seconds != null ? (
              <>
                {" "}
                · <span className="normal-case">{Math.round(r.seconds)} s</span>
              </>
            ) : null}
            {r.cached ? " · Respuesta guardada" : null}
          </p>
          <h2 className="font-bold text-3xl leading-tight sm:text-4xl">{f.title}</h2>
          <p className="text-lg text-muted-foreground">
            Fase: <span className="font-bold text-foreground">{f.phase_label}</span>
          </p>
        </header>

        <p className="max-w-[68ch] text-xl leading-normal">{f.observation}</p>

        {r.points?.length ? <Points points={r.points} /> : null}

        <div className="grid gap-4 @2xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
          {r.points?.length ? (
            <p className="max-w-[60ch] self-center text-lg text-muted-foreground">
              El orden va de lo más seguro a lo menos: primero lo que dice el modelo validado, después lo que está en
              prueba y al final las medidas comparadas con profesionales.
            </p>
          ) : (
            <DataCard subtitle="Ejercicio" title={f.drill.name}>
              <p className="text-lg leading-normal">{f.drill.instructions}</p>
            </DataCard>
          )}
          <div className="flex flex-col items-start gap-3 rounded border border-primary/30 bg-background p-5">
            <span className="label-hud text-muted-foreground">Brazo que lanza: qué tan seguro está</span>
            <Badge variant={CONFIDENCE[f.confidence]} size="md" dot>
              Confianza {f.confidence_label}
            </Badge>
            <Meter answer={f} />
            <p className="text-base">{f.certainty}</p>
          </div>
        </div>

        <section aria-labelledby="moments-title" className="flex flex-col gap-3">
          <SectionTitle id="moments-title">Momentos clave</SectionTitle>
          <MomentTiles moments={r.moments ?? []} />
        </section>

        {r.progress ? <Progress progress={r.progress} /> : null}
        {r.body?.length ? <Body measures={r.body} /> : null}

        {r.occlusion ? (
          <section aria-labelledby="why-title" className="flex flex-col gap-3">
            <SectionTitle id="why-title">En qué se fijó el modelo</SectionTitle>
            <p className="max-w-[68ch] text-lg">{f.reason}</p>
            <OcclusionMap occlusion={r.occlusion} />
          </section>
        ) : null}

        {r.slow_url ? (
          <section aria-labelledby="slow-title" className="flex flex-col gap-3">
            <SectionTitle id="slow-title">Cámara lenta</SectionTitle>
            <video
              src={r.slow_url}
              poster={evidence?.image_url}
              controls
              loop
              muted
              playsInline
              preload="metadata"
              className="aspect-video max-h-[min(55vh,420px)] w-full rounded-sm border border-primary/40 object-contain"
            />
          </section>
        ) : null}

        {previous ? <Comparison previous={previous} current={attempt} /> : null}
      </div>
    </HUDFrame>
  )
}

const percent = (p: number) => `${Math.round(p * 100)} %`

/** The probability of "drops early" against the threshold chosen for the most value. */
function Meter({ answer }: { answer: Answer }) {
  const low = Math.min(...answer.seeds)
  const high = Math.max(...answer.seeds)
  return (
    <div className="flex w-full flex-col gap-1.5">
      <div
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(answer.probability * 100)}
        aria-label="Probabilidad de que el brazo baje antes de tiempo"
        className="relative h-3 w-full rounded-sm border border-primary/40 bg-muted"
      >
        <span
          className="absolute inset-y-0 left-0 rounded-sm bg-ink/40"
          style={{ left: `${low * 100}%`, width: `${(high - low) * 100}%` }}
        />
        <span
          className={cn("absolute inset-y-0 w-1 -translate-x-1/2", answer.drops_early ? "bg-destructive" : "bg-success")}
          style={{ left: `${answer.probability * 100}%` }}
        />
        <span
          aria-hidden="true"
          className="absolute -inset-y-1 w-0.5 -translate-x-1/2 bg-foreground"
          style={{ left: `${answer.threshold * 100}%` }}
        />
      </div>
      <p className="text-sm text-muted-foreground">
        Probabilidad {percent(answer.probability)} (versiones: {percent(low)} a {percent(high)}); recomienda el
        ejercicio desde {percent(answer.threshold)}.
      </p>
    </div>
  )
}

/** The toss height on a line from 0 to past the professionals' median, their range shaded. */
function Progress({ progress }: { progress: TossProgress }) {
  const top = Math.max(progress.pros_median * 1.3, progress.height + 0.1)
  const at = (v: number) => `${(Math.min(Math.max(v, 0), top) / top) * 100}%`
  return (
    <section aria-labelledby="progress-title" className="flex flex-col gap-3">
      <SectionTitle id="progress-title">Tu mano frente a los profesionales</SectionTitle>
      <p className="max-w-[68ch] text-lg">{progress.text}</p>
      <div
        role="meter"
        aria-valuemin={0}
        aria-valuemax={Math.round(top * 100) / 100}
        aria-valuenow={progress.height}
        aria-label="Altura de la mano de lanzamiento, en torsos sobre el hombro"
        className="relative h-3 w-full max-w-xl rounded-sm border border-primary/40 bg-muted"
      >
        <span
          className="absolute inset-y-0 rounded-sm bg-success/30"
          style={{ left: at(progress.pros_floor), right: 0 }}
        />
        <span
          className={cn("absolute inset-y-0 w-1 -translate-x-1/2", progress.in_range ? "bg-success" : "bg-destructive")}
          style={{ left: at(progress.height) }}
        />
        <span
          aria-hidden="true"
          className="absolute -inset-y-1 w-0.5 -translate-x-1/2 bg-foreground"
          style={{ left: at(progress.pros_median) }}
        />
      </div>
      <p className="text-sm text-muted-foreground">
        {progress.in_range ? "Dentro" : "Fuera"} del rango de los profesionales (zona verde); la raya es su mediana. Es
        una medida de la pose, sin entrenar: sirve para seguir tu avance entre sesiones.
      </p>
    </section>
  )
}

/** Up to 3 points to improve, the surest first, each with its evidence level and its drill. */
function Points({ points }: { points: ImprovementPoint[] }) {
  return (
    <section aria-labelledby="points-title" className="flex flex-col gap-3">
      <SectionTitle id="points-title">Puntos a mejorar</SectionTitle>
      <ol className="flex flex-col gap-4">
        {points.map((p, i) => (
          <li key={p.aspect} className="flex flex-col gap-3 rounded border border-primary/30 bg-background p-5">
            <div className="flex flex-wrap items-center gap-3">
              <span className="font-display font-bold text-2xl text-ink leading-none">{i + 1}</span>
              <h4 className="font-bold text-2xl leading-tight">{p.label}</h4>
              <Badge variant={LEVEL[p.level]} size="md" dot>
                {p.level_label}
              </Badge>
            </div>
            <p className="max-w-[68ch] text-lg leading-normal">{p.seen}</p>
            <div className="rounded-sm border border-primary/20 bg-muted/40 p-4">
              <p className="label-hud text-muted-foreground">Ejercicio · {p.drill.name}</p>
              <p className="mt-1 text-lg leading-normal">{p.drill.instructions}</p>
            </div>
            <p className="max-w-[68ch] text-muted-foreground text-sm">{p.evidence}</p>
          </li>
        ))}
      </ol>
    </section>
  )
}

/** Pose measures that separate amateurs from professionals, next to the professionals'. */
function Body({ measures }: { measures: BodyMeasure[] }) {
  return (
    <section aria-labelledby="body-title" className="flex flex-col gap-3">
      <SectionTitle id="body-title">Tu cuerpo frente a los profesionales</SectionTitle>
      <dl className="grid gap-4 @2xl:grid-cols-2">
        {measures.map((m) => (
          <div key={m.aspect} className="rounded border border-primary/30 bg-background p-4">
            <dt className="flex flex-wrap items-center gap-2 font-bold text-lg">
              {m.label}
              <Badge variant={m.in_range ? "success" : "outline"} size="sm">
                {m.in_range ? "Dentro del rango" : "Fuera del rango"}
              </Badge>
            </dt>
            <dd className="mt-1 text-lg">{m.mine}</dd>
            <dd className="mt-1 text-muted-foreground">{m.pros}</dd>
          </div>
        ))}
      </dl>
      <p className="max-w-[68ch] text-sm text-muted-foreground">
        Son medidas de la pose comparadas con saques de profesionales, no un diagnóstico. Una medida solo sale arriba
        como punto a mejorar cuando además coincide con lo que marcó la persona que etiquetó los saques.
      </p>
    </section>
  )
}

/** Joints by stretch of time: red pushes toward "drops early", blue toward a clean serve. */
function OcclusionMap({ occlusion }: { occlusion: Occlusion }) {
  const largest = Math.max(0.02, ...occlusion.values.flat().map(Math.abs))
  const shade = (v: number) =>
    `color-mix(in oklch, var(${v >= 0 ? "--destructive" : "--ink"}) ${Math.round((Math.abs(v) / largest) * 85)}%, transparent)`
  return (
    <figure className="flex flex-col gap-2">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[420px] border-separate border-spacing-0.5 font-mono text-sm">
          <thead>
            <tr>
              <th scope="col" className="sr-only">
                Articulación
              </th>
              {occlusion.starts.map((s) => (
                <th key={s} scope="col" className="font-normal text-muted-foreground">
                  {s > 0 ? "+" : ""}
                  {s.toFixed(1)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {occlusion.parts.map((part, j) => (
              <tr key={part}>
                <th scope="row" className="whitespace-nowrap pr-2 text-left font-sans font-normal">
                  {part}
                </th>
                {occlusion.values[j].map((v, k) => (
                  <td
                    key={k}
                    title={`${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)} puntos`}
                    className="h-7 min-w-8 rounded-sm border border-border"
                    style={{ background: shade(v) }}
                  >
                    <span className="sr-only">{(v * 100).toFixed(1)} puntos</span>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <figcaption className="text-muted-foreground">
        Columnas: segundos desde el impacto. Cada casilla dice cuánto cambiaría la probabilidad si esa articulación se
        moviera, en ese tramo, como en un saque sin el problema. Rojo: empuja hacia «baja antes»; azul: hacia «está
        bien».
      </figcaption>
    </figure>
  )
}

function Comparison({ previous, current }: { previous: Attempt; current: Attempt }) {
  const before = previous.result.feedback
  const now = current.result.feedback
  if (!before || !now) return null
  const change = now.probability - before.probability
  const sentence =
    Math.abs(change) < 0.05
      ? `La probabilidad casi no cambió: ${percent(before.probability)} antes, ${percent(now.probability)} ahora.`
      : `La probabilidad de que el brazo baje antes ${change < 0 ? "bajó" : "subió"} de ${percent(before.probability)} a ${percent(now.probability)}.`
  const image = (a: Attempt) =>
    (a.result.moments?.find((m) => m.id === "arm_down") ?? a.result.moments?.find((m) => m.id === "contact"))
      ?.image_url

  return (
    <section aria-labelledby="compare-title" className="flex flex-col gap-3 border-primary/30 border-t pt-6">
      <SectionTitle id="compare-title">Comparación con el intento anterior</SectionTitle>
      <p className="text-lg">{sentence}</p>
      <div className="grid max-w-md grid-cols-2 gap-3">
        {[
          ["Intento " + previous.n, image(previous)],
          ["Intento " + current.n + " (este)", image(current)],
        ].map(([caption, src]) => (
          <figure key={caption} className="flex flex-col gap-1.5">
            <img
              src={src}
              alt={`${caption}: cuando la muñeca baja del hombro`}
              className="aspect-square w-full rounded-sm border border-primary/40 bg-muted object-contain"
            />
            <figcaption className="label-hud text-muted-foreground">{caption}</figcaption>
          </figure>
        ))}
      </div>
    </section>
  )
}

function SummaryPanel({ summary }: { summary: Summary }) {
  return (
    <HUDFrame label="Resumen de la sesión" className="animate-enter">
      <div className="flex flex-col gap-4">
        <p className="max-w-[68ch] text-xl leading-normal">{summary.text}</p>
        <dl className="grid gap-4 @2xl:grid-cols-2">
          <div>
            <dt className="label-hud text-muted-foreground">Ejercicio para la próxima sesión</dt>
            <dd className="mt-1 font-bold text-lg">{summary.focus_label}</dd>
          </div>
          <div>
            <dt className="label-hud text-muted-foreground">Cómo hacerlo</dt>
            <dd className="mt-1 text-lg">{summary.next_focus}</dd>
          </div>
        </dl>
        {summary.saved ? <p className="text-muted-foreground">✓ Guardado en esta computadora.</p> : null}
      </div>
    </HUDFrame>
  )
}

function SectionTitle({ id, children }: { id: string; children: ReactNode }) {
  return (
    <h3 id={id} className="label-hud text-ink">
      {children}
    </h3>
  )
}
