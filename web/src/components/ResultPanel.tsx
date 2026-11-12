import { TriangleAlertIcon } from "lucide-react"
import type { ReactNode } from "react"
import type {
  Compared,
  Config,
  ImprovementPoint,
  Occlusion,
  OtherPart,
  Result,
  Summary,
  TossModel,
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
const STATUS = { improve: "warning", fine: "success", outside: "outline", inside: "success", unseen: "outline" } as const
/** How often a point was right on validation, as a badge: green from 90 %, amber from 70 %. */
const sure = (precision: number) => (precision >= 0.9 ? "success" : precision >= 0.7 ? "warning" : "outline")
const percent = (p: number) => `${Math.round(p * 100)} %`
/** "a, b y c", or "a, b ni c" for what was not done. */
const list = (items: string[], last = "y") =>
  items.length > 1 ? `${items.slice(0, -1).join(", ")} ${last} ${items[items.length - 1]}` : (items[0] ?? "")

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
  const points = r.points ?? []
  const others = r.others ?? []
  const poster = r.moments?.find((m) => m.evidence) ?? r.moments?.find((m) => m.id === "contact")

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
          <h2 className="font-bold text-3xl leading-tight sm:text-4xl">{r.title}</h2>
          <p className="max-w-[68ch] text-lg text-muted-foreground">
            Revisamos {points.length + others.length} partes de tu saque
            {points.length
              ? `: ${points.length} ${points.length > 1 ? "tienen" : "tiene"} algo que mejorar.`
              : " y no vimos nada claro que mejorar."}
          </p>
        </header>

        {points.length ? (
          <Points points={points} />
        ) : r.drill ? (
          <DataCard subtitle="Ejercicio" title={r.drill.name}>
            <p className="text-lg leading-normal">{r.drill.instructions}</p>
          </DataCard>
        ) : null}

        {others.length ? <Others others={others} unchecked={r.unchecked ?? []} /> : null}

        <section aria-labelledby="moments-title" className="flex flex-col gap-3">
          <SectionTitle id="moments-title">Momentos clave</SectionTitle>
          <MomentTiles moments={r.moments ?? []} />
        </section>

        {r.slow_url ? (
          <section aria-labelledby="slow-title" className="flex flex-col gap-3">
            <SectionTitle id="slow-title">Cámara lenta</SectionTitle>
            <video
              src={r.slow_url}
              poster={poster?.image_url}
              controls
              loop
              muted
              playsInline
              preload="metadata"
              className="aspect-video max-h-[min(55vh,420px)] w-full rounded-sm border border-primary/40 object-contain"
            />
          </section>
        ) : null}

        {r.compare?.length ? <Professionals measures={r.compare} /> : null}
        {r.toss ? <TossDetails model={r.toss} /> : null}
        {previous ? <Comparison previous={previous} current={attempt} /> : null}
      </div>
    </HUDFrame>
  )
}

/** Up to 3 points to improve, the one that was right most often first, each with its drill. */
function Points({ points }: { points: ImprovementPoint[] }) {
  return (
    <section aria-labelledby="points-title" className="flex flex-col gap-3">
      <SectionTitle id="points-title">Puntos a mejorar</SectionTitle>
      <ol className="flex flex-col gap-4">
        {points.map((p, i) => (
          <li key={p.aspect} className="flex flex-col gap-3 rounded border border-primary/30 bg-background p-5">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
              <span className="font-display font-bold text-2xl text-ink leading-none">{i + 1}</span>
              <h4 className="font-bold text-2xl leading-tight">{p.label}</h4>
              <Badge variant={sure(p.precision)} size="md" dot>
                Acierta {p.often}
              </Badge>
            </div>
            <div className="flex max-w-[68ch] flex-col gap-1 text-lg leading-normal">
              <p>{p.seen}</p>
              <p className="text-muted-foreground">{p.goal}</p>
            </div>
            <div className="rounded-sm border border-primary/20 bg-muted/40 p-4">
              <p className="label-hud text-muted-foreground">Ejercicio · {p.drill.name}</p>
              <p className="mt-1 text-lg leading-normal">{p.drill.instructions}</p>
            </div>
            <p className="max-w-[68ch] text-muted-foreground text-sm">{p.trust}</p>
            <details className="max-w-[68ch] text-muted-foreground text-sm">
              <summary className="w-fit cursor-pointer">Detalle técnico</summary>
              <p className="mt-1">{p.detail}</p>
            </details>
          </li>
        ))}
      </ol>
      <p className="max-w-[68ch] text-muted-foreground">
        Van del punto en que la app acierta más al que menos. «Acierta» dice, de las veces que la app marcó ese punto
        en los videos con los que la probamos, cuántas lo había marcado también una persona que revisó el mismo
        video.
      </p>
    </section>
  )
}

/** The parts the app looked at that are not points: fine, next to the professionals', not seen. */
function Others({ others, unchecked }: { others: OtherPart[]; unchecked: string[] }) {
  return (
    <section aria-labelledby="others-title" className="flex flex-col gap-3">
      <SectionTitle id="others-title">Lo demás que revisamos</SectionTitle>
      <ul className="grid gap-4 @2xl:grid-cols-2">
        {others.map((o) => (
          <li key={o.aspect} className="rounded border border-primary/30 bg-background p-4">
            <p className="flex flex-wrap items-center gap-2 font-bold text-lg">
              {o.label}
              <Badge variant={STATUS[o.status]} size="sm">
                {o.status_label}
              </Badge>
            </p>
            <p className="mt-1 text-lg">{o.seen}</p>
            {o.goal ? <p className="mt-1 text-muted-foreground">{o.goal}</p> : null}
            {o.note ? <p className="mt-1 text-muted-foreground text-sm">{o.note}</p> : null}
          </li>
        ))}
      </ul>
      {unchecked.length ? (
        <p className="max-w-[68ch] text-muted-foreground text-sm">
          Todavía no revisamos {list(unchecked, "ni")}: la app aún no es confiable en eso.
        </p>
      ) : null}
    </section>
  )
}

/** Pose measures that separate amateurs from professionals, each on a line with their range and
 * what it shows in plain words; the numbers are in the bar's tooltip. */
function Professionals({ measures }: { measures: Compared[] }) {
  return (
    <section aria-labelledby="pros-title" className="flex flex-col gap-3">
      <SectionTitle id="pros-title">Frente a los profesionales</SectionTitle>
      <ul className="grid gap-4 @3xl:grid-cols-3">
        {measures.map((m) => (
          <li key={m.aspect} className="flex flex-col gap-2 rounded border border-primary/30 bg-background p-4">
            <p className="flex flex-wrap items-center gap-2 font-bold text-lg">
              {m.label}
              <Badge variant={m.in_range ? "success" : "outline"} size="sm">
                {m.in_range ? "Parecido a los profesionales" : "Por debajo de los profesionales"}
              </Badge>
            </p>
            <RangeMeter measure={m} />
            <p className="text-muted-foreground">{m.caption}</p>
          </li>
        ))}
      </ul>
      <p className="max-w-[68ch] text-muted-foreground text-sm">
        La raya de color es tu saque: más a la derecha es mejor. La zona verde es donde están 9 de cada 10 saques de
        jugadores profesionales, que medimos en video igual que el tuyo, y la raya negra es su saque típico. Sirve
        para ver tu avance entre sesiones; no es un diagnóstico.
      </p>
    </section>
  )
}

/** One measure on a line from 0 to past the professionals' median, their range shaded. */
function RangeMeter({ measure: m }: { measure: Compared }) {
  const top = Math.max(m.pros_median * 1.3, m.value + 0.1)
  const at = (v: number) => `${(Math.min(Math.max(v, 0), top) / top) * 100}%`
  return (
    <div
      role="meter"
      aria-valuemin={0}
      aria-valuemax={Math.round(top * 100) / 100}
      aria-valuenow={m.value}
      aria-label={`${m.label}, en largos de torso`}
      title={`Tú: ${m.value}. Profesionales: típico ${m.pros_median}, 9 de cada 10 desde ${m.pros_floor} (largos de torso).`}
      className="relative h-3 w-full rounded-sm border border-primary/40 bg-muted"
    >
      <span className="absolute inset-y-0 rounded-sm bg-success/30" style={{ left: at(m.pros_floor), right: 0 }} />
      <span
        className={cn("absolute inset-y-0 w-1 -translate-x-1/2", m.in_range ? "bg-success" : "bg-destructive")}
        style={{ left: at(m.value) }}
      />
      <span
        aria-hidden="true"
        className="absolute -inset-y-1 w-0.5 -translate-x-1/2 bg-foreground"
        style={{ left: at(m.pros_median) }}
      />
    </div>
  )
}

/** How the model of the tossing arm decided, folded: confidence, probability, occlusion map. */
function TossDetails({ model }: { model: TossModel }) {
  return (
    <details className="group rounded border border-primary/30 bg-background">
      <summary className="press label-hud flex cursor-pointer list-none items-center justify-between gap-3 p-4 text-ink [&::-webkit-details-marker]:hidden">
        Cómo decide el modelo del brazo que lanza
        <span aria-hidden="true" className="transition-transform duration-200 group-open:rotate-180">
          ▾
        </span>
      </summary>
      <div className="flex flex-col gap-4 border-primary/20 border-t p-4">
        <div className="flex flex-col items-start gap-3">
          <Badge variant={CONFIDENCE[model.confidence]} size="md" dot>
            Confianza {model.confidence_label}
          </Badge>
          <Meter model={model} />
          <p className="max-w-[68ch]">{model.certainty}</p>
        </div>
        <p className="max-w-[68ch] text-lg">{model.reason}</p>
        <OcclusionMap occlusion={model.occlusion} />
      </div>
    </details>
  )
}

/** The probability of "drops early" against the threshold chosen for the most value. */
function Meter({ model }: { model: TossModel }) {
  const low = Math.min(...model.seeds)
  const high = Math.max(...model.seeds)
  return (
    <div className="flex w-full max-w-xl flex-col gap-1.5">
      <div
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(model.probability * 100)}
        aria-label="Probabilidad de que el brazo baje antes de tiempo"
        className="relative h-3 w-full rounded-sm border border-primary/40 bg-muted"
      >
        <span
          className="absolute inset-y-0 left-0 rounded-sm bg-ink/40"
          style={{ left: `${low * 100}%`, width: `${(high - low) * 100}%` }}
        />
        <span
          className={cn("absolute inset-y-0 w-1 -translate-x-1/2", model.drops_early ? "bg-destructive" : "bg-success")}
          style={{ left: `${model.probability * 100}%` }}
        />
        <span
          aria-hidden="true"
          className="absolute -inset-y-1 w-0.5 -translate-x-1/2 bg-foreground"
          style={{ left: `${model.threshold * 100}%` }}
        />
      </div>
      <p className="text-sm text-muted-foreground">
        Probabilidad {percent(model.probability)} (versiones: {percent(low)} a {percent(high)}); lo marca desde{" "}
        {percent(model.threshold)}.
      </p>
    </div>
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

/** What changed in the points since the attempt before, with both contacts side by side. */
function Comparison({ previous, current }: { previous: Attempt; current: Attempt }) {
  const before = previous.result.points ?? []
  const now = current.result.points ?? []
  const names = (points: ImprovementPoint[]) => list(points.map((p) => p.label.toLowerCase()))
  const gone = before.filter((b) => !now.some((n) => n.aspect === b.aspect))
  const stays = now.filter((n) => before.some((b) => b.aspect === n.aspect))
  const fresh = now.filter((n) => !before.some((b) => b.aspect === n.aspect))
  const sentence =
    [
      gone.length ? `Ya no sale: ${names(gone)}.` : "",
      stays.length ? `Sigue: ${names(stays)}.` : "",
      fresh.length ? `Nuevo: ${names(fresh)}.` : "",
    ]
      .filter(Boolean)
      .join(" ") || "En ninguno de los dos intentos vimos un problema claro."
  const image = (a: Attempt) => a.result.moments?.find((m) => m.id === "contact")?.image_url

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
              alt={`${caption}: el impacto`}
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
