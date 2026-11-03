import { TriangleAlertIcon } from "lucide-react"
import type { ReactNode } from "react"
import type { Config, Result, Summary } from "@/api"
import type { Run } from "@/App"
import { Tips } from "@/components/InputPanel"
import { PhaseTiles, VERDICT } from "@/components/PhaseTiles"
import { ProgressPanel } from "@/components/ProgressPanel"
import { AlertBanner } from "@/components/thegridcn/alert"
import { Badge } from "@/components/thegridcn/badge"
import { DataCard } from "@/components/thegridcn/data-card"
import { EmptyState } from "@/components/thegridcn/empty-state"
import { HUDFrame } from "@/components/thegridcn/hud-frame"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
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
    "Pulsa Analizar saque: tarda unos 30 s, todo en esta computadora.",
    "Mira la fase a mejorar, el ejercicio y qué tan seguro está el asistente.",
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
  const confidence = CONFIDENCE[f.confidence]

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
          <h2 className="font-bold text-3xl leading-tight sm:text-4xl">
            {f.phase_label ? (
              <>
                <span className="text-muted-foreground">Fase a mejorar:</span> {f.phase_label}
              </>
            ) : (
              "No vemos un problema claro"
            )}
          </h2>
          {f.seen_in_label && f.seen_in !== f.phase ? (
            <p className="text-lg text-muted-foreground">
              Se ve en <span className="font-bold text-foreground">{f.seen_in_label}</span>
            </p>
          ) : null}
        </header>

        <p className="max-w-[68ch] text-xl leading-normal">{f.observation}</p>

        <div className="grid gap-4 @2xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
          <DataCard subtitle="Ejercicio" title={f.drill.name}>
            <p className="text-lg leading-normal">{f.drill.instructions}</p>
          </DataCard>
          <div className="flex flex-col items-start gap-2 rounded border border-primary/30 bg-background p-5">
            <span className="label-hud text-muted-foreground">Qué tan seguro está</span>
            <Badge variant={confidence} size="md" dot>
              Confianza {f.confidence_label}
            </Badge>
            <p className="text-base">{f.certainty}</p>
          </div>
        </div>

        <section aria-labelledby="phases-title" className="flex flex-col gap-3">
          <SectionTitle id="phases-title">Las seis fases</SectionTitle>
          <PhaseTiles phases={r.phases ?? []} />
        </section>

        {r.slow_url ? (
          <section aria-labelledby="slow-title" className="flex flex-col gap-3">
            <SectionTitle id="slow-title">Cámara lenta</SectionTitle>
            <video
              src={r.slow_url}
              poster={r.phases?.find((p) => p.id === (f.seen_in ?? f.phase ?? "contact"))?.image_url}
              controls
              loop
              muted
              playsInline
              preload="metadata"
              className="aspect-video max-h-[min(55vh,420px)] w-full rounded-sm border border-primary/40 object-contain"
            />
          </section>
        ) : null}

        {r.checks?.length ? (
          <Accordion type="single" collapsible>
            <AccordionItem value="checks" className="border-primary/30 border-y">
              <AccordionTrigger>Qué revisamos en cada fase</AccordionTrigger>
              <AccordionContent>
                <ul className="flex flex-col divide-y divide-border">
                  {r.checks.map((c, i) => {
                    const s = VERDICT[c.answer]
                    return (
                      <li key={i} className="flex gap-3 py-3">
                        <span
                          aria-hidden="true"
                          className={cn("w-5 shrink-0 text-center font-bold font-mono text-lg", s.text)}
                        >
                          {s.icon}
                        </span>
                        <div className="min-w-0">
                          <p className="label-hud text-muted-foreground">
                            {c.phase_label} <span className="sr-only">· {s.word}</span>
                          </p>
                          <p className="mt-1 font-bold text-lg leading-snug">
                            {c.part}: <span className={s.text}>{c.state}</span>
                          </p>
                        </div>
                      </li>
                    )
                  })}
                </ul>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        ) : null}

        {previous ? <Comparison previous={previous} current={attempt} /> : null}
      </div>
    </HUDFrame>
  )
}

function Comparison({ previous, current }: { previous: Attempt; current: Attempt }) {
  const before = previous.result.feedback
  const now = current.result.feedback
  const phase = now?.phase ?? before?.phase
  const label = (l?: string | null) => l ?? "ningún problema claro"
  const sentence =
    before?.phase === now?.phase
      ? `El punto a mejorar sigue siendo el mismo (${label(now?.phase_label)}).`
      : `Cambió el punto principal: antes ${label(before?.phase_label)}, ahora ${label(now?.phase_label)}.`
  const image = (a: Attempt) => a.result.phases?.find((p) => p.id === phase)?.image_url

  return (
    <section aria-labelledby="compare-title" className="flex flex-col gap-3 border-primary/30 border-t pt-6">
      <SectionTitle id="compare-title">Comparación con el intento anterior</SectionTitle>
      <p className="text-lg">{sentence}</p>
      {phase ? (
        <div className="grid max-w-md grid-cols-2 gap-3">
          {[
            ["Intento " + previous.n, image(previous)],
            ["Intento " + current.n + " (este)", image(current)],
          ].map(([caption, src]) => (
            <figure key={caption} className="flex flex-col gap-1.5">
              <img
                src={src}
                alt={`${caption}: cuadro de la fase`}
                className="aspect-square w-full rounded-sm border border-primary/40 bg-muted object-contain"
              />
              <figcaption className="label-hud text-muted-foreground">{caption}</figcaption>
            </figure>
          ))}
        </div>
      ) : null}
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
            <dt className="label-hud text-muted-foreground">Punto principal</dt>
            <dd className="mt-1 font-bold text-lg">{summary.focus_label}</dd>
          </div>
          <div>
            <dt className="label-hud text-muted-foreground">Para la próxima sesión</dt>
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
