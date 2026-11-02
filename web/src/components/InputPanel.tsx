import { CheckIcon } from "lucide-react"
import { toast } from "sonner"
import type { Config } from "@/api"
import { Chip } from "@/components/thegridcn/chip"
import { FileUpload } from "@/components/thegridcn/file-upload"
import { HUDFrame } from "@/components/thegridcn/hud-frame"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

export type Chosen = { kind: "file"; file: File; url: string } | { kind: "demo"; demo: Config["demo"][number] }

interface Props {
  config: Config | null
  chosen: Chosen | null
  onChoose: (c: Chosen) => void
  running: boolean
  onAnalyze: () => void
}

interface SessionProps {
  running: boolean
  onReset: () => void
  onFinish: () => void
  attempts: number
  player: string
  onPlayer: (v: string) => void
  summarizing: boolean
}

const VIDEO = /\.(mp4|mov)$/i

export function InputPanel(p: Props) {
  function onFiles(files: File[]) {
    const file = files[0]
    if (!file) return
    if (!VIDEO.test(file.name)) {
      toast.error("Elige un video .mp4 o .mov.")
      return
    }
    p.onChoose({ kind: "file", file, url: URL.createObjectURL(file) })
  }

  const src = p.chosen?.kind === "file" ? p.chosen.url : p.chosen?.demo.video_url
  const name = p.chosen?.kind === "file" ? p.chosen.file.name : p.chosen?.demo.label

  return (
    <aside className="flex flex-col gap-6 lg:self-start">
      <HUDFrame label="Entrada">
        <div className="flex flex-col gap-5">
          <FileUpload
            accept=".mp4,.mov,video/mp4,video/quicktime"
            label="Sube tu saque"
            description="Arrastra un video o toca para elegirlo · .mp4 o .mov"
            disabled={p.running}
            onFiles={onFiles}
          />

          {src ? (
            <figure className="animate-enter">
              <video
                key={src}
                src={src}
                controls
                muted
                playsInline
                preload="metadata"
                className="aspect-video w-full rounded-sm object-contain"
              />
              <figcaption className="mt-2 flex items-center gap-2 text-base">
                <span className="label-hud text-muted-foreground">Elegido</span>
                <span className="truncate font-mono text-sm">{name}</span>
              </figcaption>
            </figure>
          ) : null}

          <section aria-labelledby="demos" className="flex flex-col gap-3">
            <h3 id="demos" className="label-hud text-muted-foreground">
              Videos de la demo
            </h3>
            <div className="grid grid-cols-2 gap-2">
              {p.config?.demo.map((d) => {
                const selected = p.chosen?.kind === "demo" && p.chosen.demo.id === d.id
                const [kind, id] = d.label.includes(" · ") ? d.label.split(" · ") : [d.kind, d.label]
                return (
                  <Chip
                    key={d.id}
                    size="md"
                    selected={selected}
                    disabled={p.running}
                    onClick={() => p.onChoose({ kind: "demo", demo: d })}
                    className="min-h-14 justify-start gap-2 rounded-md px-3 py-1.5 text-left normal-case tracking-normal"
                  >
                    <span className="flex min-w-0 flex-1 flex-col leading-tight">
                      <span className="label-hud text-muted-foreground">{kind}</span>
                      <span className="truncate text-base">{id}</span>
                    </span>
                    {selected ? <CheckIcon aria-hidden="true" className="size-5 shrink-0 text-ink" /> : null}
                  </Chip>
                )
              })}
            </div>
          </section>

          <Button size="lg" className="w-full" disabled={!p.chosen || p.running} onClick={p.onAnalyze}>
            {p.running ? "Analizando…" : "Analizar saque"}
          </Button>

          <Accordion type="single" collapsible>
            <AccordionItem value="tips" className="border-primary/30 border-t">
              <AccordionTrigger>Cómo grabar el saque</AccordionTrigger>
              <AccordionContent>
                <Tips tips={p.config?.tips ?? []} />
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </div>
      </HUDFrame>
    </aside>
  )
}

export function SessionPanel(p: SessionProps) {
  return (
    <HUDFrame label="Sesión" className="self-start">
      <div className="flex flex-col gap-4">
        <p className="text-muted-foreground">
          {p.attempts === 0
            ? "Aún no hay intentos en esta sesión."
            : `${p.attempts} ${p.attempts === 1 ? "intento" : "intentos"} en esta sesión.`}
        </p>
        <label className="flex flex-col gap-1.5">
          <span className="label-hud text-muted-foreground">ID de jugador (opcional)</span>
          <Input
            value={p.player}
            onChange={(e) => p.onPlayer(e.target.value)}
            placeholder="Ej. jugador-07"
            autoComplete="off"
          />
        </label>
        <div className="grid grid-cols-2 gap-2">
          <Button variant="outline" disabled={p.attempts === 0 || p.running || p.summarizing} onClick={p.onFinish}>
            {p.summarizing ? "Resumiendo…" : "Terminar sesión"}
          </Button>
          <Button variant="ghost" className="border border-transparent" onClick={p.onReset}>
            Empezar de nuevo
          </Button>
        </div>
      </div>
    </HUDFrame>
  )
}

export function Tips({ tips }: { tips: string[] }) {
  return (
    <ol className="flex flex-col gap-2.5">
      {tips.map((tip, i) => (
        <li key={tip} className="flex gap-3">
          <span className="font-mono text-ink text-sm leading-[1.6rem]">{String(i + 1).padStart(2, "0")}</span>
          <span>{tip}</span>
        </li>
      ))}
    </ol>
  )
}
