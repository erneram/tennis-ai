import type { Job } from "@/api"
import { HUDFrame } from "@/components/thegridcn/hud-frame"
import { ProgressBar } from "@/components/thegridcn/progress-bar"
import { Stepper } from "@/components/thegridcn/stepper"
import { Skeleton } from "@/components/ui/skeleton"

const STEPS = [
  { label: "Revisar video" },
  { label: "Buscar fases" },
  { label: "Revisar cada fase" },
  { label: "Cámara lenta" },
]

export function ProgressPanel({ job }: { job: Job | null }) {
  const step = Math.max(0, (job?.step ?? 0) - 1)
  return (
    <HUDFrame label="Analizando" className="animate-enter" aria-busy="true">
      <div className="flex flex-col gap-5 @container">
        <Stepper steps={STEPS} currentStep={step} />
        <ProgressBar value={(job?.progress ?? 0) * 100} label="Progreso" showValue />
        <p aria-live="polite" className="text-lg">
          {job?.message || "Preparando el análisis…"}
        </p>
        <p className="text-muted-foreground">Tarda unos 30 segundos. Todo ocurre en esta computadora.</p>

        <div aria-hidden="true" className="mt-2 flex flex-col gap-3">
          <Skeleton className="h-4 w-48" />
          <Skeleton className="h-8 w-3/4" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-5/6" />
          <div className="mt-3 grid grid-cols-3 gap-3 @2xl:grid-cols-6">
            {Array.from({ length: 6 }, (_, i) => (
              <Skeleton key={i} className="aspect-square w-full" />
            ))}
          </div>
        </div>
      </div>
    </HUDFrame>
  )
}
