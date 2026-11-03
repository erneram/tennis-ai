import { useState } from "react"
import type { Phase, Verdict } from "@/api"
import { HUDCornerFrame } from "@/components/thegridcn/hud-corner-frame"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { cn } from "@/lib/utils"

// Verdict -> mark, words and colors. Color is never the only signal: icon + word always shown.
export const VERDICT: Record<Verdict | "none", { icon: string; word: string; text: string; border: string }> = {
  yes: { icon: "✓", word: "Se ve bien", text: "text-success", border: "border-success" },
  no: { icon: "✕", word: "A mejorar", text: "text-destructive", border: "border-destructive" },
  unclear: { icon: "?", word: "No se ve claro", text: "text-muted-foreground", border: "border-muted-foreground" },
  none: { icon: "—", word: "Sin revisar", text: "text-muted-foreground", border: "border-border" },
}

export function PhaseTiles({ phases }: { phases: Phase[] }) {
  const [open, setOpen] = useState<Phase | null>(null)
  const v = open ? VERDICT[open.verdict ?? "none"] : null

  return (
    <>
      <ol className="grid grid-cols-2 gap-x-3 gap-y-5 @lg:grid-cols-3">
        {phases.map((p, i) => {
          const s = VERDICT[p.verdict ?? "none"]
          return (
            <li key={p.id} className="animate-enter" style={{ animationDelay: `${i * 40}ms` }}>
              <button
                type="button"
                onClick={() => setOpen(p)}
                className="press group flex w-full flex-col gap-2 rounded-sm text-left"
                aria-label={`${p.label}: ${s.word}${p.evidence ? ", aquí se ve el problema" : ""}. Ver en grande`}
              >
                <span
                  className={cn(
                    "relative block aspect-[4/3] w-full overflow-hidden rounded-sm border bg-muted",
                    s.border,
                    p.verdict ? "border-2" : "border-dashed",
                  )}
                >
                  <img
                    src={p.image_url}
                    alt=""
                    loading="lazy"
                    className="size-full object-contain transition-transform duration-200 ease-out group-hover:scale-[1.03]"
                  />
                  {(["top-left", "top-right", "bottom-left", "bottom-right"] as const).map((pos) => (
                    <HUDCornerFrame key={pos} position={pos} size={14} className={cn("m-1", s.border)} />
                  ))}
                  {p.evidence ? (
                    <span className="absolute top-1.5 left-1.5 rounded-sm bg-foreground px-1.5 py-0.5 font-mono text-background text-xs">
                      ★ Aquí
                    </span>
                  ) : null}
                </span>
                <span className={cn("flex items-center gap-1.5 font-bold text-sm leading-tight", s.text)}>
                  <span aria-hidden="true" className="font-mono">
                    {s.icon}
                  </span>
                  {s.word}
                </span>
                <span className="-mt-1.5 text-base leading-tight">{p.label}</span>
              </button>
            </li>
          )
        })}
      </ol>

      <Dialog open={!!open} onOpenChange={(o) => !o && setOpen(null)}>
        <DialogContent className="max-h-[92dvh] grid-rows-[auto_minmax(0,1fr)] sm:max-w-2xl">
          {open && v ? (
            <>
              <DialogHeader className="pr-10 text-left">
                <DialogTitle className="text-2xl">{open.label}</DialogTitle>
                <DialogDescription className={cn("flex items-center gap-1.5 font-bold text-base", v.text)}>
                  <span aria-hidden="true" className="font-mono">
                    {v.icon}
                  </span>
                  {v.word}
                  {open.evidence ? <span className="text-foreground">· ★ aquí se ve el problema</span> : null}
                </DialogDescription>
              </DialogHeader>
              <img
                src={open.image_url}
                alt={`Cuadro clave de ${open.label}`}
                className={cn(
                  "mx-auto max-h-[75dvh] min-h-0 w-auto max-w-full rounded-sm border-2 bg-muted object-contain",
                  v.border,
                )}
              />
            </>
          ) : null}
        </DialogContent>
      </Dialog>
    </>
  )
}
