import { useState } from "react"
import type { Moment } from "@/api"
import { HUDCornerFrame } from "@/components/thegridcn/hud-corner-frame"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { cn } from "@/lib/utils"

// The key moments around contact, with the skeleton drawn (tossing arm in orange).
// The moment that shows the problem is marked with a word and a border, never color alone.
export function MomentTiles({ moments }: { moments: Moment[] }) {
  const [open, setOpen] = useState<Moment | null>(null)

  return (
    <>
      <ol className="grid grid-cols-2 gap-x-3 gap-y-5 @2xl:grid-cols-4">
        {moments.map((m, i) => (
          <li key={m.id} className="animate-enter" style={{ animationDelay: `${i * 40}ms` }}>
            <button
              type="button"
              onClick={() => setOpen(m)}
              className="press group flex w-full flex-col gap-2 rounded-sm text-left"
              aria-label={`${m.label}${m.evidence ? ", aquí se ve el problema" : ""}. Ver en grande`}
            >
              <span
                className={cn(
                  "relative block aspect-square w-full overflow-hidden rounded-sm border bg-muted",
                  m.evidence ? "border-2 border-destructive" : "border-primary/40",
                )}
              >
                <img
                  src={m.image_url}
                  alt=""
                  loading="lazy"
                  className="size-full object-contain transition-transform duration-200 ease-out group-hover:scale-[1.03]"
                />
                {(["top-left", "top-right", "bottom-left", "bottom-right"] as const).map((pos) => (
                  <HUDCornerFrame
                    key={pos}
                    position={pos}
                    size={14}
                    className={cn("m-1", m.evidence ? "border-destructive" : "border-primary")}
                  />
                ))}
                {m.evidence ? (
                  <span className="absolute top-1.5 left-1.5 rounded-sm bg-foreground px-1.5 py-0.5 font-mono text-background text-xs">
                    ★ Aquí
                  </span>
                ) : null}
              </span>
              <span className="text-base leading-tight">{m.label}</span>
            </button>
          </li>
        ))}
      </ol>

      <Dialog open={!!open} onOpenChange={(o) => !o && setOpen(null)}>
        <DialogContent className="max-h-[92dvh] grid-rows-[auto_minmax(0,1fr)] sm:max-w-2xl">
          {open ? (
            <>
              <DialogHeader className="pr-10 text-left">
                <DialogTitle className="text-2xl">{open.label}</DialogTitle>
                <DialogDescription className="text-base">
                  {open.evidence ? "★ Aquí se ve el problema. " : ""}Brazo que lanza en naranja.
                </DialogDescription>
              </DialogHeader>
              <img
                src={open.image_url}
                alt={`Cuadro de ${open.label} con el esqueleto dibujado`}
                className={cn(
                  "mx-auto max-h-[75dvh] min-h-0 w-auto max-w-full rounded-sm border-2 bg-muted object-contain",
                  open.evidence ? "border-destructive" : "border-primary/40",
                )}
              />
            </>
          ) : null}
        </DialogContent>
      </Dialog>
    </>
  )
}
