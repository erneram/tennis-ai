
import type * as React from "react";
import { cn } from "@/lib/utils";

interface Step {
  description?: string;
  label: string;
}

interface StepperProps extends React.HTMLAttributes<HTMLDivElement> {
  currentStep: number;
  orientation?: "horizontal" | "vertical";
  steps: Step[];
}

export function Stepper({
  steps,
  currentStep,
  orientation = "horizontal",
  className,
  ...props
}: StepperProps) {
  const isVertical = orientation === "vertical";

  return (
    <div
      data-slot="tron-stepper"
      className={cn(
        "relative overflow-hidden rounded border border-primary/40 bg-card px-3 py-4 sm:p-5",
        className
      )}
      {...props}
    >
      <div
        className={cn(
          "flex",
          isVertical ? "flex-col gap-0" : "items-start gap-0"
        )}
      >
        {steps.map((step, i) => {
          const isActive = i === currentStep;
          const isCompleted = i < currentStep;
          const isLast = i === steps.length - 1;

          return (
            <div
              key={i}
              aria-current={isActive ? "step" : undefined}
              className={cn(
                "flex",
                isVertical
                  ? "flex-row gap-3"
                  : "flex-1 flex-col items-center gap-2"
              )}
            >
              <div
                className={cn(
                  "flex",
                  isVertical ? "flex-col items-center" : "w-full items-center"
                )}
              >
                {/* Step circle */}
                <div
                  className={cn(
                    "relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border-2 font-bold font-mono text-sm transition-[background-color,border-color,color] duration-300",
                    isCompleted &&
                      "border-success bg-success/10 text-success",
                    isActive &&
                      "border-ink bg-primary/25 text-foreground",
                    !(isCompleted || isActive) &&
                      "border-foreground/25 text-muted-foreground"
                  )}
                >
                  {isCompleted ? (
                    <span aria-hidden="true">&#10003;</span>
                  ) : (
                    <span>{i + 1}</span>
                  )}
                  {isActive && (
                    <div className="absolute inset-0 animate-ping rounded-full border border-primary opacity-20" />
                  )}
                </div>

                {/* Connector line */}
                {!isLast && (
                  <div
                    className={cn(
                      "relative overflow-hidden",
                      isVertical ? "mx-auto h-8 w-0.5" : "h-0.5 flex-1"
                    )}
                  >
                    <div
                      className={cn(
                        "absolute inset-0",
                        isCompleted ? "bg-success" : "bg-foreground/15"
                      )}
                    />
                    {isActive && (
                      <div
                        className="motion-loop absolute bg-primary"
                        style={{
                          animation: "stepperPulse 1.5s ease-in-out infinite",
                          ...(isVertical
                            ? { height: "40%", left: 0, right: 0, top: 0 }
                            : { bottom: 0, left: 0, top: 0, width: "40%" }),
                        }}
                      />
                    )}
                  </div>
                )}
              </div>

              {/* Label */}
              <div
                className={cn(
                  "px-1",
                  isVertical ? "flex-1 pb-4" : "text-center",
                  !(isCompleted || isActive) && "text-muted-foreground"
                )}
              >
                <div
                  className={cn(
                    "font-bold text-xs uppercase tracking-[0.08em] leading-tight sm:text-sm",
                    isActive && "text-ink"
                  )}
                >
                  {step.label}
                </div>
                {step.description ? (
                  <div className="mt-0.5 text-muted-foreground text-xs leading-relaxed">
                    {step.description}
                  </div>
                ) : null}
              </div>
            </div>
          );
        })}
      </div>

      {/* Corner decorations */}
      <div className="pointer-events-none absolute top-0 left-0 h-3 w-3 border-primary/50 border-t-2 border-l-2" />
      <div className="pointer-events-none absolute top-0 right-0 h-3 w-3 border-primary/50 border-t-2 border-r-2" />
      <div className="pointer-events-none absolute bottom-0 left-0 h-3 w-3 border-primary/50 border-b-2 border-l-2" />
      <div className="pointer-events-none absolute right-0 bottom-0 h-3 w-3 border-primary/50 border-r-2 border-b-2" />
    </div>
  );
}
