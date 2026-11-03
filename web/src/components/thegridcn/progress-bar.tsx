
import type * as React from "react";
import { cn } from "@/lib/utils";

interface ProgressBarProps extends React.HTMLAttributes<HTMLDivElement> {
  animated?: boolean;
  label?: string;
  max?: number;
  showValue?: boolean;
  size?: "sm" | "md" | "lg";
  striped?: boolean;
  value: number;
  variant?: "default" | "success" | "warning" | "danger";
}

const variantColors: Record<string, { bar: string }> = {
  danger: { bar: "bg-destructive" },
  default: { bar: "bg-primary" },
  success: { bar: "bg-success" },
  warning: { bar: "bg-warning" },
};

const sizeConfig: Record<string, { track: string; text: string }> = {
  lg: { text: "text-xs", track: "h-4" },
  md: { text: "text-sm", track: "h-2.5" },
  sm: { text: "text-xs", track: "h-1.5" },
};

export function ProgressBar({
  value,
  max = 100,
  label,
  showValue = false,
  variant = "default",
  size = "md",
  animated = true,
  striped = false,
  className,
  ...props
}: ProgressBarProps) {
  const pct = Math.min(Math.max((value / max) * 100, 0), 100);
  const colors = variantColors[variant];
  const sizeC = sizeConfig[size];

  // Fill follows the value with a short transform transition (was: re-ran 0 -> pct on every update).
  const displayPct = pct;

  return (
    <div
      data-slot="tron-progress-bar"
      className={cn("relative", className)}
      {...props}
    >
      {/* Label row */}
      {label || showValue ? (
        <div className="mb-1.5 flex items-center justify-between">
          {label ? (
            <span
              className={cn(
                "font-mono text-muted-foreground uppercase tracking-[0.12em]",
                sizeC.text
              )}
            >
              {label}
            </span>
          ) : null}
          {showValue ? (
            <span
              className={cn(
                "font-mono text-foreground tabular-nums",
                sizeC.text
              )}
            >
              {Math.round(displayPct)}%
            </span>
          ) : null}
        </div>
      ) : null}

      {/* Track */}
      <div
        className={cn(
          "relative overflow-hidden rounded-sm bg-foreground/10",
          sizeC.track
        )}
      >
        {/* Fill */}
        <div
          className={cn(
            "absolute inset-0 origin-left rounded-sm",
            animated && "transition-transform duration-500 ease-out",
            colors.bar
          )}
          style={{ transform: `scaleX(${displayPct / 100})` }}
        >
          {/* Stripe overlay */}
          {striped ? (
            <div
              className="absolute inset-0 opacity-20"
              style={{
                animation: animated
                  ? "progressStripe 0.8s linear infinite"
                  : undefined,
                backgroundImage:
                  "repeating-linear-gradient(45deg, transparent, transparent 4px, rgba(255,255,255,0.3) 4px, rgba(255,255,255,0.3) 8px)",
              }}
            />
          ) : null}

        </div>

      </div>
    </div>
  );
}
