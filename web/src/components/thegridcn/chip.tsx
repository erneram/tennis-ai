
import type * as React from "react";
import { cn } from "@/lib/utils";

interface ChipProps extends React.HTMLAttributes<HTMLButtonElement> {
  disabled?: boolean;
  onRemove?: () => void;
  selected?: boolean;
  size?: "sm" | "md";
  variant?: "default" | "success" | "warning" | "danger";
}

export function Chip({
  selected = false,
  onRemove,
  variant = "default",
  size = "sm",
  disabled = false,
  className,
  children,
  ...props
}: ChipProps) {
  const variants: Record<string, { base: string; active: string }> = {
    danger: {
      active: "border-destructive bg-destructive/10 text-destructive",
      base: "border-destructive/30 bg-card text-foreground hover:border-destructive",
    },
    default: {
      active: "border-ink bg-primary/20 text-foreground",
      base: "border-primary/40 bg-card text-foreground hover:border-ink hover:bg-primary/5",
    },
    success: {
      active: "border-success bg-success/10 text-success",
      base: "border-success/30 bg-card text-foreground hover:border-success",
    },
    warning: {
      active: "border-warning-ink bg-warning text-warning-foreground",
      base: "border-warning-ink/40 bg-card text-foreground hover:border-warning-ink",
    },
  };

  const v = variants[variant];

  return (
    <button
      type="button"
      disabled={disabled}
      aria-pressed={selected}
      data-slot="tron-chip"
      className={cn(
        "press inline-flex items-center gap-1.5 rounded-full border font-mono uppercase tracking-[0.08em]",
        size === "sm" ? "min-h-9 px-3 text-xs" : "min-h-11 px-4 text-sm",
        disabled && "cursor-not-allowed opacity-40",
        selected ? v.active : v.base,
        className
      )}
      {...props}
    >
      {children}
      {onRemove ? (
        <span
          role="button"
          tabIndex={0}
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.stopPropagation();
              onRemove();
            }
          }}
          className="ml-0.5 flex h-3 w-3 items-center justify-center rounded-full hover:bg-foreground/10"
        >
          <svg
            aria-hidden="true"
            width="5"
            height="5"
            viewBox="0 0 5 5"
            fill="none"
          >
            <path
              d="M0.5 0.5l4 4M4.5 0.5l-4 4"
              stroke="currentColor"
              strokeWidth="0.8"
              strokeLinecap="round"
            />
          </svg>
        </span>
      ) : null}
    </button>
  );
}
