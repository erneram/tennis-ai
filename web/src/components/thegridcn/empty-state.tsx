
import type * as React from "react";
import { cn } from "@/lib/utils";

interface EmptyStateProps extends React.HTMLAttributes<HTMLDivElement> {
  action?: { label: string; onClick?: () => void };
  description?: string;
  icon?: React.ReactNode;
  title: string;
}

export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
  children,
  ...props
}: EmptyStateProps) {
  return (
    <div
      data-slot="tron-empty-state"
      className={cn(
        "relative flex flex-col items-center justify-center rounded border border-primary/50 border-dashed bg-card px-6 py-10 text-center sm:px-10",
        className
      )}
      {...props}
    >
      {icon ? (
        <span className="mb-3 flex h-12 w-12 items-center justify-center text-ink">
          {icon}
        </span>
      ) : (
        <svg
          aria-hidden="true"
          width="48"
          height="48"
          viewBox="0 0 48 48"
          fill="none"
          className="mb-4 text-ink"
        >
          <rect
            x="8"
            y="8"
            width="32"
            height="32"
            rx="4"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeDasharray="4 4"
          />
          <path
            d="M20 24h8M24 20v8"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </svg>
      )}

      <h3 className="label-hud text-ink">
        {title}
      </h3>

      {description ? (
        <p className="mt-1.5 max-w-md text-muted-foreground">
          {description}
        </p>
      ) : null}

      {action ? (
        <button
          type="button"
          onClick={action.onClick}
          className="mt-4 rounded border border-primary/30 bg-primary/10 px-4 py-1.5 font-mono text-xs text-ink uppercase tracking-widest press hover:bg-primary/20"
        >
          {action.label}
        </button>
      ) : null}

      {children}

      {/* Corner decorations */}
      <div className="pointer-events-none absolute top-1 left-1 h-2.5 w-2.5 border-ink/50 border-t border-l" />
      <div className="pointer-events-none absolute top-1 right-1 h-2.5 w-2.5 border-ink/50 border-t border-r" />
      <div className="pointer-events-none absolute bottom-1 left-1 h-2.5 w-2.5 border-ink/50 border-b border-l" />
      <div className="pointer-events-none absolute right-1 bottom-1 h-2.5 w-2.5 border-ink/50 border-r border-b" />
    </div>
  );
}
