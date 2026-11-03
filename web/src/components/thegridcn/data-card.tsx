
import type * as React from "react";
import { cn } from "@/lib/utils";

interface DataFieldProps {
  highlight?: boolean;
  label: string;
  value: string;
}

function DataField({ label, value, highlight = false }: DataFieldProps) {
  return (
    <div className="space-y-1">
      <div className="label-hud text-muted-foreground">
        {label}
      </div>
      <div className="flex items-center gap-2">
        <span className="text-ink">|</span>
        <span
          className={cn(
            "font-mono text-sm uppercase tracking-wide",
            highlight && "bg-primary/20 px-2 py-0.5"
          )}
        >
          {value}
        </span>
      </div>
    </div>
  );
}

interface DataCardProps extends React.HTMLAttributes<HTMLDivElement> {
  fields?: { label: string; value: string; highlight?: boolean }[];
  status?: "active" | "inactive" | "alert";
  subtitle?: string;
  title?: string;
}

export function DataCard({
  title,
  subtitle,
  fields = [],
  status = "active",
  className,
  children,
  ...props
}: DataCardProps) {
  const statusColors = {
    active: "border-primary/50",
    alert: "border-destructive/50",
    inactive: "border-muted",
  };

  return (
    <div
      data-slot="tron-data-card"
      data-status={status}
      className={cn(
        "relative overflow-hidden rounded border bg-card",
        statusColors[status],
        className
      )}
      {...props}
    >
      {/* Header */}
      {title || subtitle ? (
        <div className="border-border/70 border-b px-5 py-3">
          {subtitle ? (
            <div className="label-hud text-muted-foreground">
              {subtitle}
            </div>
          ) : null}
          {title ? (
            <div className="flex items-center gap-2">
              <span className="text-ink">|</span>
              <h3 className="font-bold text-xl leading-tight">
                {title}
              </h3>
            </div>
          ) : null}
        </div>
      ) : null}

      {/* Fields */}
      <div className="space-y-3 px-5 py-4">
        {children}
        {fields.map((field, index) => (
          <DataField
            key={index}
            label={field.label}
            value={field.value}
            highlight={field.highlight}
          />
        ))}
      </div>

      {/* Corner decorations */}
      <div className="pointer-events-none absolute top-0 left-0 h-4 w-4 border-primary/50 border-t-2 border-l-2" />
      <div className="pointer-events-none absolute top-0 right-0 h-4 w-4 border-primary/50 border-t-2 border-r-2" />
      <div className="pointer-events-none absolute bottom-0 left-0 h-4 w-4 border-primary/50 border-b-2 border-l-2" />
      <div className="pointer-events-none absolute right-0 bottom-0 h-4 w-4 border-primary/50 border-r-2 border-b-2" />
    </div>
  );
}
