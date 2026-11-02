
import type * as React from "react";
import { cn } from "@/lib/utils";

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  dot?: boolean;
  pulse?: boolean;
  size?: "sm" | "md";
  variant?: "default" | "success" | "warning" | "danger" | "outline";
}

export function Badge({
  variant = "default",
  size = "sm",
  dot = false,
  pulse = false,
  className,
  children,
  ...props
}: BadgeProps) {
  const variants: Record<string, string> = {
    danger: "border-destructive/40 bg-destructive/10 text-destructive",
    default: "border-ink/40 bg-primary/10 text-ink",
    outline: "border-foreground/25 bg-muted text-muted-foreground",
    success: "border-success/40 bg-success/10 text-success",
    warning: "border-warning-ink/50 bg-warning text-warning-foreground",
  };

  const dotColors: Record<string, string> = {
    danger: "bg-destructive",
    default: "bg-primary",
    outline: "bg-muted-foreground",
    success: "bg-success",
    warning: "bg-warning-ink",
  };

  return (
    <span
      data-slot="tron-badge"
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border font-mono uppercase tracking-[0.1em]",
        size === "sm" ? "px-2.5 py-0.5 text-xs" : "px-3 py-1 text-sm",
        variants[variant],
        className
      )}
      {...props}
    >
      {dot ? (
        <span className="relative flex h-1.5 w-1.5">
          {pulse ? (
            <span
              className={cn(
                "absolute inline-flex h-full w-full animate-ping rounded-full opacity-75",
                dotColors[variant]
              )}
            />
          ) : null}
          <span
            className={cn(
              "relative inline-flex h-1.5 w-1.5 rounded-full",
              dotColors[variant]
            )}
          />
        </span>
      ) : null}
      {children}
    </span>
  );
}
