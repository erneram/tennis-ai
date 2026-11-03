import type * as React from "react";
import { cn } from "@/lib/utils";

// Gridcn AlertBanner, adapted to the light theme: solid fill + dark text, optional icon and body.
interface AlertBannerProps extends Omit<React.HTMLAttributes<HTMLDivElement>, "title"> {
  icon?: React.ReactNode;
  subtitle?: string;
  title: string;
  variant?: "warning" | "danger" | "info";
}

const variantStyles = {
  danger: "border-destructive bg-destructive/10 text-foreground",
  info: "border-ink/40 bg-primary/10 text-foreground",
  warning: "border-warning-ink/50 bg-warning text-warning-foreground",
};

export function AlertBanner({
  title,
  subtitle,
  icon,
  variant = "warning",
  className,
  children,
  ...props
}: AlertBannerProps) {
  return (
    <div
      role="alert"
      data-slot="tron-alert-banner"
      data-variant={variant}
      className={cn(
        "relative flex gap-3 overflow-hidden rounded-sm border px-5 py-4",
        variantStyles[variant],
        className
      )}
      {...props}
    >
      {icon ? <span className="mt-0.5 shrink-0">{icon}</span> : null}
      <div className="min-w-0">
        {subtitle ? <div className="label-hud mb-1">{subtitle}</div> : null}
        <div className="font-bold text-xl leading-tight">{title}</div>
        {children ? <div className="mt-1.5 text-[1.0625rem]">{children}</div> : null}
      </div>
    </div>
  );
}
