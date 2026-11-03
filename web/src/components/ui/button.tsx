import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "press inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md font-bold uppercase tracking-[0.08em] disabled:pointer-events-none disabled:opacity-45 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-5 shrink-0 [&_svg]:shrink-0 outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink",
  {
    variants: {
      variant: {
        default:
          "bg-primary text-primary-foreground hover:bg-[oklch(0.8_0.16_195)] focus-visible:shadow-[0_0_0_4px_color-mix(in_oklch,var(--primary)_35%,transparent)]",
        destructive: "bg-destructive text-white hover:bg-destructive/90",
        outline:
          "border border-primary/50 bg-card text-foreground hover:border-ink hover:bg-primary/5",
        secondary: "bg-secondary text-secondary-foreground hover:bg-accent",
        ghost: "text-foreground hover:bg-muted",
        link: "text-ink underline underline-offset-4 normal-case tracking-normal",
      },
      size: {
        default: "h-11 px-5 text-base",
        sm: "h-11 px-4 text-sm",
        lg: "h-13 px-6 text-lg",
        icon: "size-11",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

function Button({
  className,
  variant = "default",
  size = "default",
  asChild = false,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean
  }) {
  const Comp = asChild ? Slot : "button"

  return (
    <Comp
      data-slot="button"
      data-variant={variant}
      data-size={size}
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
}

export { Button, buttonVariants }
