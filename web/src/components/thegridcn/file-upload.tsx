
import * as React from "react";
import { cn } from "@/lib/utils";

interface FileUploadProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, "onDrop"> {
  accept?: string;
  description?: string;
  disabled?: boolean;
  label?: string;
  maxSize?: number;
  multiple?: boolean;
  onFiles?: (files: File[]) => void;
}

export function FileUpload({
  accept,
  multiple = false,
  maxSize,
  onFiles,
  label = "DROP FILES HERE",
  description = "or click to browse",
  disabled = false,
  className,
  ...props
}: FileUploadProps) {
  const [dragging, setDragging] = React.useState(false);
  const inputRef = React.useRef<HTMLInputElement>(null);

  function handleFiles(fileList: FileList) {
    const arr = Array.from(fileList);
    const filtered = maxSize ? arr.filter((f) => f.size <= maxSize) : arr;
    onFiles?.(filtered);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    if (disabled || !e.dataTransfer.files.length) {
      return;
    }
    handleFiles(e.dataTransfer.files);
  }

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files) {
      handleFiles(e.target.files);
    }
    e.target.value = "";
  }

  function formatSize(bytes: number) {
    if (bytes < 1024) {
      return `${bytes} B`;
    }
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  return (
    <div
      data-slot="tron-file-upload"
      className={cn("space-y-2", className)}
      {...props}
    >
      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-disabled={disabled}
        aria-label={`${label}. ${description}`}
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(e) => {
          if (!disabled && (e.key === "Enter" || e.key === " ")) {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) {
            setDragging(true);
          }
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        className={cn(
          "group relative flex cursor-pointer flex-col items-center justify-center rounded border-2 border-dashed px-6 py-8 text-center transition-[background-color,border-color]",
          disabled && "cursor-not-allowed opacity-40",
          dragging
            ? "border-ink bg-primary/10"
            : "border-primary/50 bg-card hover:border-ink hover:bg-primary/5"
        )}
      >
        {/* Upload icon */}
        <svg
          aria-hidden="true"
          width="28"
          height="28"
          viewBox="0 0 28 28"
          fill="none"
          className={cn(
            "mb-3 transition-colors",
            dragging
              ? "text-ink"
              : "text-ink"
          )}
        >
          <path
            d="M14 18V6M14 6l-5 5M14 6l5 5"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M4 18v4a2 2 0 002 2h16a2 2 0 002-2v-4"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </svg>

        <span
          className={cn(
            "font-bold text-lg leading-tight",
            dragging ? "text-ink" : "text-foreground"
          )}
        >
          {label}
        </span>
        <span className="mt-1 text-muted-foreground text-base">
          {description}
        </span>

        {maxSize ? (
          <span className="mt-1 font-mono text-muted-foreground text-xs">
            MAX {formatSize(maxSize)}
          </span>
        ) : null}

        <input
          ref={inputRef}
          type="file"
          accept={accept}
          multiple={multiple}
          onChange={handleChange}
          className="hidden"
          tabIndex={-1}
        />

        {/* Corner decorations */}
        <div className="pointer-events-none absolute top-1 left-1 h-3 w-3 border-primary/30 border-t border-l" />
        <div className="pointer-events-none absolute top-1 right-1 h-3 w-3 border-primary/30 border-t border-r" />
        <div className="pointer-events-none absolute bottom-1 left-1 h-3 w-3 border-primary/30 border-b border-l" />
        <div className="pointer-events-none absolute right-1 bottom-1 h-3 w-3 border-primary/30 border-r border-b" />
      </div>

    </div>
  );
}
