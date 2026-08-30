import type { ReactNode } from "react";

export function FilterChip({
  children,
  warn,
  title,
}: {
  children: ReactNode;
  warn?: boolean;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={
        warn
          ? "rounded border border-tcip-warn bg-tcip-warn/10 px-1.5 py-0.5 text-tcip-warn"
          : "rounded border border-tcip-border bg-tcip-bg px-1.5 py-0.5 text-tcip-muted"
      }
    >
      {children}
    </span>
  );
}
