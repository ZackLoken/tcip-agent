import type { Ref } from "react";

import type { EmbeddedToolOutcome } from "@/hooks/useEmbeddedToolRetry";

/**
 * A titled chrome bar over an iframe, for a tool the platform runs beside the app
 * (TensorBoard). Presentation only: launching, polling, reloading and stopping whatever
 * is embedded belong to the caller, which hands over what the launch has come to and reaches
 * into a same-origin frame through `frameRef`. A failed launch and a request that reached no
 * answer offer `onRetry`; a run that wrote no logs offers nothing, since no retry changes that.
 *
 * The iframe fills its parent, so the parent must have a definite height.
 */
export function EmbeddedTool({
  title,
  outcome,
  onRetry,
  frameRef,
}: {
  title: string;
  outcome: EmbeddedToolOutcome;
  onRetry: () => void;
  frameRef?: Ref<HTMLIFrameElement>;
}) {
  return (
    <div className="tcip-panel flex h-full flex-col overflow-hidden">
      <div className="flex items-center gap-2 border-b border-tcip-border px-3 py-1.5">
        <h2 className="tcip-heading">{title}</h2>
        <span className="flex-1" />
        {outcome.kind === "serving" && (
          <a
            className="text-[13px] text-tcip-accent underline"
            href={outcome.url}
            target="_blank"
            rel="noopener noreferrer"
            aria-label={`Open in a new tab: ${title}`}
          >
            Open in a new tab
          </a>
        )}
      </div>
      <div className="flex-1 overflow-hidden">
        {outcome.kind === "serving" ? (
          <iframe
            ref={frameRef}
            src={outcome.url}
            className="w-full h-full border-0"
            title={title}
          />
        ) : outcome.kind === "pending" ? (
          <div className="flex h-full items-center justify-center text-[13px] text-tcip-muted">
            Starting…
          </div>
        ) : (
          <div
            role="status"
            aria-live="polite"
            className="flex h-full flex-col items-center justify-center gap-2 px-3 text-center"
          >
            {outcome.kind === "no_logs" ? (
              <span className="text-[13px] tcip-problem">{outcome.message}</span>
            ) : (
              <>
                <span className="text-[13px] tcip-problem">
                  {outcome.kind === "unreached"
                    ? `${title} could not be reached: ${outcome.message}`
                    : outcome.output
                      ? `${outcome.error}: ${outcome.output}`
                      : outcome.error}
                </span>
                <button
                  type="button"
                  className="tcip-btn text-[13px]"
                  onClick={onRetry}
                  aria-label={`Try again: ${title}`}
                >
                  Try again
                </button>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
