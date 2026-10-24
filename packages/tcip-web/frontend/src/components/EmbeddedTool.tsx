import type { Ref } from "react";

/**
 * A titled chrome bar over an iframe, for a tool the platform runs beside the app
 * (TensorBoard). Presentation only: launching, polling, reloading and stopping whatever
 * is embedded belong to the caller, which mounts this once it has a url, a launch in flight,
 * or a failure to show, and reaches into a same-origin frame through `frameRef`.
 *
 * The iframe fills its parent, so the parent must have a definite height.
 */
export function EmbeddedTool({
  title,
  url,
  loading = false,
  error = null,
  onRetry,
  frameRef,
}: {
  title: string;
  url: string | null;
  /** A launch was asked for and no url has come back yet. */
  loading?: boolean;
  error?: string | null;
  onRetry?: () => void;
  frameRef?: Ref<HTMLIFrameElement>;
}) {
  if (!url && !loading && !error) return null;

  return (
    <div className="tcip-panel flex h-full flex-col overflow-hidden">
      <div className="flex items-center gap-2 border-b border-tcip-border px-3 py-1.5">
        <h2 className="tcip-heading">{title}</h2>
        <span className="flex-1" />
        {url && (
          <a
            className="text-[13px] text-tcip-accent underline"
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            aria-label={`Open in a new tab: ${title}`}
          >
            Open in a new tab
          </a>
        )}
      </div>
      <div className="flex-1 overflow-hidden">
        {url ? (
          <iframe ref={frameRef} src={url} className="w-full h-full border-0" title={title} />
        ) : loading ? (
          <div className="flex h-full items-center justify-center text-[13px] text-tcip-muted">
            Starting…
          </div>
        ) : (
          <div
            role="status"
            aria-live="polite"
            className="flex h-full flex-col items-center justify-center gap-2 px-3 text-center"
          >
            <span className="text-[13px] tcip-problem">{error}</span>
            {onRetry && (
              <button
                type="button"
                className="tcip-btn text-[13px]"
                onClick={onRetry}
                aria-label={`Try again: ${title}`}
              >
                Try again
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
