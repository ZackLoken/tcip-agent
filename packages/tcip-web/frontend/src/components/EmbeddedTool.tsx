/**
 * A titled chrome bar over an iframe, for the tools the platform runs beside the app
 * (TensorBoard, Ray's dashboard). Presentation only: launching, polling and stopping whatever
 * is embedded belong to the caller, which mounts this once it has a url, a launch in flight,
 * or a failure to show.
 *
 * The iframe fills its parent, so the parent must have a definite height.
 */
export function EmbeddedTool({
  title,
  url,
  loading = false,
  error = null,
  onRetry,
}: {
  title: string;
  url: string | null;
  /** A launch was asked for and no url has come back yet. */
  loading?: boolean;
  error?: string | null;
  onRetry?: () => void;
}) {
  if (!url && !loading && !error) return null;

  return (
    <div className="tcip-panel flex h-full flex-col overflow-hidden">
      <div className="flex items-center gap-2 border-b border-tcip-border px-3 py-1.5">
        <h2 className="tcip-heading">{title}</h2>
        <span className="flex-1" />
        {url && (
          <a
            className="text-[11px] text-tcip-accent underline"
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
          <iframe src={url} className="w-full h-full border-0" title={title} />
        ) : loading ? (
          <div className="flex h-full items-center justify-center text-[11px] text-tcip-muted">
            Starting…
          </div>
        ) : (
          <div
            role="status"
            aria-live="polite"
            className="flex h-full flex-col items-center justify-center gap-2 px-3 text-center"
          >
            <span className="text-[11px] text-tcip-fp">{error}</span>
            {onRetry && (
              <button
                type="button"
                className="tcip-btn text-[11px]"
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
