/** A load error with its Retry control. */

/**
 * The error `message` in the problem tone, followed inline by a Retry button that calls
 * `onRetry`. `className` is the placement and text size the site gives the line.
 */
export function ErrorRetry({
  message,
  onRetry,
  className,
}: {
  message: string;
  onRetry: () => void;
  className: string;
}) {
  return (
    <div className={`tcip-problem ${className}`}>
      {message}{" "}
      <button type="button" className="tcip-btn ml-1" onClick={onRetry}>
        Retry
      </button>
    </div>
  );
}
