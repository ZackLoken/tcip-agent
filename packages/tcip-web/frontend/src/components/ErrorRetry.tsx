/** A load error with its Retry control. */

const RETRY = "Retry";

/**
 * The error `message` in the problem tone, followed inline by a Retry button that calls
 * `onRetry`. `actsOn` is the short noun phrase for what the button reads again; the button is
 * named "Retry" followed by it. `className` is the placement and text size the site gives the line.
 */
export function ErrorRetry({
  message,
  onRetry,
  actsOn,
  className,
}: {
  message: string;
  onRetry: () => void;
  actsOn: string;
  className: string;
}) {
  return (
    <div className={`tcip-problem ${className}`}>
      {message}{" "}
      <button
        type="button"
        className="tcip-btn ml-1"
        aria-label={`${RETRY} ${actsOn}`}
        onClick={onRetry}
      >
        {RETRY}
      </button>
    </div>
  );
}
