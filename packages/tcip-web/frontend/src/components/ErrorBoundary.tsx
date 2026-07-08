import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  /** When this value changes the boundary clears its error (e.g. pass the active
   *  tab so switching away from a crashed tab recovers automatically). */
  resetKey?: string | number;
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * Catches render-time errors in a subtree so one crashing tab does not white-screen
 * the entire app shell (the frontend previously had no error boundary anywhere, so a
 * single render-time TypeError took down everything). Shows a minimal fallback and
 * recovers when `resetKey` changes.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("ErrorBoundary caught:", error, info.componentStack);
  }

  componentDidUpdate(prev: Props): void {
    if (prev.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null });
    }
  }

  render(): ReactNode {
    if (this.state.error) {
      return (
        <div className="flex-1 flex items-center justify-center bg-tcip-canvas p-6">
          <div className="max-w-lg rounded-lg border border-tcip-fp/40 bg-tcip-panel px-5 py-4">
            <p className="text-sm font-semibold text-tcip-fp">Something went wrong in this view</p>
            <p className="mt-1 text-xs text-tcip-muted break-words">{this.state.error.message}</p>
            <button
              className="tcip-btn mt-3 text-[11px]"
              onClick={() => this.setState({ error: null })}
            >
              Try again
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
