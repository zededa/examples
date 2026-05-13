import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertCircle, RotateCw } from 'lucide-react';

interface Props {
  children: ReactNode;
  /** If provided, used as the label in the reset button. */
  resetLabel?: string;
}
interface State {
  err: Error | null;
}

/**
 * App-level error boundary. Catches render errors (e.g. a response with an
 * unexpected shape being rendered as a React child) and shows a visible
 * recovery UI instead of blanking the screen.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { err: null };

  static getDerivedStateFromError(err: Error): State {
    return { err };
  }

  componentDidCatch(err: Error, info: ErrorInfo): void {
    // Surface to the console for easier debugging without swallowing.
    console.error('ErrorBoundary caught:', err, info);
  }

  reset = () => this.setState({ err: null });

  render() {
    if (!this.state.err) return this.props.children;

    return (
      <div
        className="m-3 flex flex-col items-start gap-3 rounded-xl border p-4 text-sm"
        style={{
          background: 'var(--color-error-light)',
          borderColor: 'rgba(239, 68, 68, 0.3)',
          color: 'var(--gray-800)',
        }}
      >
        <div className="flex items-center gap-2 font-semibold">
          <AlertCircle className="h-4 w-4" style={{ color: 'var(--color-error)' }} />
          <span>Something went wrong.</span>
        </div>
        <pre
          className="max-w-full overflow-auto rounded-lg border px-3 py-2 font-mono text-xs leading-5"
          style={{
            borderColor: 'var(--gray-200)',
            background: 'var(--island-bg)',
            color: 'var(--gray-700)',
            maxHeight: 200,
          }}
        >
          {String(this.state.err?.message ?? this.state.err)}
        </pre>
        <button
          type="button"
          onClick={this.reset}
          className="flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium"
          style={{
            borderColor: 'var(--zededa-cyan-border)',
            background: 'var(--primary-10)',
            color: 'var(--zededa-cyan)',
          }}
        >
          <RotateCw className="h-3 w-3" /> {this.props.resetLabel ?? 'Try again'}
        </button>
      </div>
    );
  }
}
