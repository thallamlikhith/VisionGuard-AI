import { Component } from "react";
import type { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/** Top-level crash guard: one bad render (a malformed /state payload, a zone
 *  with no points) shows a recovery screen instead of blanking the whole app. */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error): void {
    // visible in the devtools console; the screen itself stays human-readable
    console.error("UI crashed:", error);
  }

  render(): ReactNode {
    if (this.state.error) {
      return (
        <div className="app-shell">
          <div className="page">
            <main className="main">
              <div className="inline-note error" role="alert">
                Something went wrong showing this page ({this.state.error.message}).
              </div>
              <div className="button-row">
                <button
                  type="button"
                  className="primary"
                  onClick={() => this.setState({ error: null })}
                >
                  Try again
                </button>
                <button type="button" onClick={() => window.location.reload()}>
                  Reload
                </button>
              </div>
            </main>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
