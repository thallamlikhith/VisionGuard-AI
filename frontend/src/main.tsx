import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import ErrorBoundary from "./components/ErrorBoundary";
import { installScrollLock } from "./utils/scrollLock";
import "./styles.css";

// The document and the shell are locked (overflow: clip + viewport-fixed
// shell); this is the safety net: no restored offset survives a reload and
// any document scroll offset is reset to 0. Runs before first paint.
installScrollLock();

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
);
