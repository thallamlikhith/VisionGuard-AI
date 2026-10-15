/** Scroll safety net for the locked document/shell.
 *
 *  CSS alone (overflow: clip + a viewport-fixed .app-shell) makes the document
 *  and the shell unscrollable in supporting browsers, but two gaps remain:
 *  browsers without `overflow: clip` fall back to `overflow: hidden` (which
 *  still allows programmatic scrolling), and the browser may restore a saved
 *  scroll offset after a reload. So this module:
 *  1. sets history.scrollRestoration = "manual" so no past offset survives a
 *     reload, and
 *  2. installs ONE scroll listener that resets any document scroll offset
 *     back to 0 (scrollingElement, documentElement and body are all covered,
 *     because which one moves differs per browser).
 *
 *  Only the document offset is touched: .page (the real scroll container) and
 *  .modal-card keep whatever position they have.
 */
export function installScrollLock(): () => void {
  try {
    // Assigned directly (not guarded by `in`): jsdom's history mock has no
    // such property, but the expando still records the intent under test,
    // while real browsers get the real opt-out.
    (window.history as History & { scrollRestoration: ScrollRestoration }).scrollRestoration =
      "manual";
  } catch {
    /* a read-only history (some test doubles): the listener below still guards */
  }

  const resetDocumentScroll = (): void => {
    const scrolling = document.scrollingElement as HTMLElement | null;
    if (scrolling) {
      if (scrolling.scrollTop !== 0) scrolling.scrollTop = 0;
      if (scrolling.scrollLeft !== 0) scrolling.scrollLeft = 0;
    }
    const root = document.documentElement;
    if (root && root !== scrolling) {
      if (root.scrollTop !== 0) root.scrollTop = 0;
      if (root.scrollLeft !== 0) root.scrollLeft = 0;
    }
    const body = document.body;
    if (body && body !== scrolling) {
      if (body.scrollTop !== 0) body.scrollTop = 0;
      if (body.scrollLeft !== 0) body.scrollLeft = 0;
    }
  };

  // ONE listener: window sees every document scroll (scroll events do not
  // bubble, but the document scroll surfaces on window). Capture is
  // unnecessary for window-level document scrolls and would complicate
  // cleanup, so the default (bubble) phase is used.
  window.addEventListener("scroll", resetDocumentScroll, { passive: true });
  return () => window.removeEventListener("scroll", resetDocumentScroll);
}
