import type { DrawTool } from "../types";

/** Paint-style shapes need many more corners than a hand-placed polygon: a
 *  circle is 24 and freehand is simplified down to 24, so the ceiling is 32. */
export const MAX_POINTS = 32;
export const MIN_POINTS = 3;

/** the toolbar, in the order a Paint user expects them */
export const TOOLS: { id: DrawTool; label: string; icon: string }[] = [
  { id: "select", label: "Select", icon: "↖" },
  { id: "rectangle", label: "Rectangle", icon: "▭" },
  { id: "quad", label: "4-point", icon: "◇" },
  { id: "polygon", label: "Polygon", icon: "⬠" },
  { id: "circle", label: "Circle", icon: "◯" },
  { id: "freehand", label: "Freehand", icon: "✎" },
];

/** What the six box colours on the picture mean.
 *
 *  The detector paints the boxes into the JPEG itself, so the words have to
 *  live under the video. Red is reserved for an active alert. Fire is scarlet
 *  and smoke steel blue, both clear of the person/zone palette. PPE violations
 *  are magenta: PPE is grading-only, so its boxes must never read as an
 *  alert. */
export const LEGEND: { label: string; tone: "safe" | "warn" | "occupy" | "danger" | "fire" | "smoke" | "ppe" }[] = [
  { label: "Person safe", tone: "safe" },
  { label: "Person in zone", tone: "danger" },
  { label: "Danger zone", tone: "warn" },
  { label: "Fire", tone: "fire" },
  { label: "Smoke", tone: "smoke" },
  { label: "PPE violation", tone: "ppe" },
];
