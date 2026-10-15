import type { ReactNode } from "react";

export interface DetectionSystemRow {
  key: "hazard" | "fire" | "smoke" | "crowd" | "ppe";
  title: string;
  enabled: boolean;
}

export interface DetectionSystemsModel {
  rows: DetectionSystemRow[];
  /** the panel is usable only after /config has answered */
  ready: boolean;
  onToggle: (key: DetectionSystemRow["key"], enabled: boolean) => void;
}

/** Live Monitor "Detection systems" panel: one checkbox per system, bound to
 *  the same stored settings as the Settings card. Unticking stops new
 *  results for that system; stored alerts stay visible in Alerts/History.
 *  Presentation only — all state and saving live in App. */
export default function DetectionSystemsPanel({ model }: { model: DetectionSystemsModel }): ReactNode {
  return (
    <>
      {model.rows.map((row) => (
        <label key={row.key} className="system-row">
          <input
            type="checkbox"
            aria-label={`${row.title} enabled`}
            checked={row.enabled}
            disabled={!model.ready}
            onChange={(event) => model.onToggle(row.key, event.target.checked)}
          />
          <span>{row.title}</span>
        </label>
      ))}
    </>
  );
}
