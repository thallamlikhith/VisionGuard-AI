import type { ReactNode } from "react";
import type { CrowdState, PpeState } from "../types";

export interface SidePathsReadout {
  crowd: CrowdState | null;
  ppe: PpeState | null;
  running: boolean;
}

/** Short display name for a canonical "NO-X" violation class. */
function shortViolation(canonical: string): string {
  const withoutNo = canonical.startsWith("NO-") ? canonical.slice("NO-".length) : canonical;
  return withoutNo === "Safety Vest" ? "Vest" : withoutNo;
}

/** Right-column cards for the annotation-only side paths: crowd numbers and
 *  the PPE grade. The cards always render so the column layout holds; the
 *  live figures inside show only when ticked and running, otherwise a muted
 *  Off line. Deliberately neutral styling — neither path raises alerts, so
 *  neither may use the red/amber alert tones. */
export default function SidePathsReadout({ model }: { model: SidePathsReadout }): ReactNode {
  const crowdLive = model.running && model.crowd?.enabled === true;
  const ppeLive = model.running && model.ppe?.enabled === true;
  const metrics = model.crowd?.metrics;
  const grade = model.ppe?.grade ?? "none";
  const stats = model.ppe?.stats;
  return (
    <div data-testid="side-paths">
      <section className="card" data-testid="crowd-card">
        <h2>People counted</h2>
        {crowdLive ? (
          <div className="tile" data-testid="crowd-tile">
            <span className="tile-label">People in view</span>
            <span className="tile-value" data-testid="crowd-count">
              {metrics?.person_count ?? 0}
            </span>
            <span className="tile-of" data-testid="crowd-detail">
              {((metrics?.coverage_fraction ?? 0) * 100).toFixed(1)}% of view ·{" "}
              {(metrics?.density_per_megapixel ?? 0).toFixed(1)}/MP
            </span>
            <span className="tile-of">Not the same as zone capacity</span>
          </div>
        ) : (
          <p className="muted small">Off</p>
        )}
      </section>
      <section className="card" data-testid="ppe-card">
        <h2>PPE grade</h2>
        {ppeLive ? (
          <div className="tile" data-testid="ppe-tile">
            <span className="tile-label">Grade</span>
            <span className="tile-value" data-testid="ppe-grade">
              {grade}
            </span>
            <span className="tile-of" data-testid="ppe-detail">
              {(stats?.violation_count ?? 0)} violations / {(stats?.person_count ?? 0)} people
            </span>
            {(stats?.violations_by_class ?? []).length > 0 && (
              <span className="tile-of" data-testid="ppe-breakdown">
                {(stats?.violations_by_class ?? [])
                  .map(([name, count]) => `${shortViolation(name)} ${count}`)
                  .join(" · ")}
              </span>
            )}
            {(stats?.fall_count ?? 0) > 0 && (
              <span className="tile-of" data-testid="ppe-falls">
                {stats?.fall_count} {stats?.fall_count === 1 ? "fall" : "falls"} — reported, not graded
              </span>
            )}
            {model.ppe?.deferred !== false && (
              <span className="tile-of" data-testid="ppe-deferred-note">
                Grading only, alerts not enabled
              </span>
            )}
          </div>
        ) : (
          <p className="muted small">Off</p>
        )}
      </section>
    </div>
  );
}
