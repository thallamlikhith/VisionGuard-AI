import { useState, type ReactNode } from "react";
import { alertSnapshotUrl, fireAlertSnapshotUrl } from "../api";
import EvidenceModal from "../components/EvidenceModal";
import type { AlertEvent, FireAlert } from "../types";
import { alertLabel, alertWho, badgeText, badgeTone, clockTime, formatCount, zoneDisplayName } from "../utils/format";

export interface AlertsModel {
  alertTab: "active" | "acknowledged";
  activeAlerts: AlertEvent[];
  /** true unacknowledged alerts on the backend - the tab reads this, not just
   *  the loaded rows. Zone alerts only; fire/smoke has its own counts below,
   *  so these numbers behave exactly as before. */
  activeTotal: number;
  clearedAlerts: AlertEvent[];
  /** true acknowledged alerts on the backend - the tab reads this, not just
   *  the loaded rows, so it keeps counting past the /events page size. */
  clearedTotal: number;
  /** fire/smoke rows for this tab (same Active/Acknowledged split, newest
   *  first), with their own backend totals. Rendered under their own
   *  subhead so zone rows, counts and order are untouched. Optional,
   *  because callers from before the fire section do not send it. */
  fireAlerts?: FireAlert[];
  fireActiveTotal?: number;
  fireClearedTotal?: number;
  alertsError: string | null;
  evidence: AlertEvent | null;
  ackingAll: boolean;
  evidenceHasPrev: boolean;
  evidenceHasNext: boolean;
  onTab: (tab: "active" | "acknowledged") => void;
  onAcknowledge: (id: number) => void;
  onAcknowledgeAll: () => void;
  onAcknowledgeFire?: (id: number) => void;
  /** running machine name for the zone/source cells. Optional, because
   *  callers from before it fall back to the row's own machine id. */
  machineName?: string;
  /** prebuilt Alert Details rows for the open zone modal, and an ack action
   *  for it. Optional; without them the modal renders as before. */
  evidenceDetails?: { label: string; value: string }[] | null;
  onAcknowledgeEvidence?: (() => void) | null;
  /** same pair for the open fire modal. */
  fireEvidenceDetails?: { label: string; value: string }[] | null;
  onAcknowledgeFireEvidence?: (() => void) | null;
  onEvidence: (event: AlertEvent | null) => void;
  onEvidenceStep: (direction: 1 | -1) => void;
  fireEvidence?: FireAlert | null;
  fireEvidenceHasPrev?: boolean;
  fireEvidenceHasNext?: boolean;
  onFireEvidence?: (alert: FireAlert | null) => void;
  onFireEvidenceStep?: (direction: 1 | -1) => void;
  /** export one fire/smoke alert as a hard negative (also acks it), and the
   *  id currently being exported. Optional, like the rest of the fire block. */
  onFalseAlarm?: (id: number) => void;
  falseAlarming?: number | null;
}

function ackedClock(event: AlertEvent): string {
  const clock = clockTime(event.acknowledged_at);
  return clock ? `Acknowledged ${clock}` : "Acknowledged";
}

/** The snapshot thumbnail: a grey block while the picture loads, a "no
 *  snapshot" icon when the load fails, nothing at all when no picture was
 *  ever saved for the alert. `imageUrl` overrides the zone-alert URL, which
 *  is how fire/smoke rows (a different id space) reuse the same thumbnail. */
function Thumb({ event, label, imageUrl }: {
  event: Pick<AlertEvent, "id" | "has_snapshot">;
  label: string;
  imageUrl?: string;
}): ReactNode {
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);
  const url = imageUrl ?? (event.has_snapshot ? alertSnapshotUrl(event.id) : null);
  if (url === null) {
    return <span className="alert-thumb empty" aria-hidden="true" />;
  }
  if (failed) {
    return (
      <span className="alert-thumb failed" role="img" aria-label="No snapshot">
        ◧
      </span>
    );
  }
  return (
    <>
      {!loaded && <span className="alert-thumb loading" aria-hidden="true" />}
      <img
        className={`alert-thumb${loaded ? "" : " thumb-hidden"}`}
        src={url}
        alt={label}
        onLoad={() => setLoaded(true)}
        onError={() => setFailed(true)}
      />
    </>
  );
}

function fireMeta(alert: FireAlert): string {
  const parts = [
    `${alert.alert_type.toUpperCase()} · ${alert.severity}`,
    `conf ${alert.confidence.toFixed(2)}`,
  ];
  if (alert.captured_at !== null && alert.captured_at !== undefined) {
    parts.push(`video ${alert.captured_at.toFixed(1)}s`);
  }
  const clock = clockTime(alert.reported_at);
  if (clock) parts.push(clock);
  return parts.join(" · ");
}

function fireBadgeTone(alert: FireAlert): string {
  return alert.alert_type === "fire" ? "danger" : "warn";
}

/** Zone cell machine line: the running machine's name when given, else the
 *  id the row itself carries. */
function machineNameOf(machineName: string | undefined, event: AlertEvent): string {
  if (machineName) return machineName;
  if (event.machine_id !== null && event.machine_id !== undefined) {
    return String(event.machine_id);
  }
  return "";
}

function dwellText(event: AlertEvent): string | null {
  if (event.event_type === "CAPACITY_ALERT") return null;
  if (event.dwell_seconds === null || event.dwell_seconds === undefined) return null;
  return `${event.dwell_seconds.toFixed(1)}s`;
}

export default function AlertsPage({ model }: { model: AlertsModel }): ReactNode {
  const [confirmingAll, setConfirmingAll] = useState(false);
  const [confirmingFalseAlarm, setConfirmingFalseAlarm] = useState<number | null>(null);
  const falseAlarming = model.falseAlarming ?? null;
  const active = model.alertTab === "active";
  const shown = active ? model.activeAlerts : model.clearedAlerts;
  // fire/smoke rows keep their own data and callbacks; every fallback keeps
  // callers from before the fire section rendering exactly as they did.
  const fireRows = model.fireAlerts ?? [];
  const fireTotal = active
    ? (model.fireActiveTotal ?? fireRows.length)
    : (model.fireClearedTotal ?? fireRows.length);
  const onAckFire = model.onAcknowledgeFire ?? (() => {});
  const fireEv = model.fireEvidence ?? null;
  const onFireEv = model.onFireEvidence ?? (() => {});
  const onFireEvStep = model.onFireEvidenceStep ?? (() => {});
  const fireEvPrev = model.fireEvidenceHasPrev ?? false;
  const fireEvNext = model.fireEvidenceHasNext ?? false;
  return (
    <section className="card alerts-page">
      <div className="alerts-page-head">
        <h2>Alerts</h2>
        <div className="seg alert-tabs" role="tablist" aria-label="Alert filter">
          <button
            type="button"
            role="tab"
            aria-selected={active}
            className={`seg-item ${active ? "active" : ""}`}
            onClick={() => model.onTab("active")}
          >
            Active ({model.activeTotal})
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={!active}
            className={`seg-item ${!active ? "active" : ""}`}
            onClick={() => model.onTab("acknowledged")}
          >
            Acknowledged ({model.clearedTotal})
          </button>
        </div>
      </div>
      {model.alertsError && (
        <div className="inline-note error" role="status">
          Cannot read alerts: {model.alertsError}
        </div>
      )}
      {active && model.activeTotal > 0 && (
        <div className="alerts-actions-bar">
          {confirmingAll ? (
            <span className="confirm-inline" role="alertdialog" aria-label="Confirm acknowledge all">
              <span className="muted small">
                Acknowledge {formatCount(model.activeTotal)} alert
                {model.activeTotal === 1 ? "" : "s"}?
              </span>
              <button
                type="button"
                className="danger-btn"
                disabled={model.ackingAll}
                onClick={() => {
                  setConfirmingAll(false);
                  model.onAcknowledgeAll();
                }}
              >
                {model.ackingAll ? "Acknowledging…" : "Confirm"}
              </button>
              <button type="button" disabled={model.ackingAll} onClick={() => setConfirmingAll(false)}>
                Cancel
              </button>
            </span>
          ) : (
            <button type="button" onClick={() => setConfirmingAll(true)}>
              Acknowledge all
            </button>
          )}
        </div>
      )}
      {shown.length === 0 && fireRows.length === 0 ? (
        <p className="alerts-empty" data-testid="alerts-empty">
          {active ? (
            <>
              <span className="alerts-empty-mark" aria-hidden="true">
                ✓
              </span>{" "}
              No active alerts
            </>
          ) : (
            "Nothing acknowledged yet"
          )}
        </p>
      ) : (
        <>
        {shown.length > 0 && (
        <>
        <h3 className="alert-subhead" data-testid="zone-subhead">
          People in Danger Zones ({active ? model.activeTotal : model.clearedTotal})
        </h3>
        <table className="alert-table" data-testid="alert-list">
          <thead>
            <tr>
              <th scope="col">Snapshot</th>
              <th scope="col">Type</th>
              <th scope="col">Zone</th>
              <th scope="col">Worker</th>
              <th scope="col">Time</th>
              <th scope="col">Action</th>
            </tr>
          </thead>
          <tbody>
          {shown.map((event) => {
            const dwell = dwellText(event);
            const enteredClock = clockTime(event.entered_at ?? event.alert_at);
            return (
              <tr key={event.id} className={`alert-row ${event.acknowledged ? "acked" : ""}`}>
                <td>
                  <button
                    type="button"
                    className="alert-thumb-btn"
                    title="View evidence"
                    aria-label={`View evidence for alert ${event.id}`}
                    onClick={() => model.onEvidence(event)}
                  >
                    <Thumb event={event} label={`Alert ${event.id}`} />
                  </button>
                </td>
                <td>
                  <span className={`event-badge ${badgeTone(event.event_type)}`}>
                    {badgeText(event.event_type)}
                  </span>
                  <div className="alert-kind">{alertLabel(event)}</div>
                </td>
                <td>
                  <div>{zoneDisplayName(event)}</div>
                  <div className="muted small">{machineNameOf(model.machineName, event)}</div>
                </td>
                <td>{alertWho(event)}</td>
                <td>
                  <div>{enteredClock ? `Entered ${enteredClock}` : ""}</div>
                  {dwell !== null && (
                    <div className="muted small" data-testid={`alert-duration-${event.id}`}>
                      {dwell}
                    </div>
                  )}
                  {event.acknowledged && (
                    <div className="muted small" data-testid={`acked-line-${event.id}`}>
                      {ackedClock(event)}
                    </div>
                  )}
                </td>
                <td>
                  <div className="alert-actions">
                    <button
                      type="button"
                      className="alert-evidence"
                      onClick={() => model.onEvidence(event)}
                    >
                      View Evidence
                    </button>
                    {event.acknowledged ? (
                      <button type="button" className="alert-ack-state" disabled title={ackedClock(event)}>
                        ✓ Acknowledged
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="alert-ack danger-btn"
                        onClick={() => model.onAcknowledge(event.id)}
                      >
                        Acknowledge
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            );
          })}
          </tbody>
        </table>
        </>
        )}
        {!active && model.clearedTotal > shown.length && (
          <p className="muted small" data-testid="acked-more">
            Showing {shown.length} of {model.clearedTotal} - see History for the rest.
          </p>
        )}
        {active && model.activeTotal > shown.length && (
          <p className="muted small" data-testid="active-more">
            Showing {shown.length} of {model.activeTotal} - acknowledge some to work through the backlog.
          </p>
        )}
        </>
      )}

      {fireRows.length > 0 && (
        <>
          <h3 className="fire-subhead" data-testid="fire-subhead">
            Fire &amp; smoke ({fireTotal})
          </h3>
          <table className="alert-table" data-testid="fire-alert-list">
            <thead>
              <tr>
                <th scope="col">Snapshot</th>
                <th scope="col">Type</th>
                <th scope="col">Source</th>
                <th scope="col">Time</th>
                <th scope="col">Action</th>
              </tr>
            </thead>
            <tbody>
            {fireRows.map((alert) => {
              const clock = clockTime(alert.reported_at);
              return (
              <tr key={`fire-${alert.id}`} className={`alert-row ${alert.acknowledged ? "acked" : ""}`}>
                <td>
                  <button
                    type="button"
                    className="alert-thumb-btn"
                    title="View evidence"
                    aria-label={`View evidence for ${alert.alert_type} alert ${alert.id}`}
                    onClick={() => onFireEv(alert)}
                  >
                    <Thumb
                      event={alert}
                      label={`${alert.alert_type} alert ${alert.id}`}
                      imageUrl={alert.has_snapshot ? fireAlertSnapshotUrl(alert.id) : undefined}
                    />
                  </button>
                </td>
                <td>
                  <span className={`event-badge ${fireBadgeTone(alert)}`}>
                    {alert.alert_type === "fire" ? "FIRE" : "SMOKE"} {alert.confidence.toFixed(2)}
                  </span>
                  <div className="alert-kind">{alert.severity}</div>
                  <div className="alert-meta">{fireMeta(alert)}</div>
                </td>
                <td>{alert.source_label ?? "Unknown source"}</td>
                <td>
                  <div>{clock}</div>
                  {alert.acknowledged && (
                    <div className="muted small" data-testid={`fire-acked-line-${alert.id}`}>
                      {(() => {
                        const acked = clockTime(alert.acknowledged_at);
                        return acked ? `Acknowledged ${acked}` : "Acknowledged";
                      })()}
                    </div>
                  )}
                </td>
                <td>
                  <div className="alert-actions">
                    <button
                      type="button"
                      className="alert-evidence"
                      onClick={() => onFireEv(alert)}
                    >
                      View Evidence
                    </button>
                    {alert.acknowledged ? (
                      <button type="button" className="alert-ack-state" disabled title="Acknowledged">
                        ✓ Acknowledged
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="alert-ack danger-btn"
                        onClick={() => onAckFire(alert.id)}
                      >
                        Acknowledge
                      </button>
                    )}
                    {alert.false_alarm ? (
                      <span className="alert-acked-line" data-testid={`fire-falsealarm-${alert.id}`}>
                        Marked as false alarm ✓
                      </span>
                    ) : confirmingFalseAlarm === alert.id ? (
                      <span className="confirm-inline" role="alertdialog" aria-label="Confirm false alarm">
                        <button
                          type="button"
                          className="danger-btn"
                          disabled={falseAlarming === alert.id}
                          onClick={() => {
                            setConfirmingFalseAlarm(null);
                            model.onFalseAlarm?.(alert.id);
                          }}
                        >
                          {falseAlarming === alert.id ? "Exporting…" : "Confirm"}
                        </button>
                        <button type="button" onClick={() => setConfirmingFalseAlarm(null)}>
                          Cancel
                        </button>
                      </span>
                    ) : (
                      <button
                        type="button"
                        className="alert-false-alarm"
                        title="Export the clean frame for retraining"
                        onClick={() => setConfirmingFalseAlarm(alert.id)}
                      >
                        False alarm
                      </button>
                    )}
                  </div>
                </td>
              </tr>
              );
            })}
            </tbody>
          </table>
        </>
      )}

      {model.evidence && (
        <EvidenceModal
          model={{
            event: model.evidence,
            hasPrev: model.evidenceHasPrev,
            hasNext: model.evidenceHasNext,
            onPrev: () => model.onEvidenceStep(-1),
            onNext: () => model.onEvidenceStep(1),
            onClose: () => model.onEvidence(null),
            details: model.evidenceDetails ?? null,
            onAcknowledge: model.onAcknowledgeEvidence ?? null,
            acknowledged: model.evidence.acknowledged,
          }}
        />
      )}

      {fireEv && (
        <EvidenceModal
          model={{
            event: {
              id: fireEv.id,
              worker_id: null,
              machine_id: null,
              event_type: fireEv.alert_type === "fire" ? "FIRE" : "SMOKE",
              reason: null,
              inside: null,
              entered_at: null,
              alert_at: fireEv.reported_at,
              dwell_seconds: null,
              zone_index: null,
              zone_name: null,
              has_snapshot: fireEv.has_snapshot,
              acknowledged: fireEv.acknowledged,
              acknowledged_at: fireEv.acknowledged_at,
            },
            hasPrev: fireEvPrev,
            hasNext: fireEvNext,
            onPrev: () => onFireEvStep(-1),
            onNext: () => onFireEvStep(1),
            onClose: () => onFireEv(null),
            imageUrl: fireEv.has_snapshot
              ? fireAlertSnapshotUrl(fireEv.id)
              : null,
            titleOverride:
              `${fireEv.alert_type === "fire" ? "FIRE" : "SMOKE"} · ` +
              fireEv.severity,
            metaOverride: fireMeta(fireEv),
            onFalseAlarm: model.onFalseAlarm ? () => model.onFalseAlarm?.(fireEv.id) : null,
            falseAlarmBusy: (model.falseAlarming ?? null) === fireEv.id,
            falseAlarmMarked: fireEv.false_alarm === true,
            details: model.fireEvidenceDetails ?? null,
            onAcknowledge: model.onAcknowledgeFireEvidence ?? null,
            acknowledged: fireEv.acknowledged,
          }}
        />
      )}
    </section>
  );
}
