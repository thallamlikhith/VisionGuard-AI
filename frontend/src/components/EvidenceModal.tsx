import { useState, type ReactNode } from "react";
import { alertSnapshotUrl } from "../api";
import type { AlertEvent } from "../types";
import { alertTime, evidenceTitle } from "../utils/format";

export interface EvidenceDetail {
  label: string;
  value: string;
}

export interface EvidenceModalModel {
  event: AlertEvent;
  hasPrev: boolean;
  hasNext: boolean;
  onPrev: () => void;
  onNext: () => void;
  onClose: () => void;
  /** fire/smoke rows live in their own id space with their own snapshot
   *  route: when set, the picture, title and meta come from these instead
   *  of the zone event. Null imageUrl keeps the "no picture" message. */
  imageUrl?: string | null;
  titleOverride?: string;
  metaOverride?: string;
  /** key/value detail rows (Alert Details panel). When absent the legacy
   *  one-line meta renders instead, so older callers look unchanged. */
  details?: EvidenceDetail[] | null;
  /** acknowledge this alert from the modal (also flips the row behind it).
   *  Absent for callers without an ack action (History). */
  onAcknowledge?: (() => void) | null;
  acknowledged?: boolean;
  /** export this fire/smoke alert as a hard negative (also acks it).
   *  Absent for zone rows, which have no such action. */
  onFalseAlarm?: (() => void) | null;
  falseAlarmBusy?: boolean;
  falseAlarmMarked?: boolean;
}

function dwellText(event: AlertEvent): string | null {
  if (event.event_type === "CAPACITY_ALERT") return null;
  if (event.dwell_seconds === null || event.dwell_seconds === undefined) return null;
  return `${event.dwell_seconds.toFixed(1)}s`;
}

/** The alert evidence dialog, shared by the Alerts and History pages: title,
 *  snapshot, dwell line, close, and stepping through the list it was opened
 *  from. Fire/smoke alerts reuse the same dialog through the image/title/meta
 *  overrides (their own snapshot route and id space). */
export default function EvidenceModal({ model }: { model: EvidenceModalModel }): ReactNode {
  const [confirmingFalseAlarm, setConfirmingFalseAlarm] = useState(false);
  const dwell = dwellText(model.event);
  const image = model.imageUrl !== undefined
    ? model.imageUrl
    : model.event.has_snapshot
      ? alertSnapshotUrl(model.event.id)
      : null;
  const title = model.titleOverride ?? evidenceTitle(model.event);
  const meta = model.metaOverride ?? (
    <>
      {alertTime(model.event)}
      {model.event.zone_name ? ` · ${model.event.zone_name}` : ""}
      {dwell !== null ? ` · Dwell ${dwell}` : ""}
    </>
  );
  const [imageFailed, setImageFailed] = useState(false);
  return (
    <div
      className="modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label="Alert evidence"
      onClick={model.onClose}
    >
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <strong>{title}</strong>
          <button
            type="button"
            className="modal-close"
            aria-label="Close evidence"
            onClick={model.onClose}
          >
            ×
          </button>
        </div>
        {image !== null && !imageFailed ? (
          <img
            className="modal-image"
            src={image}
            alt="Alert evidence"
            onError={() => setImageFailed(true)}
          />
        ) : image !== null ? (
          <div className="modal-image-missing" data-testid="modal-snapshot-missing" role="status">
            Snapshot unavailable — the file was pruned or deleted.
          </div>
        ) : (
          <p className="muted small">No picture was saved for this alert.</p>
        )}
        {model.details != null && model.details.length > 0 ? (
          <dl className="modal-details" data-testid="modal-details">
            {model.details.map((row) => (
              <div className="modal-detail-row" key={row.label}>
                <dt>{row.label}</dt>
                <dd>{row.value}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <div className="modal-meta">
            {meta}
          </div>
        )}
        {(model.onAcknowledge != null || model.onFalseAlarm != null) && (
          <div className="modal-actions">
            {model.onAcknowledge != null && !model.acknowledged && (
              <button type="button" className="danger-btn" onClick={model.onAcknowledge}>
                Acknowledge
              </button>
            )}
            {model.onAcknowledge != null && model.acknowledged === true && (
              <button type="button" className="alert-ack-state" disabled>
                ✓ Acknowledged
              </button>
            )}
          </div>
        )}
        {model.falseAlarmMarked === true ? (
          <p className="muted small" data-testid="modal-false-alarm-marked">
            Marked as false alarm ✓
          </p>
        ) : model.onFalseAlarm != null ? (
          confirmingFalseAlarm ? (
            <span className="confirm-inline" role="alertdialog" aria-label="Confirm false alarm">
              <button
                type="button"
                className="danger-btn"
                disabled={model.falseAlarmBusy === true}
                onClick={() => {
                  setConfirmingFalseAlarm(false);
                  model.onFalseAlarm?.();
                }}
              >
                {model.falseAlarmBusy === true ? "Exporting…" : "Confirm"}
              </button>
              <button type="button" onClick={() => setConfirmingFalseAlarm(false)}>
                Cancel
              </button>
            </span>
          ) : (
            <button
              type="button"
              className="alert-false-alarm"
              title="Export the clean frame for retraining"
              onClick={() => setConfirmingFalseAlarm(true)}
            >
              False alarm
            </button>
          )
        ) : null}
        <div className="modal-nav">
          <button
            type="button"
            aria-label="Previous alert"
            disabled={!model.hasPrev}
            onClick={model.onPrev}
          >
            ← Prev
          </button>
          <button
            type="button"
            aria-label="Next alert"
            disabled={!model.hasNext}
            onClick={model.onNext}
          >
            Next →
          </button>
        </div>
      </div>
    </div>
  );
}
