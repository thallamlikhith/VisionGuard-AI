import type { ChangeEvent, ReactNode } from "react";

/** one real source: the webcam, or an uploaded clip with its saved count */
export interface SourceOption {
  value: string;
  label: string;
  /** saved zones for this source, null while still loading or failed */
  zones: number | null;
}

export interface PendingSourceSwitch {
  value: string;
  label: string;
}

export interface VideoSourcePickerModel {
  options: SourceOption[];
  /** the current source in option-value terms ("camera" or a clip path) */
  value: string;
  /** edit mode: picking another source would drop unsaved work */
  disabled: boolean;
  onSelect: (value: string) => void;
  uploadDisabled: boolean;
  onPickFile: (event: ChangeEvent<HTMLInputElement>) => void;
  uploadPct: number | null;
  /** a running-feed switch waiting for confirmation */
  pending: PendingSourceSwitch | null;
  onConfirmSwitch: () => void;
  onCancelSwitch: () => void;
  /** the selected source is known to have no saved zones */
  noZoneHint: boolean;
  /** replay box under the picker, shown for a test clip only (the webcam has
   *  no end to replay from). The same stored setting as the Settings Camera
   *  card, so either box moves both. Null hides it. */
  loopVideo?: {
    checked: boolean;
    disabled: boolean;
    onChange: (value: boolean) => void;
  } | null;
}

function zoneSuffix(zones: number | null): string {
  if (zones === null) return "";
  return ` · ${zones} ${zones === 1 ? "zone" : "zones"}`;
}

/** Live Monitor source picker: a compact dropdown over the real sources plus
 *  an upload button reusing the Settings upload flow. Presentation only -
 *  the selected value, the switching and the upload all live in App on the
 *  same stored source setting the Settings page writes. */
export default function VideoSourcePicker({ model }: { model: VideoSourcePickerModel }): ReactNode {
  return (
    <>
      <div className="source-picker-row" data-testid="live-source-picker">
        <select
          aria-label="Video source"
          value={model.value}
          disabled={model.disabled}
          onChange={(event) => model.onSelect(event.target.value)}
          data-testid="live-source-select"
        >
          {model.options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
              {zoneSuffix(option.zones)}
            </option>
          ))}
        </select>
        <label
          className={`upload-button ${model.uploadDisabled ? "disabled" : ""}`}
        >
          Upload video
          <input
            type="file"
            accept="video/mp4,video/x-msvideo,video/quicktime,video/webm,.mp4,.avi,.mov,.mkv,.webm"
            onChange={model.onPickFile}
            disabled={model.uploadDisabled}
            data-testid="live-video-file"
          />
        </label>
      </div>
      {(model.uploadPct !== null || model.pending !== null || model.noZoneHint) && (
        <div className="source-picker-stack">
          {model.uploadPct !== null && (
            <div className="upload-bar" data-testid="live-upload-field">
              <progress max={100} value={model.uploadPct} data-testid="live-upload-progress" />
              <span className="muted small">Uploading… {model.uploadPct}%</span>
            </div>
          )}
          {model.pending !== null && (
            <div
              className="confirm-row"
              role="alertdialog"
              aria-label="Confirm source switch"
              data-testid="live-source-confirm"
            >
              <span className="muted small">
                This stops the current feed and starts the new one: {model.pending.label}?
              </span>
              <div className="button-row">
                <button type="button" onClick={model.onConfirmSwitch}>
                  Switch
                </button>
                <button type="button" onClick={model.onCancelSwitch}>
                  Keep current
                </button>
              </div>
            </div>
          )}
          {model.pending === null && model.noZoneHint && (
            <p className="muted small" data-testid="live-no-zone-hint">
              No zone saved for this clip. Use Edit zones.
            </p>
          )}
        </div>
      )}
      {model.loopVideo !== undefined && model.loopVideo !== null && (
        <label className="check">
          <input
            type="checkbox"
            aria-label="Loop video"
            data-testid="live-loop-video"
            checked={model.loopVideo.checked}
            disabled={model.loopVideo.disabled}
            onChange={(event) => model.loopVideo?.onChange(event.target.checked)}
          />
          <span>Loop video</span>
        </label>
      )}
    </>
  );
}
