import { useCallback, useEffect, useState } from "react";
import type { ChangeEvent, ReactNode } from "react";
import { exportCsvUrl, fireAlertSnapshotUrl, getConfig, getDaily, getFilterOptions, getHistory, type HistoryQuery } from "../api";
import EvidenceModal from "../components/EvidenceModal";
import type { AlertEvent, DailyDay, FilterOptions, HistoryRow } from "../types";
import { alertLabel, alertWho, badgeText, badgeTone, clockTime, historyStatus, istDateTime } from "../utils/format";

const CHART_DAYS = 14;
const PAGE_SIZE = 5;
/** the backend clamps the daily window to 1..31: stay inside it so a
 *  misconfigured value shows a chart instead of a 422. */
const CHART_DAYS_MIN = 1;
const CHART_DAYS_MAX = 31;

const EVENT_TYPE_OPTIONS: { value: string; label: string }[] = [
  { value: "alerts", label: "Alerts only" },
  { value: "", label: "All types" },
  { value: "ENTRY_ALERT", label: "Entry alert" },
  { value: "DWELL_ALERT", label: "Dwell alert" },
  { value: "CAPACITY_ALERT", label: "Capacity alert" },
  { value: "FIRE", label: "Fire" },
  { value: "SMOKE", label: "Smoke" },
  { value: "ZONE_ENTRY", label: "Zone entry" },
  { value: "ZONE_EXIT", label: "Zone exit" },
];

const STATUS_OPTIONS: { value: string; label: string }[] = [
  { value: "", label: "All statuses" },
  { value: "active", label: "Active" },
  { value: "acked", label: "Acknowledged" },
  { value: "false", label: "False alarm" },
];

/** the alert kinds: entry/exit rows stay opt-in so the history is not
 *  flooded by every visit. Fire and smoke are alerts, so they are in. */
const ALERT_TYPES = ["ENTRY_ALERT", "DWELL_ALERT", "CAPACITY_ALERT", "FIRE", "SMOKE"];

interface DraftFilters {
  eventType: string;
  status: string;
  worker: string;
  zone: string;
  dateFrom: string;
  dateTo: string;
}

const EMPTY_DRAFT: DraftFilters = { eventType: "alerts", status: "", worker: "", zone: "", dateFrom: "", dateTo: "" };

function toQuery(draft: DraftFilters, page: number): HistoryQuery {
  return {
    eventTypes: draft.eventType === "alerts" ? [...ALERT_TYPES] : draft.eventType ? [draft.eventType] : [],
    status: draft.status === "active" || draft.status === "acked" || draft.status === "false" ? draft.status : "",
    workerId: draft.worker === "" ? undefined : Number(draft.worker),
    zone: draft.zone === "" ? undefined : draft.zone,
    dateFrom: draft.dateFrom === "" ? undefined : draft.dateFrom,
    dateTo: draft.dateTo === "" ? undefined : draft.dateTo,
    limit: PAGE_SIZE,
    offset: (page - 1) * PAGE_SIZE,
  };
}

/** Page numbers with ellipsis for long histories: 1 … 4 5 6 … 12. */
export function pageWindow(current: number, totalPages: number): (number | "…")[] {
  if (totalPages <= 7) {
    return Array.from({ length: totalPages }, (_, index) => index + 1);
  }
  const pages = new Set([1, 2, current - 1, current, current + 1, totalPages - 1, totalPages]);
  const sorted = [...pages].filter((page) => page >= 1 && page <= totalPages).sort((a, b) => a - b);
  const out: (number | "…")[] = [];
  for (const page of sorted) {
    if (out.length > 0 && page - (out[out.length - 1] as number) > 1) out.push("…");
    out.push(page);
  }
  return out;
}

function dwellCell(row: HistoryRow): string {
  if (row.kind === "fire") return "—";
  const event = row as AlertEvent;
  if (event.event_type === "CAPACITY_ALERT") return "—";
  if (event.dwell_seconds === null || event.dwell_seconds === undefined) return "—";
  return `${event.dwell_seconds.toFixed(1)}s`;
}

/** the one-line fire meta for the evidence dialog: type, severity,
 *  confidence, then when. Same words as the Alerts page fire rows. */
function fireMeta(row: HistoryRow): string {
  const parts = [
    `${String(row.alert_type ?? "").toUpperCase()} · ${row.severity ?? ""}`,
    `conf ${(Number(row.confidence ?? 0)).toFixed(2)}`,
  ];
  const clock = clockTime(row.reported_at ?? row.alert_at);
  if (clock) parts.push(clock);
  const source = row.source_label ? ` · ${row.source_label}` : "";
  return `${parts.join(" · ")}${source}`;
}

/** round the Y axis up to a 1/2/5 step so the top gridline reads human. */
export function niceCeiling(max: number): number {
  if (max <= 0) return 1;
  const power = 10 ** Math.floor(Math.log10(max));
  for (const mult of [1, 2, 5, 10]) {
    if (max <= mult * power) return mult * power;
  }
  return 10 * power;
}

/** Stacked entry (light red) + dwell (red) + capacity (amber) + fire
 *  (scarlet) + smoke (steel blue) bars, one per day. Pure SVG with native
 *  <title> tooltips, so there is no chart dependency to install. */
export function DailyChart({ days }: { days: DailyDay[] }): ReactNode {
  const width = 600;
  const height = 240;
  const left = 36;
  const bottom = 30;
  const top = 12;
  const plotW = width - left - 8;
  const plotH = height - top - bottom;
  const dayTotal = (day: DailyDay) =>
    day.entry + day.dwell + day.capacity + (day.fire ?? 0) + (day.smoke ?? 0);
  const max = Math.max(0, ...days.map(dayTotal));
  const ceil = niceCeiling(max);
  const y = (value: number) => top + plotH - (value / ceil) * plotH;
  const slot = days.length > 0 ? plotW / days.length : 0;
  const barW = Math.max(4, Math.min(28, slot * 0.62));

  const step = niceCeiling(ceil / 4);
  const ticks: number[] = [];
  for (let value = 0; value <= ceil + 1e-9; value += step) ticks.push(value);
  if (ticks[ticks.length - 1] < ceil - 1e-9) ticks.push(ceil);

  return (
    <div className="daily-chart-wrap">
      <svg
        className="daily-chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Alerts per day for the last ${days.length} days`}
        data-testid="daily-chart"
      >
        {ticks.map((tick) => (
          <g key={tick}>
            <line
              x1={left}
              x2={width - 8}
              y1={y(tick)}
              y2={y(tick)}
              className="chart-grid"
            />
            <text x={left - 6} y={y(tick) + 3.5} textAnchor="end" className="chart-tick">
              {tick}
            </text>
          </g>
        ))}
        {days.map((day, index) => {
          const x = left + index * slot + (slot - barW) / 2;
          const fire = day.fire ?? 0;
          const smoke = day.smoke ?? 0;
          const dwellH = (day.dwell / ceil) * plotH;
          const entryH = (day.entry / ceil) * plotH;
          const capH = (day.capacity / ceil) * plotH;
          const fireH = (fire / ceil) * plotH;
          const smokeH = (smoke / ceil) * plotH;
          return (
            <g key={day.date}>
              <title>{`${day.date}: ${day.entry} entry, ${day.dwell} dwell, ${day.capacity} capacity, ${fire} fire, ${smoke} smoke`}</title>
              <rect x={x} y={top} width={barW} height={plotH} fill="transparent" />
              <rect
                x={x}
                y={y(day.dwell)}
                width={barW}
                height={dwellH}
                className="chart-bar-dwell"
                data-testid={`dwell-bar-${day.date}`}
              />
              <rect
                x={x}
                y={y(day.dwell + day.entry)}
                width={barW}
                height={entryH}
                className="chart-bar-entry"
                data-testid={`entry-bar-${day.date}`}
              />
              <rect
                x={x}
                y={y(day.dwell + day.entry + day.capacity)}
                width={barW}
                height={capH}
                className="chart-bar-capacity"
                data-testid={`capacity-bar-${day.date}`}
              />
              <rect
                x={x}
                y={y(day.dwell + day.entry + day.capacity + fire)}
                width={barW}
                height={fireH}
                className="chart-bar-fire"
                data-testid={`fire-bar-${day.date}`}
              />
              <rect
                x={x}
                y={y(day.dwell + day.entry + day.capacity + fire + smoke)}
                width={barW}
                height={smokeH}
                className="chart-bar-smoke"
                data-testid={`smoke-bar-${day.date}`}
              />
              <text
                x={x + barW / 2}
                y={height - 8}
                textAnchor="middle"
                className="chart-day"
              >
                {day.date.slice(5)}
              </text>
            </g>
          );
        })}
      </svg>
      <ul className="legend chart-legend" aria-label="Chart key">
        <li className="legend-item danger">
          <span className="legend-swatch" aria-hidden="true" />
          Dwell
        </li>
        <li className="legend-item entry-legend">
          <span className="legend-swatch" aria-hidden="true" />
          Entry
        </li>
        <li className="legend-item warn">
          <span className="legend-swatch" aria-hidden="true" />
          Capacity
        </li>
        <li className="legend-item fire">
          <span className="legend-swatch" aria-hidden="true" />
          Fire
        </li>
        <li className="legend-item smoke">
          <span className="legend-swatch" aria-hidden="true" />
          Smoke
        </li>
      </ul>
    </div>
  );
}

/** History: the alert chart on top, then the filterable event table with
 *  pagination and CSV export. The chart always shows the last 14 days -
 *  the filters only drive the table and the file. */
export default function HistoryPage(): ReactNode {
  const [days, setDays] = useState<DailyDay[] | null>(null);
  /** chart window in days, from config with a 14-day fallback (see above). */
  const [chartDays, setChartDays] = useState(CHART_DAYS);
  const [chartError, setChartError] = useState<string | null>(null);

  const [options, setOptions] = useState<FilterOptions | null>(null);
  const [draft, setDraft] = useState<DraftFilters>(EMPTY_DRAFT);
  const [applied, setApplied] = useState<DraftFilters>(EMPTY_DRAFT);
  const [page, setPage] = useState(1);
  const [rows, setRows] = useState<HistoryRow[] | null>(null);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [evidence, setEvidence] = useState<HistoryRow | null>(null);

  useEffect(() => {
    let live = true;
    (async () => {
      // the chart window comes from config, falling back to 14 for a
      // backend (or a failed read) that predates the limits block
      let span = CHART_DAYS;
      try {
        const window = (await getConfig()).limits?.history_days;
        if (typeof window === "number" && Number.isFinite(window)) {
          span = Math.min(CHART_DAYS_MAX, Math.max(CHART_DAYS_MIN, Math.floor(window)));
        }
      } catch {
        /* fallback below applies */
      }
      if (!live) return;
      setChartDays(span);
      try {
        const body = await getDaily(span);
        if (!live) return;
        setDays(Array.isArray(body?.days) ? body.days : []);
      } catch (err) {
        if (live) setChartError((err as Error).message);
      }
    })();
    return () => {
      live = false;
    };
  }, []);

  useEffect(() => {
    let live = true;
    getFilterOptions()
      .then((body) => {
        if (live) setOptions(body);
      })
      .catch(() => {
        if (live) setOptions({ workers: [], zones: [] });
      });
    return () => {
      live = false;
    };
  }, []);

  const load = useCallback(async (filters: DraftFilters, nextPage: number) => {
    setLoading(true);
    setError(null);
    try {
      const body = await getHistory(toQuery(filters, nextPage));
      setRows(body.rows ?? []);
      setTotal(body.total ?? 0);
    } catch (err) {
      setRows([]);
      setTotal(0);
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load(applied, page);
  }, [applied, page, load]);

  const apply = () => {
    setApplied({ ...draft });
    setPage(1);
  };
  const clear = () => {
    setDraft(EMPTY_DRAFT);
    setApplied(EMPTY_DRAFT);
    setPage(1);
  };

  const chartTotal = (days ?? []).reduce(
    (sum, day) => sum + day.entry + day.dwell + day.capacity + (day.fire ?? 0) + (day.smoke ?? 0),
    0
  );

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const safePage = Math.min(page, totalPages);
  const start = total === 0 ? 0 : (safePage - 1) * PAGE_SIZE + 1;
  const end = Math.min(safePage * PAGE_SIZE, total);
  const csvUrl = exportCsvUrl(toQuery(applied, 1));

  const evidenceIndex =
    evidence === null
      ? -1
      : (rows ?? []).findIndex(
          (row) => row.kind === evidence.kind && row.id === evidence.id
        );

  const setField = (field: keyof DraftFilters) => (
    event: ChangeEvent<HTMLSelectElement | HTMLInputElement>
  ) => setDraft((current) => ({ ...current, [field]: event.target.value }));

  return (
    <>
      <section className="card history-chart-card" data-testid="history-chart">
        <h2>Alerts per day (last {chartDays} days)</h2>
        <p className="muted small">
          The chart always shows the last {chartDays} days. The filters below drive the
          table and the CSV export only.
        </p>
        {chartError && (
          <div className="inline-note error" role="status">
            Cannot read alert counts: {chartError}
          </div>
        )}
        {days === null && !chartError && <p className="muted small">Loading alert counts…</p>}
        {days !== null && (
          <>
            {chartTotal === 0 && (
              <p className="muted small" data-testid="history-empty">
                No alerts in the last {chartDays} days.
              </p>
            )}
            <DailyChart days={days} />
          </>
        )}
      </section>

      <section className="card history-page" data-testid="history-page">
        <h2>History</h2>
        <form className="history-filters" onSubmit={(e) => e.preventDefault()}>
          <label className="field">
            <span>Event type</span>
            <select
              aria-label="Event type"
              value={draft.eventType}
              onChange={setField("eventType")}
            >
              {EVENT_TYPE_OPTIONS.map((option) => (
                <option key={option.label} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Status</span>
            <select
              aria-label="Status"
              value={draft.status}
              onChange={setField("status")}
            >
              {STATUS_OPTIONS.map((option) => (
                <option key={option.label} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Worker</span>
            <select aria-label="Worker" value={draft.worker} onChange={setField("worker")}>
              <option value="">All workers</option>
              {(options?.workers ?? []).map((id) => (
                <option key={id} value={String(id)}>
                  Worker #{id}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Zone</span>
            <select aria-label="Zone" value={draft.zone} onChange={setField("zone")}>
              <option value="">All zones</option>
              {(options?.zones ?? []).map((zone) => (
                <option key={zone.index} value={String(zone.index)}>
                  {zone.name}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Date from</span>
            <input
              type="date"
              aria-label="Date from"
              value={draft.dateFrom}
              onChange={setField("dateFrom")}
            />
          </label>
          <label className="field">
            <span>Date to</span>
            <input
              type="date"
              aria-label="Date to"
              value={draft.dateTo}
              onChange={setField("dateTo")}
            />
          </label>
          <div className="button-row">
            <button type="button" className="primary" onClick={apply}>
              Apply
            </button>
            <button type="button" onClick={clear}>
              Clear
            </button>
          </div>
        </form>

        {error && (
          <div className="inline-note error" role="status">
            Cannot read events: {error}
          </div>
        )}

        <div className="history-toolbar">
          <span className="muted small" data-testid="history-count">
            Showing {start} to {end} of {total} rows
          </span>
          <a
            className="button-link primary"
            href={csvUrl}
            download={`history-export.csv`}
            data-testid="export-csv"
          >
            Export CSV ({total} rows)
          </a>
        </div>

        {loading || rows === null ? (
          <p className="muted small">Loading events…</p>
        ) : rows.length === 0 ? (
          <p className="muted small" data-testid="history-table-empty">
            No events match these filters.
          </p>
        ) : (
          <>
            <div className="history-table-wrap">
              <table className="history-table" data-testid="history-table">
                <thead>
                  <tr>
                    <th scope="col">Time</th>
                    <th scope="col">Worker</th>
                    <th scope="col">Zone</th>
                    <th scope="col">Source</th>
                    <th scope="col">Event</th>
                    <th scope="col">Dwell</th>
                    <th scope="col">Status</th>
                    <th scope="col">Evidence</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr key={`${row.kind}-${row.id}`}>
                      <td data-testid={`history-time-${row.kind}-${row.id}`}>
                        {istDateTime(row.kind === "fire" ? (row.reported_at ?? row.alert_at) : (row.alert_at ?? row.entered_at))}
                      </td>
                      <td>{row.kind === "fire" ? "—" : alertWho(row)}</td>
                      <td>{row.kind === "fire" ? "—" : ((row as AlertEvent).zone_name ?? "")}</td>
                      <td>{row.kind === "fire" ? (row.source_label ?? "Unknown source") : ""}</td>
                      <td>
                        <span className={`event-badge ${badgeTone(row.event_type)}`}>
                          {badgeText(row.event_type)}
                        </span>{" "}
                        <span className="muted small">
                          {row.kind === "fire"
                            ? `${row.severity ?? ""} · ${(Number(row.confidence ?? 0)).toFixed(2)}`
                            : alertLabel(row)}
                        </span>
                      </td>
                      <td>{dwellCell(row)}</td>
                      <td data-testid={`history-status-${row.kind}-${row.id}`}>
                        {historyStatus(row)}
                      </td>
                      <td>
                        <button
                          type="button"
                          className="evidence-icon-btn"
                          aria-label={
                            row.kind === "fire"
                              ? `View evidence for ${String(row.alert_type ?? "fire")} alert ${row.id}`
                              : `View evidence for event ${row.id}`
                          }
                          title={row.has_snapshot ? "View evidence" : "No snapshot saved"}
                          disabled={!row.has_snapshot}
                          onClick={() => setEvidence(row)}
                        >
                          🔍
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <nav className="pagination" aria-label="History pages">
              <button
                type="button"
                aria-label="Previous page"
                disabled={safePage <= 1}
                onClick={() => setPage(safePage - 1)}
              >
                ← Prev
              </button>
              {pageWindow(safePage, totalPages).map((item, index) =>
                item === "…" ? (
                  <span key={`gap-${index}`} className="page-gap" aria-hidden="true">
                    …
                  </span>
                ) : (
                  <button
                    key={item}
                    type="button"
                    aria-label={`Page ${item}`}
                    aria-current={item === safePage ? "page" : undefined}
                    className={item === safePage ? "page-current" : undefined}
                    onClick={() => setPage(item)}
                  >
                    {item}
                  </button>
                )
              )}
              <button
                type="button"
                aria-label="Next page"
                disabled={safePage >= totalPages}
                onClick={() => setPage(safePage + 1)}
              >
                Next →
              </button>
            </nav>
          </>
        )}

        {evidence && (
          <EvidenceModal
            model={{
              event: evidence,
              hasPrev: evidenceIndex > 0,
              hasNext: evidenceIndex !== -1 && evidenceIndex < (rows ?? []).length - 1,
              onPrev: () => {
                const prev = (rows ?? [])[evidenceIndex - 1];
                if (prev) setEvidence(prev);
              },
              onNext: () => {
                const next = (rows ?? [])[evidenceIndex + 1];
                if (next) setEvidence(next);
              },
              onClose: () => setEvidence(null),
              // Fire rows live in their own id space with their own snapshot
              // route, so the picture and title come from overrides; the
              // History log itself stays read-only (no ack action here).
              imageUrl:
                evidence.kind === "fire"
                  ? evidence.has_snapshot
                    ? fireAlertSnapshotUrl(evidence.id)
                    : null
                  : undefined,
              titleOverride:
                evidence.kind === "fire"
                  ? `${String(evidence.alert_type ?? "").toUpperCase()} · ${evidence.severity ?? ""}`
                  : undefined,
              metaOverride: evidence.kind === "fire" ? fireMeta(evidence) : undefined,
            }}
          />
        )}
      </section>
    </>
  );
}
