import type { AlertEvent, HistoryRow } from "../types";

/** a plain word for each alert kind, so the list never shows a raw enum.
 *  Rows written before the reason existed fall back to the same words via
 *  their event type. */
export function alertLabel(event: AlertEvent): string {
  if (event.event_type === "ENTRY_ALERT") return "Entered danger zone";
  if (event.event_type === "CAPACITY_ALERT") return "Too many people";
  if (event.event_type === "DWELL_ALERT") return "Stayed too long";
  if (event.event_type === "ZONE_ENTRY") return "Zone entry";
  if (event.event_type === "ZONE_EXIT") return "Zone exit";
  if (event.event_type === "FIRE") return "Fire";
  if (event.event_type === "SMOKE") return "Smoke";
  return event.event_type;
}

/** the zone an event is about, from its saved name or its position. */
export function zoneDisplayName(event: AlertEvent): string {
  if (event.zone_name) return event.zone_name;
  if (event.zone_index !== null && event.zone_index !== undefined && event.zone_index >= 0) {
    return `Zone ${event.zone_index + 1}`;
  }
  return "Zone";
}

/** who (or what crowd) an alert row is about. Capacity alerts are about a
 *  crowd, so they name the zone and the count - never "Worker #-1". Other
 *  rows name the worker, falling back to the zone when there is no worker. */
export function alertWho(event: AlertEvent): string {
  if (event.event_type === "CAPACITY_ALERT") {
    const count = event.inside ?? null;
    const people = count === null ? "" : ` · ${count} ${count === 1 ? "person" : "people"}`;
    return `${zoneDisplayName(event)}${people}`;
  }
  if (event.worker_id !== null && event.worker_id !== undefined && event.worker_id >= 0) {
    return `Worker #${event.worker_id}`;
  }
  return zoneDisplayName(event);
}

/** the small grey line under an alert: who, where, when. */
export function alertMeta(event: AlertEvent): string {
  if (event.event_type === "CAPACITY_ALERT") {
    const time = alertTime(event);
    return time ? `${alertWho(event)} · ${time}` : alertWho(event);
  }
  const zone = event.zone_name ? ` · ${event.zone_name}` : "";
  const clock = clockTime(event.entered_at);
  const when = clock ? ` · Entered ${clock}` : "";
  return `${alertWho(event)}${zone}${when}`;
}

/** the evidence dialog title: type, who, zone, when - each its own segment.
 *  A capacity row already names its zone inside the who ("Zone 1 · 2 people"),
 *  so the zone segment is only added when it would say something new. */
export function evidenceTitle(event: AlertEvent): string {
  const clock = clockTime(event.alert_at ?? event.entered_at);
  const when = clock ? ` - ${clock}` : "";
  const zone = event.zone_name ? ` - ${event.zone_name}` : "";
  if (event.event_type === "CAPACITY_ALERT") {
    return `${alertLabel(event)} - ${alertWho(event)}${when}`;
  }
  return `${alertLabel(event)} - ${alertWho(event)}${zone}${when}`;
}

/** "fired 5s ago", "fired 4 min ago", or the date once it is old. A missing
 *  time is left blank rather than shown as 1970. */
export function alertTime(event: AlertEvent): string {
  const stamp = event.alert_at ?? event.entered_at;
  if (!stamp) return "";
  const when = new Date(stamp);
  if (Number.isNaN(when.getTime())) return "";
  const seconds = Math.max(0, Math.round((Date.now() - when.getTime()) / 1000));
  if (seconds < 60) return `fired ${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `fired ${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `fired ${hours} h ago`;
  return when.toLocaleDateString();
}

/** "10:24:18" wall-clock for a stamp, blank when missing/unreadable. */
export function clockTime(stamp: string | null): string {
  if (!stamp) return "";
  const when = new Date(stamp);
  if (Number.isNaN(when.getTime())) return "";
  return when.toLocaleTimeString("en-GB", { hour12: false });
}

/** the card badge for one event kind: entry and dwell red, capacity amber,
 *  plain entries blue, exits grey. Fire and smoke borrow their burned-in
 *  box colours (scarlet / steel blue), never the verdict colours. */
export function badgeTone(eventType: string): string {
  if (eventType === "ENTRY_ALERT") return "badge-entry";
  if (eventType === "DWELL_ALERT") return "badge-dwell";
  if (eventType === "CAPACITY_ALERT") return "badge-capacity";
  if (eventType === "ZONE_ENTRY") return "badge-zone-entry";
  if (eventType === "ZONE_EXIT") return "badge-zone-exit";
  if (eventType === "FIRE") return "badge-fire";
  if (eventType === "SMOKE") return "badge-smoke";
  return "badge-other";
}

/** the card badge words for one event kind. */
export function badgeText(eventType: string): string {
  if (eventType === "ENTRY_ALERT") return "ENTRY ALERT";
  if (eventType === "DWELL_ALERT") return "DWELL ALERT";
  if (eventType === "CAPACITY_ALERT") return "CAPACITY ALERT";
  if (eventType === "ZONE_ENTRY") return "ZONE ENTRY";
  if (eventType === "ZONE_EXIT") return "ZONE EXIT";
  if (eventType === "FIRE") return "FIRE";
  if (eventType === "SMOKE") return "SMOKE";
  return eventType;
}

/** the History Status cell in the row's own words: a false-alarm flag reads
 *  as what it is (not as acknowledged), everything else is Active until
 *  somebody acks it. Never a raw boolean. */
export function historyStatus(row: HistoryRow): string {
  if (row.kind === "fire" && row.false_alarm) return "False alarm";
  if (row.acknowledged) return "Acknowledged";
  return "Active";
}

/** "1,051" for the acknowledge-all confirm: big backlogs stay readable. */
export function formatCount(value: number): string {
  return Number(value).toLocaleString("en-US");
}

/** "YYYY-MM-DD HH:MM:SS" in the plant's local time (IST), blank when the
 *  stamp is missing or unreadable. */
export function istDateTime(stamp: string | null): string {
  if (!stamp) return "";
  const when = new Date(stamp);
  if (Number.isNaN(when.getTime())) return "";
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Kolkata",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).formatToParts(when);
  const get = (type: string) => parts.find((part) => part.type === type)?.value ?? "";
  return `${get("year")}-${get("month")}-${get("day")} ${get("hour")}:${get("minute")}:${get("second")}`;
}
