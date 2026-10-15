"""Tiny SQLite layer for danger zone events.

Three tables are not needed - one table and plain helper functions is enough.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

from zone.danger_zone import CAPACITY_ALERT, DWELL_ALERT, ENTRY_ALERT

# The plant runs on Indian time, whatever the server's own clock says. All
# history dates - filtering, grouping and CSV export - are read in IST so a
# night shift never lands on the wrong day because the box sits in UTC.
IST = timezone(timedelta(hours=5, minutes=30))

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id        INTEGER NOT NULL,
    machine_id       TEXT    NOT NULL,
    event_type       TEXT    NOT NULL,
    entered_at       REAL,
    alert_at         REAL,
    dwell_seconds    REAL    NOT NULL DEFAULT 0,
    zone_index       INTEGER,
    zone_name        TEXT,
    reason           TEXT,
    inside           INTEGER,
    snapshot_path    TEXT,
    acknowledged     INTEGER NOT NULL DEFAULT 0,
    acknowledged_at  REAL
);
"""

CREATE_INDEX = "CREATE INDEX IF NOT EXISTS idx_events_id_desc ON events (id DESC);"

# Columns added after the first release. A database that already has the table
# gets them through ALTER TABLE, so an existing install keeps its events.
EVENT_COLUMNS = {
    "zone_index": "INTEGER",
    "zone_name": "TEXT",
    "reason": "TEXT",
    "inside": "INTEGER",
    "snapshot_path": "TEXT",
    "acknowledged": "INTEGER NOT NULL DEFAULT 0",
    "acknowledged_at": "REAL",
}

SETTINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
"""

# Fire/smoke alerts live in their own table (not the zone events table), so
# zone rows, counts, history and CSV exports behave exactly as before.
FIRE_SCHEMA = """
CREATE TABLE IF NOT EXISTS fire_alerts (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_type       TEXT    NOT NULL,
    severity         TEXT    NOT NULL,
    confidence       REAL    NOT NULL DEFAULT 0,
    bbox_xyxy        TEXT,
    captured_at      REAL,
    reported_at      REAL,
    snapshot_path    TEXT,
    clean_snapshot_path TEXT,
    acknowledged     INTEGER NOT NULL DEFAULT 0,
    acknowledged_at  REAL,
    false_alarm      INTEGER NOT NULL DEFAULT 0,
    false_alarm_at   REAL,
    source_label     TEXT
);
"""

# Columns added after the fire table first shipped (Phase 3). A database
# that already has fire_alerts rows gets them through ALTER TABLE, so no
# alert is ever lost to an upgrade.
FIRE_COLUMNS = {
    "clean_snapshot_path": "TEXT",
    "false_alarm": "INTEGER NOT NULL DEFAULT 0",
    "false_alarm_at": "REAL",
    "source_label": "TEXT",
}

CREATE_FIRE_INDEX = "CREATE INDEX IF NOT EXISTS idx_fire_alerts_id_desc ON fire_alerts (id DESC);"


class DatabaseError(Exception):
    """Raised when SQLite cannot be used."""


class Database:
    def __init__(self, path: str = "data/trinetra.db") -> None:
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        try:
            if parent:
                os.makedirs(parent, exist_ok=True)
        except OSError as exc:
            raise DatabaseError(f"could not create folder for {path}: {exc}") from exc
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        try:
            return sqlite3.connect(self.path)
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not open {self.path}: {exc}") from exc

    def _init_schema(self) -> None:
        try:
            with self._connect() as conn:
                conn.execute(SCHEMA)
                self._migrate_events(conn)
                conn.execute(CREATE_INDEX)
                conn.execute(SETTINGS_SCHEMA)
                conn.execute(FIRE_SCHEMA)
                self._migrate_fire_alerts(conn)
                conn.execute(CREATE_FIRE_INDEX)
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not create schema in {self.path}: {exc}") from exc

    @staticmethod
    def _migrate_events(conn: sqlite3.Connection) -> None:
        """Add any event column a database from an earlier version is missing.

        Runs on every startup but is a no-op once the columns are there, so a
        fresh install and an upgraded one end up with the same table.
        """
        existing = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
        for name, declaration in EVENT_COLUMNS.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE events ADD COLUMN {name} {declaration}")

    @staticmethod
    def _migrate_fire_alerts(conn: sqlite3.Connection) -> None:
        """Add any fire_alerts column a Phase 2 database is missing.

        Runs on every startup but is a no-op once the columns are there, so
        existing rows (and their snapshots) survive the upgrade untouched.
        """
        existing = {row[1] for row in conn.execute("PRAGMA table_info(fire_alerts)")}
        for name, declaration in FIRE_COLUMNS.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE fire_alerts ADD COLUMN {name} {declaration}")

    # -- write -----------------------------------------------------------

    def insert_event(self, event) -> Optional[int]:
        """Store one zone event dict / Event object. Returns the new row id."""
        data = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        zone_index = data.get("zone_index")
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    "INSERT INTO events (worker_id, machine_id, event_type, entered_at, alert_at,"
                    " dwell_seconds, zone_index, zone_name, reason, inside)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        int(data["worker_id"]),
                        str(data["machine_id"]),
                        str(data["event_type"]),
                        data.get("entered_at"),
                        data.get("alert_at"),
                        float(data.get("dwell_seconds") or 0.0),
                        None if zone_index is None else int(zone_index),
                        data.get("zone_name"),
                        data.get("reason"),
                        data.get("inside"),
                    ),
                )
                return cursor.lastrowid
        except (sqlite3.Error, KeyError, TypeError, ValueError) as exc:
            raise DatabaseError(f"could not store event: {exc}") from exc

    def insert_events(self, events) -> int:
        return sum(1 for event in events if self.insert_event(event))

    # -- read ------------------------------------------------------------

    @staticmethod
    def _day_bounds_ist(day: str) -> tuple[float, float]:
        """Midnight-to-midnight epoch bounds for a plant-local YYYY-MM-DD."""
        try:
            morning = datetime.strptime(day.strip(), "%Y-%m-%d").replace(tzinfo=IST)
        except (ValueError, AttributeError) as exc:
            raise DatabaseError(f"bad date {day!r}: expected YYYY-MM-DD") from exc
        return morning.timestamp(), (morning + timedelta(days=1)).timestamp()

    @classmethod
    def _event_where(
        cls,
        event_type: Optional[str] = None,
        event_types: Optional[list[str]] = None,
        acknowledged: Optional[bool] = None,
        worker_id: Optional[int] = None,
        zone: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> tuple[list[str], list]:
        """The WHERE clause every event listing, count and export shares, so a
        filter can never mean one thing in the table and another in the CSV.
        Every value is bound as a parameter, never pasted into the SQL."""
        types: list[str] = []
        if event_type:
            types.append(event_type)
        if event_types:
            types.extend(event_types)

        where: list[str] = []
        params: list = []
        if types:
            placeholders = ",".join("?" for _ in types)
            where.append(f"event_type IN ({placeholders})")
            params.extend(types)
        if acknowledged is not None:
            where.append("acknowledged = ?")
            params.append(1 if acknowledged else 0)
        if worker_id is not None:
            where.append("worker_id = ?")
            params.append(int(worker_id))
        if zone is not None and str(zone) != "":
            if str(zone).strip().lstrip("-").isdigit():
                where.append("zone_index = ?")
                params.append(int(str(zone).strip()))
            else:
                where.append("zone_name = ?")
                params.append(str(zone))
        moment = "COALESCE(alert_at, entered_at)"
        if date_from:
            start, _ = cls._day_bounds_ist(date_from)
            where.append(f"{moment} >= ?")
            params.append(start)
        if date_to:
            _, end = cls._day_bounds_ist(date_to)
            where.append(f"{moment} < ?")
            params.append(end)
        return where, params

    def list_events(
        self,
        limit: int = 100,
        event_type: Optional[str] = None,
        event_types: Optional[list[str]] = None,
        acknowledged: Optional[bool] = None,
        worker_id: Optional[int] = None,
        zone: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        offset: int = 0,
    ) -> list[dict]:
        """Newest first. Every filter is bound as a parameter, never pasted
        into the SQL, so a caller-supplied type can only ever be a value."""
        where, params = self._event_where(
            event_type=event_type,
            event_types=event_types,
            acknowledged=acknowledged,
            worker_id=worker_id,
            zone=zone,
            date_from=date_from,
            date_to=date_to,
        )

        sql = "SELECT * FROM events"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([int(limit), max(0, int(offset))])

        try:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                return [dict(row) for row in conn.execute(sql, params).fetchall()]
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not read events: {exc}") from exc

    def count_events(
        self,
        event_type: Optional[str] = None,
        event_types: Optional[list[str]] = None,
        acknowledged: Optional[bool] = None,
        worker_id: Optional[int] = None,
        zone: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> int:
        """How many rows the current filters match, ignoring paging."""
        where, params = self._event_where(
            event_type=event_type,
            event_types=event_types,
            acknowledged=acknowledged,
            worker_id=worker_id,
            zone=zone,
            date_from=date_from,
            date_to=date_to,
        )
        sql = "SELECT COUNT(*) FROM events"
        if where:
            sql += " WHERE " + " AND ".join(where)
        try:
            with self._connect() as conn:
                row = conn.execute(sql, params).fetchone()
                return int(row[0]) if row else 0
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not count events: {exc}") from exc

    def filter_options(self) -> dict:
        """Distinct workers and zones for the history dropdowns. The -1 worker
        id is the capacity sentinel, not a person, so it is left out; zones
        come back index-aligned with the latest name each index was saved
        under."""
        try:
            with self._connect() as conn:
                worker_rows = conn.execute(
                    "SELECT DISTINCT worker_id FROM events"
                    " WHERE worker_id IS NOT NULL AND worker_id >= 0"
                    " ORDER BY worker_id"
                ).fetchall()
                zone_rows = conn.execute(
                    "SELECT zone_index, zone_name FROM events"
                    " WHERE zone_index IS NOT NULL ORDER BY id DESC"
                ).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not read filter options: {exc}") from exc
        zones: list[dict] = []
        seen: set = set()
        for index, name in zone_rows:
            if index in seen:
                continue
            seen.add(index)
            zones.append({"index": index, "name": name or f"Zone {index + 1}"})
        zones.sort(key=lambda zone: zone["index"])
        return {
            "workers": [row[0] for row in worker_rows],
            "zones": zones,
        }

    def alert_incidents_by_day(
        self,
        days: int = 14,
        zone: Optional[str] = None,
        worker_id: Optional[int] = None,
    ) -> dict:
        """One incident per person per zone visit, bucketed by plant-local day.

        A dwell visit can leave many DWELL_ALERT rows behind (the repeat
        interval re-fires while somebody stays), so repeats are folded into one
        incident per (worker, zone, entry). Entry and capacity alerts already
        fire once per incident, so the same grouping is a no-op for them.
        ENTRY/EXIT history rows are never counted. Each incident lands on the
        local date of its first alert, and the result maps (day, event_type)
        to the count - days with no alerts are simply absent, and the caller
        fills the zeros.
        """
        where: list[str] = []
        params: list = []
        where.append("event_type IN ('ENTRY_ALERT','DWELL_ALERT','CAPACITY_ALERT')")
        where.append("COALESCE(alert_at, entered_at) IS NOT NULL")
        # Plant-local (IST) days, matching the IST bounds the table filters
        # use: 'localtime' would bucket by the server's own timezone instead.
        where.append(
            "date(COALESCE(alert_at, entered_at), 'unixepoch', '+5 hours', '+30 minutes')"
            " >= date('now', '+5 hours', '+30 minutes', ?)"
        )
        params.append(f"-{max(1, int(days)) - 1} days")
        if zone is not None and str(zone) != "":
            if str(zone).strip().lstrip("-").isdigit():
                where.append("zone_index = ?")
                params.append(int(str(zone).strip()))
            else:
                where.append("zone_name = ?")
                params.append(str(zone))
        if worker_id is not None:
            where.append("worker_id = ?")
            params.append(int(worker_id))

        sql = (
            "SELECT day, event_type, COUNT(*) AS n FROM ("
            " SELECT event_type,"
            " date(COALESCE(MIN(alert_at), MIN(entered_at)), 'unixepoch', '+5 hours', '+30 minutes') AS day"
            " FROM events WHERE " + " AND ".join(where) +
            " GROUP BY event_type, worker_id, zone_index, entered_at"
            ") GROUP BY day, event_type"
        )
        try:
            with self._connect() as conn:
                return {
                    (day, event_type): count
                    for day, event_type, count in conn.execute(sql, params).fetchall()
                }
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not count alerts by day: {exc}") from exc

    def get_event(self, event_id: int) -> Optional[dict]:
        try:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT * FROM events WHERE id = ?", (int(event_id),)).fetchone()
                return dict(row) if row else None
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not read event {event_id}: {exc}") from exc

    def set_snapshot_path(self, event_id: int, snapshot_path: str) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE events SET snapshot_path = ? WHERE id = ?",
                    (str(snapshot_path), int(event_id)),
                )
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not set snapshot for event {event_id}: {exc}") from exc

    def acknowledge_event(self, event_id: int, acknowledged_at: float) -> bool:
        """Mark one event acknowledged. Returns False when the id is unknown."""
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    "UPDATE events SET acknowledged = 1, acknowledged_at = ? WHERE id = ?",
                    (float(acknowledged_at), int(event_id)),
                )
                return cursor.rowcount > 0
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not acknowledge event {event_id}: {exc}") from exc

    def acknowledge_all(
        self,
        acknowledged_at: float,
        event_types: Optional[list[str]] = None,
        zone: Optional[str] = None,
    ) -> int:
        """Mark every unacknowledged alert acknowledged in one query, all
        sharing the same timestamp. Only real alert kinds are ever touched -
        entries, exits and return-to-safe rows are records, not prompts - and
        an explicit type filter is intersected with them, so asking for
        ZONE_ENTRY acks nothing. Returns how many rows changed."""
        ackable = [ENTRY_ALERT, DWELL_ALERT, CAPACITY_ALERT]
        wanted = [t for t in (event_types or []) if t in ackable]
        types = wanted if event_types else ackable

        where = ["acknowledged = 0"]
        params: list = []
        placeholders = ",".join("?" for _ in types)
        where.append(f"event_type IN ({placeholders})")
        params.extend(types)
        if zone is not None and str(zone) != "":
            if str(zone).strip().lstrip("-").isdigit():
                where.append("zone_index = ?")
                params.append(int(str(zone).strip()))
            else:
                where.append("zone_name = ?")
                params.append(str(zone))
        params.insert(0, float(acknowledged_at))

        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    "UPDATE events SET acknowledged = 1, acknowledged_at = ? WHERE "
                    + " AND ".join(where),
                    params,
                )
                return cursor.rowcount
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not acknowledge all events: {exc}") from exc

    def latest_event(self, event_type: Optional[str] = None) -> Optional[dict]:
        rows = self.list_events(limit=1, event_type=event_type)
        return rows[0] if rows else None

    def count(self) -> int:
        try:
            with self._connect() as conn:
                return int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not count events: {exc}") from exc

    # -- fire/smoke alerts (own table; zone rows and counts untouched) -----

    @staticmethod
    def _fire_row(row) -> dict:
        try:
            bbox = json.loads(row[4]) if row[4] else None
        except (ValueError, TypeError):
            bbox = None
        return {
            "id": row[0],
            "alert_type": row[1],
            "severity": row[2],
            "confidence": row[3],
            "bbox_xyxy": bbox,
            "captured_at": row[5],
            "reported_at": row[6],
            "snapshot_path": row[7],
            "clean_snapshot_path": row[8],
            "acknowledged": bool(row[9]),
            "acknowledged_at": row[10],
            "false_alarm": bool(row[11]),
            "false_alarm_at": row[12],
            "source_label": row[13],
        }

    _FIRE_SELECT = (
        "SELECT id, alert_type, severity, confidence, bbox_xyxy,"
        " captured_at, reported_at, snapshot_path, clean_snapshot_path,"
        " acknowledged, acknowledged_at, false_alarm, false_alarm_at,"
        " source_label"
        " FROM fire_alerts"
    )

    def insert_fire_alert(self, alert_type: str, severity: str, confidence: float,
                          bbox_xyxy, captured_at: Optional[float],
                          reported_at: float, source_label: Optional[str] = None) -> int:
        """Store one fire/smoke alert. Returns the new row id (stable)."""
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    "INSERT INTO fire_alerts (alert_type, severity, confidence,"
                    " bbox_xyxy, captured_at, reported_at, source_label)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (str(alert_type), str(severity), float(confidence),
                     json.dumps([float(v) for v in bbox_xyxy]) if bbox_xyxy is not None else None,
                     float(captured_at) if captured_at is not None else None,
                     float(reported_at),
                     str(source_label) if source_label else None),
                )
                return int(cursor.lastrowid)
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not store fire alert: {exc}") from exc

    def set_fire_snapshot_path(self, alert_id: int, snapshot_path: str) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE fire_alerts SET snapshot_path = ? WHERE id = ?",
                    (str(snapshot_path), int(alert_id)),
                )
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not set fire snapshot for alert {alert_id}: {exc}") from exc

    def get_fire_alert(self, alert_id: int) -> Optional[dict]:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    self._FIRE_SELECT + " WHERE id = ?",
                    (int(alert_id),),
                ).fetchone()
                return self._fire_row(row) if row else None
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not read fire alert {alert_id}: {exc}") from exc

    def list_fire_alerts(self, acknowledged: Optional[bool] = None,
                         limit: int = 100, offset: int = 0) -> list:
        where = []
        params: list = []
        if acknowledged is not None:
            where.append("acknowledged = ?")
            params.append(1 if acknowledged else 0)
        query = self._FIRE_SELECT
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([max(1, int(limit)), max(0, int(offset))])
        try:
            with self._connect() as conn:
                return [self._fire_row(row) for row in conn.execute(query, params)]
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not list fire alerts: {exc}") from exc

    def count_fire_alerts(self, acknowledged: Optional[bool] = None) -> int:
        query = "SELECT COUNT(*) FROM fire_alerts"
        params: list = []
        if acknowledged is not None:
            query += " WHERE acknowledged = ?"
            params.append(1 if acknowledged else 0)
        try:
            with self._connect() as conn:
                return int(conn.execute(query, params).fetchone()[0])
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not count fire alerts: {exc}") from exc

    # -- merged history (zone events + fire/smoke alerts, one query) ----
    #
    # The History page shows both tables newest-first. The page slice comes
    # out of a single UNION ALL ordered by timestamp, so paging and totals
    # can never disagree about where a row belongs; only the hydration of
    # that slice is two small IN queries. Zone rows keep their exact
    # historical filter meaning (a zone filter never changes what the zone
    # table returns); the same filter value matches a fire row on its
    # source label instead. "FIRE"/"SMOKE" in the type filter select fire
    # rows and exclude zone rows, and vice versa. A worker filter excludes
    # fire rows (they are about a place, not a person). false_alarm=True
    # selects flagged fire rows only; False keeps every zone row and the
    # unflagged fire rows.

    FIRE_HISTORY_TYPES = ("FIRE", "SMOKE")

    @classmethod
    def _split_history_types(
        cls, event_types: Optional[list[str]]
    ) -> tuple[list[str], Optional[list[str]]]:
        """Split a mixed type filter into zone kinds and fire kinds.

        Returns (zone_types, fire_kinds) where fire_kinds is None for "all
        fire rows" and [] for "no fire rows".
        """
        if not event_types:
            return [], None
        zone_types = [t for t in event_types if str(t).upper() not in cls.FIRE_HISTORY_TYPES]
        fire_kinds = [str(t).upper() for t in event_types if str(t).upper() in cls.FIRE_HISTORY_TYPES]
        return zone_types, fire_kinds

    @classmethod
    def _fire_history_where(
        cls,
        fire_kinds: Optional[list[str]],
        acknowledged: Optional[bool],
        worker_id: Optional[int],
        zone: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
        false_alarm: Optional[bool],
    ) -> tuple[list[str], list]:
        """WHERE clause for the fire half of merged history. Bound params only."""
        if worker_id is not None:
            return ["1 = 0"], []
        if fire_kinds is not None and not fire_kinds:
            return ["1 = 0"], []
        where: list[str] = []
        params: list = []
        if fire_kinds:
            placeholders = ",".join("?" for _ in fire_kinds)
            where.append(f"UPPER(alert_type) IN ({placeholders})")
            params.extend(fire_kinds)
        if acknowledged is not None:
            where.append("acknowledged = ?")
            params.append(1 if acknowledged else 0)
        if zone is not None and str(zone) != "":
            # A numeric zone index never matches a fire row; comparing the
            # label to the raw value keeps that a plain empty result.
            where.append("source_label = ?")
            params.append(str(zone))
        if date_from:
            start, _ = cls._day_bounds_ist(date_from)
            where.append("reported_at >= ?")
            params.append(start)
        if date_to:
            _, end = cls._day_bounds_ist(date_to)
            where.append("reported_at < ?")
            params.append(end)
        if false_alarm is not None:
            where.append("false_alarm = ?")
            params.append(1 if false_alarm else 0)
        return where, params

    def _history_union(
        self,
        event_types: Optional[list[str]],
        acknowledged: Optional[bool],
        worker_id: Optional[int],
        zone: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
        false_alarm: Optional[bool],
    ) -> tuple[str, list]:
        """The UNION ALL both the page query and (via counts) the total share."""
        zone_types, fire_kinds = self._split_history_types(event_types)
        parts: list[str] = []
        params: list = []
        if not (event_types and not zone_types) and false_alarm is not True:
            zone_where, zone_params = self._event_where(
                event_types=zone_types or None,
                acknowledged=acknowledged,
                worker_id=worker_id,
                zone=zone,
                date_from=date_from,
                date_to=date_to,
            )
            zone_sql = "SELECT 'zone' AS kind, id, COALESCE(alert_at, entered_at) AS ts FROM events"
            if zone_where:
                zone_sql += " WHERE " + " AND ".join(zone_where)
            parts.append(zone_sql)
            params.extend(zone_params)
        fire_where, fire_params = self._fire_history_where(
            fire_kinds, acknowledged, worker_id, zone, date_from, date_to, false_alarm
        )
        fire_sql = (
            "SELECT 'fire' AS kind, id, reported_at AS ts FROM fire_alerts"
        )
        if fire_where:
            fire_sql += " WHERE " + " AND ".join(fire_where)
        parts.append(fire_sql)
        params.extend(fire_params)
        return " UNION ALL ".join(parts), params

    def list_history(
        self,
        event_types: Optional[list[str]] = None,
        acknowledged: Optional[bool] = None,
        worker_id: Optional[int] = None,
        zone: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        false_alarm: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """One merged page, newest first. Each row is its table's dict plus
        a "kind" key ("zone" or "fire")."""
        union_sql, params = self._history_union(
            event_types, acknowledged, worker_id, zone,
            date_from, date_to, false_alarm,
        )
        page_sql = (
            f"SELECT kind, id FROM ({union_sql})"
            " ORDER BY ts DESC, kind ASC, id DESC LIMIT ? OFFSET ?"
        )
        params.extend([max(1, int(limit)), max(0, int(offset))])
        try:
            with self._connect() as conn:
                pairs = [(kind, int(row_id)) for kind, row_id in conn.execute(page_sql, params).fetchall()]
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not read history: {exc}") from exc
        if not pairs:
            return []
        zone_ids = [row_id for kind, row_id in pairs if kind == "zone"]
        fire_ids = [row_id for kind, row_id in pairs if kind == "fire"]
        by_id: dict[tuple[str, int], dict] = {}
        try:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                if zone_ids:
                    placeholders = ",".join("?" for _ in zone_ids)
                    for row in conn.execute(
                        f"SELECT * FROM events WHERE id IN ({placeholders})", zone_ids
                    ).fetchall():
                        item = dict(row)
                        by_id[("zone", int(item["id"]))] = {"kind": "zone", **item}
                if fire_ids:
                    placeholders = ",".join("?" for _ in fire_ids)
                    for row in conn.execute(
                        self._FIRE_SELECT + f" WHERE id IN ({placeholders})", fire_ids
                    ).fetchall():
                        item = self._fire_row(row)
                        by_id[("fire", int(item["id"]))] = {"kind": "fire", **item}
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not read history: {exc}") from exc
        # A row deleted between the slice and the hydration simply drops out;
        # the order is the union's order, never the tables'.
        return [by_id[key] for key in pairs if key in by_id]

    def count_history(
        self,
        event_types: Optional[list[str]] = None,
        acknowledged: Optional[bool] = None,
        worker_id: Optional[int] = None,
        zone: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        false_alarm: Optional[bool] = None,
    ) -> int:
        """How many merged rows the current filters match, ignoring paging."""
        zone_types, fire_kinds = self._split_history_types(event_types)
        total = 0
        if not (event_types and not zone_types) and false_alarm is not True:
            total += self.count_events(
                event_types=zone_types or None,
                acknowledged=acknowledged,
                worker_id=worker_id,
                zone=zone,
                date_from=date_from,
                date_to=date_to,
            )
        fire_where, fire_params = self._fire_history_where(
            fire_kinds, acknowledged, worker_id, zone, date_from, date_to, false_alarm
        )
        fire_sql = "SELECT COUNT(*) FROM fire_alerts"
        if fire_where:
            fire_sql += " WHERE " + " AND ".join(fire_where)
        try:
            with self._connect() as conn:
                row = conn.execute(fire_sql, fire_params).fetchone()
                total += int(row[0]) if row else 0
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not count history: {exc}") from exc
        return total

    def fire_counts_by_day(
        self,
        days: int = 14,
        zone: Optional[str] = None,
        worker_id: Optional[int] = None,
    ) -> dict:
        """Fire/smoke alerts per plant-local day: (day, KIND) -> count.

        Fire rows belong to a place, not a person, so a worker filter matches
        nothing; a zone filter matches the stored source label. Days with no
        alerts are absent and the caller fills the zeros.
        """
        if worker_id is not None:
            return {}
        where = ["reported_at IS NOT NULL"]
        params: list = []
        where.append(
            "date(reported_at, 'unixepoch', '+5 hours', '+30 minutes')"
            " >= date('now', '+5 hours', '+30 minutes', ?)"
        )
        params.append(f"-{max(1, int(days)) - 1} days")
        if zone is not None and str(zone) != "":
            where.append("source_label = ?")
            params.append(str(zone))
        sql = (
            "SELECT date(reported_at, 'unixepoch', '+5 hours', '+30 minutes') AS day,"
            " UPPER(alert_type) AS kind, COUNT(*) AS n FROM fire_alerts"
            " WHERE " + " AND ".join(where) + " GROUP BY day, kind"
        )
        try:
            with self._connect() as conn:
                return {
                    (day, kind): count
                    for day, kind, count in conn.execute(sql, params).fetchall()
                }
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not count fire alerts by day: {exc}") from exc

    def acknowledge_fire_alert(self, alert_id: int, acknowledged_at: float) -> bool:
        """Mark one fire/smoke alert acknowledged. False when unknown."""
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    "UPDATE fire_alerts SET acknowledged = 1, acknowledged_at = ? WHERE id = ?",
                    (float(acknowledged_at), int(alert_id)),
                )
                return cursor.rowcount > 0
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not acknowledge fire alert {alert_id}: {exc}") from exc

    def set_fire_clean_snapshot_path(self, alert_id: int, snapshot_path: str) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE fire_alerts SET clean_snapshot_path = ? WHERE id = ?",
                    (str(snapshot_path), int(alert_id)),
                )
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not set fire clean snapshot for alert {alert_id}: {exc}") from exc

    def mark_fire_false_alarm(self, alert_id: int, marked_at: float) -> bool:
        """Flag one fire/smoke alert as a false alarm (and acknowledge it).

        Only an unflagged row changes: a repeat mark is a no-op returning
        False, which is what makes the export endpoint idempotent. Returns
        False as well when the id is unknown (check with get_fire_alert to
        tell the two apart).
        """
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    "UPDATE fire_alerts SET false_alarm = 1, false_alarm_at = ?,"
                    " acknowledged = 1, acknowledged_at = ?"
                    " WHERE id = ? AND false_alarm = 0",
                    (float(marked_at), float(marked_at), int(alert_id)),
                )
                return cursor.rowcount > 0
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not mark fire alert {alert_id} false alarm: {exc}") from exc

    # -- settings (polygon, allowed people, video source) -----------------

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
                return row[0] if row else default
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not read setting '{key}': {exc}") from exc

    def set_setting(self, key: str, value: str) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO settings (key, value) VALUES (?, ?)"
                    " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, str(value)),
                )
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not save setting '{key}': {exc}") from exc

    def all_settings(self) -> dict:
        try:
            with self._connect() as conn:
                return {key: value for key, value in conn.execute("SELECT key, value FROM settings")}
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not read settings: {exc}") from exc

    def update_json_setting(self, key: str, mutate) -> dict:
        """Read-modify-write a JSON-object setting in one transaction.

        `mutate` is handed the current dict and returns the dict to store. Both
        the read and the write happen under a single BEGIN IMMEDIATE, so two
        saves racing on the same key cannot each read the old value and write
        back over one another: the second waits for the first to commit and
        then sees its result.
        """
        import json

        try:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT value FROM settings WHERE key = ?", (key,)
                ).fetchone()
                try:
                    current = json.loads(row[0]) if row and row[0] else {}
                except (json.JSONDecodeError, TypeError, ValueError):
                    current = {}
                if not isinstance(current, dict):
                    current = {}
                updated = mutate(current)
                conn.execute(
                    "INSERT INTO settings (key, value) VALUES (?, ?)"
                    " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, json.dumps(updated)),
                )
                return updated
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not update setting '{key}': {exc}") from exc
