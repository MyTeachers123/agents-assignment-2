"""
Offline unit tests for the calendar tools (no Google account needed).

A small in-memory fake replaces the Google Calendar client, so these tests
check the tools' logic: input validation, conflict detection, free-slot
search, and friendly error messages.

Usage (from workspace_assistant/):  python -m pytest tests/test_calendar_offline.py -q
"""

from datetime import datetime, timedelta

import pytest

import tools.calendar_tools as ct


class _Exec:
    def __init__(self, value):
        self._value = value

    def execute(self):
        if isinstance(self._value, Exception):
            raise self._value
        return self._value


class FakeCalendar:
    """Implements just enough of the Calendar v3 client used by the tools."""

    def __init__(self, events=None, unreadable=()):
        self.events_store = {e["id"]: e for e in (events or [])}
        self.unreadable = set(unreadable)
        self.sent_updates = []

    # --- events() -----------------------------------------------------
    def events(self):
        return self

    def list(self, calendarId, timeMin, timeMax=None, q=None, **_):
        lo = ct._parse_datetime(timeMin)
        hi = ct._parse_datetime(timeMax) if timeMax else None
        out = []
        for e in sorted(self.events_store.values(), key=lambda e: e["start"]["dateTime"]):
            s = ct._parse_datetime(e["start"]["dateTime"])
            en = ct._parse_datetime(e["end"]["dateTime"])
            if en > lo and (hi is None or s < hi) and (not q or q.lower() in e["summary"].lower()):
                out.append(e)
        return _Exec({"items": out})

    def insert(self, calendarId, body, sendUpdates="none"):
        new = dict(body, id=f"evt{len(self.events_store) + 1}", htmlLink="https://cal/x")
        self.events_store[new["id"]] = new
        self.sent_updates.append(sendUpdates)
        return _Exec(new)

    def get(self, calendarId, eventId):
        if eventId not in self.events_store:
            return _Exec(KeyError("not found"))
        return _Exec(self.events_store[eventId])

    def patch(self, calendarId, eventId, body, sendUpdates="none"):
        self.events_store[eventId].update(body)
        self.sent_updates.append(sendUpdates)
        return _Exec(self.events_store[eventId])

    # --- freebusy() ---------------------------------------------------
    def freebusy(self):
        return self

    def query(self, body):
        lo, hi = ct._parse_datetime(body["timeMin"]), ct._parse_datetime(body["timeMax"])
        cals = {}
        for item in body["items"]:
            cid = item["id"]
            if cid in self.unreadable:
                cals[cid] = {"errors": [{"reason": "notFound"}]}
                continue
            busy = []
            if cid == "primary":
                for e in self.events_store.values():
                    s = ct._parse_datetime(e["start"]["dateTime"])
                    en = ct._parse_datetime(e["end"]["dateTime"])
                    if en > lo and s < hi:
                        busy.append({"start": e["start"]["dateTime"], "end": e["end"]["dateTime"]})
            cals[cid] = {"busy": busy}
        return _Exec({"calendars": cals})


def _day(offset_days: int):
    """A weekday date `offset_days` or more in the future (YYYY-MM-DD)."""
    d = datetime.now(ct.TZ).date() + timedelta(days=offset_days)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d.isoformat()


def _event(eid, summary, day, start, end):
    return {"id": eid, "summary": summary,
            "start": {"dateTime": f"{day}T{start}:00"},
            "end": {"dateTime": f"{day}T{end}:00"}}


@pytest.fixture
def fake(monkeypatch):
    day = _day(7)
    cal = FakeCalendar(
        events=[_event("e1", "Team sync", day, "10:00", "11:00"),
                _event("e2", "Dentist", day, "14:00", "15:00")],
        unreadable={"stranger@example.com"},
    )
    monkeypatch.setattr(ct, "get_calendar_service", lambda: cal)
    cal.day = day
    return cal


def test_current_datetime():
    r = ct.get_current_datetime()
    assert r["status"] == "success" and len(r["date"]) == 10


def test_list_events_in_range(fake):
    r = ct.list_upcoming_events(start_date=fake.day, end_date=fake.day)
    assert r["status"] == "success" and r["count"] == 2
    assert r["events"][0]["summary"] == "Team sync"


def test_list_events_bad_date(fake):
    r = ct.list_upcoming_events(start_date="next tuesday")
    assert r["status"] == "error" and "YYYY-MM-DD" in r["hint"]


def test_find_slots_skips_busy_time(fake):
    r = ct.find_available_slots(fake.day, duration_minutes=60, max_slots=20)
    starts = [s["start"][-5:] for s in r["slots"]]
    assert r["status"] == "success"
    assert "10:00" not in starts and "14:00" not in starts and "13:30" not in starts
    assert starts[0] == "09:00" and "11:00" in starts and "15:00" in starts


def test_find_slots_reports_unreadable_attendee(fake):
    r = ct.find_available_slots(fake.day, attendee_emails=["stranger@example.com"])
    assert r["unchecked_attendees"] == ["stranger@example.com"] and "note" in r


def test_check_conflicts(fake):
    clash = ct.check_conflicts(f"{fake.day} 10:30", duration_minutes=30)
    free = ct.check_conflicts(f"{fake.day} 12:00", f"{fake.day} 13:00")
    assert clash["has_conflict"] and clash["conflicts"][0]["summary"] == "Team sync"
    assert free["has_conflict"] is False


def test_create_event_blocks_double_booking(fake):
    r = ct.create_event("Overlap", f"{fake.day} 10:30")
    assert r["status"] == "conflict" and len(fake.events_store) == 2


def test_create_event_success_sends_invites(fake):
    r = ct.create_event("1:1", f"{fake.day} 16:00", duration_minutes=30,
                        attendee_emails=["alice@example.com"])
    assert r["status"] == "success" and r["event"]["attendees"] == ["alice@example.com"]
    assert fake.sent_updates[-1] == "all"


def test_create_event_validation(fake):
    assert ct.create_event("", f"{fake.day} 16:00")["status"] == "error"
    assert ct.create_event("x", "2020-01-01 10:00")["status"] == "error"
    assert ct.create_event("x", f"{fake.day} 16:00", f"{fake.day} 15:00")["status"] == "error"
    bad = ct.create_event("x", f"{fake.day} 16:00", attendee_emails=["not-an-email"])
    assert bad["status"] == "error" and "invalid" in bad["message"]


def test_reschedule_keeps_duration_and_detects_conflict(fake):
    moved = ct.reschedule_event("e1", f"{fake.day} 16:00")
    assert moved["status"] == "success" and "from" in moved["message"]
    assert fake.events_store["e1"]["end"]["dateTime"].startswith(f"{fake.day}T17:00")
    clash = ct.reschedule_event("e1", f"{fake.day} 14:30")
    assert clash["status"] == "conflict"


def test_reschedule_unknown_event(fake):
    r = ct.reschedule_event("missing", f"{fake.day} 16:00")
    assert r["status"] == "error"


def test_missing_credentials_message(monkeypatch):
    def boom():
        raise FileNotFoundError("credentials.json")
    monkeypatch.setattr(ct, "get_calendar_service", boom)
    r = ct.list_upcoming_events()
    assert r["status"] == "error" and "google_auth_setup" in r["hint"]
