"""
Option A: Calendar Assistant Tools

Six tools for Google Calendar, each a plain Python function with type hints and
a contract-style docstring ("Use this when ...") so the LLM can pick the right
one. Every tool:

- validates its inputs before touching the API,
- never raises: it returns {"status": "success", ...} or
  {"status": "error", "message": <user-friendly text>, "hint": <how to fix>},
- works in one configurable time zone (CALENDAR_TIMEZONE, default
  America/Los_Angeles), so "2pm" always means 2pm local time.

Write operations (create_event, reschedule_event) check for conflicts first and
refuse to double-book unless the caller explicitly passes allow_conflicts=True.
"""

import os
import re
from datetime import date, datetime, time, timedelta
from typing import List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tools.auth import get_calendar_service

try:  # Only for nicer error messages; tools still work if these are missing.
    from googleapiclient.errors import HttpError
except ImportError:  # pragma: no cover
    HttpError = None  # type: ignore[assignment]

try:
    from google.auth.exceptions import RefreshError
except ImportError:  # pragma: no cover
    RefreshError = None  # type: ignore[assignment]


# =============================================================================
# Configuration
# =============================================================================
CALENDAR_ID = os.getenv("CALENDAR_ID", "primary")
MAX_RESULTS_CAP = int(os.getenv("CALENDAR_MAX_RESULTS", "50"))
SLOT_STEP_MINUTES = 30          # candidate slots start on :00 / :30
MAX_SEARCH_DAYS = 31            # find_available_slots search window limit
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _get_timezone():
    """IANA time zone from CALENDAR_TIMEZONE.

    Windows has no system time-zone database, so the `tzdata` package (in
    requirements.txt) provides it. If it is missing, fall back to the
    computer's local time zone instead of crashing.
    """
    name = os.getenv("CALENDAR_TIMEZONE", "America/Los_Angeles")
    for candidate in (name, "America/Los_Angeles"):
        try:
            return ZoneInfo(candidate), candidate
        except (ZoneInfoNotFoundError, ValueError):
            continue
    local = datetime.now().astimezone().tzinfo
    return local, "UTC"  # name only used for API formatting; offsets stay correct


TZ, TZ_NAME = _get_timezone()


# =============================================================================
# Internal helpers (not exposed to the LLM)
# =============================================================================
class _InputError(ValueError):
    """Invalid user input; the message is shown to the user as-is."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


def _error(message: str, hint: str = "") -> dict:
    result = {"status": "error", "message": message}
    if hint:
        result["hint"] = hint
    return result


def _api_error(exc: Exception, action: str) -> dict:
    """Translate API / auth exceptions into friendly, actionable messages."""
    if isinstance(exc, _InputError):
        return _error(str(exc), exc.hint)
    if isinstance(exc, FileNotFoundError):
        return _error(
            "Google Calendar is not connected yet (credentials.json not found).",
            "Follow docs/google_auth_setup.md and save the OAuth file to "
            "config/credentials/credentials.json.",
        )
    if RefreshError is not None and isinstance(exc, RefreshError):
        return _error(
            "Your Google sign-in has expired or was revoked.",
            "Delete config/credentials/token.json and run the agent again to sign in.",
        )
    if HttpError is not None and isinstance(exc, HttpError):
        code = getattr(getattr(exc, "resp", None), "status", None)
        messages = {
            400: ("Google Calendar rejected the request as invalid.",
                  "Check the date/time format (YYYY-MM-DD HH:MM) and that end is after start."),
            401: ("Google Calendar authorization failed.",
                  "Delete config/credentials/token.json and sign in again."),
            403: ("Access to Google Calendar was denied or the rate limit was hit.",
                  "Make sure the Calendar API is enabled; if you sent many requests, wait a minute."),
            404: ("That event or calendar was not found.",
                  "List events first to get a valid event id."),
            409: ("That event was changed at the same time by someone else.",
                  "List the event again and retry."),
            410: ("That event has already been deleted.", "List events to see what still exists."),
            429: ("Too many requests to Google Calendar.", "Wait about a minute and try again."),
        }
        if code in messages:
            msg, hint = messages[code]
            return _error(f"Could not {action}: {msg}", hint)
        if code and int(code) >= 500:
            return _error(f"Could not {action}: Google Calendar is temporarily unavailable.",
                          "Try again in a moment.")
    return _error(f"Could not {action}: {type(exc).__name__}: {exc}")


def _parse_datetime(value: str, field: str = "time") -> datetime:
    """Parse 'YYYY-MM-DD HH:MM' / ISO 8601 into an aware datetime in TZ."""
    if not value or not isinstance(value, str):
        raise _InputError(f"Missing {field}.", "Use the format YYYY-MM-DD HH:MM, e.g. 2026-10-13 14:00.")
    text = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        raise _InputError(
            f"I couldn't understand the {field} '{value}'.",
            "Use the format YYYY-MM-DD HH:MM, e.g. 2026-10-13 14:00.",
        ) from None
    if len(text) == 10:  # date only -> start of day
        dt = datetime.combine(dt.date(), time(0, 0))
    return dt.replace(tzinfo=TZ) if dt.tzinfo is None else dt.astimezone(TZ)


def _parse_date(value: str, field: str = "date") -> date:
    try:
        return date.fromisoformat(value.strip()[:10])
    except (ValueError, AttributeError):
        raise _InputError(f"I couldn't understand the {field} '{value}'.",
                          "Use the format YYYY-MM-DD, e.g. 2026-10-13.") from None


def _parse_clock(value: str, field: str) -> time:
    try:
        return time.fromisoformat(value.strip())
    except (ValueError, AttributeError):
        raise _InputError(f"I couldn't understand the {field} '{value}'.",
                          "Use 24-hour HH:MM, e.g. 09:00 or 17:30.") from None


def _validate_emails(emails: Optional[List[str]]) -> List[str]:
    cleaned = [e.strip() for e in (emails or []) if e and e.strip()]
    bad = [e for e in cleaned if not EMAIL_RE.match(e)]
    if bad:
        raise _InputError(f"These attendee emails look invalid: {', '.join(bad)}.",
                          "Use full addresses like name@example.com.")
    return cleaned


def _resolve_end(start: datetime, end_time: Optional[str], duration_minutes: int) -> datetime:
    if end_time:
        end = _parse_datetime(end_time, "end time")
    else:
        if not 5 <= int(duration_minutes) <= 24 * 60:
            raise _InputError("Duration must be between 5 minutes and 24 hours.")
        end = start + timedelta(minutes=int(duration_minutes))
    if end <= start:
        raise _InputError("The end time must be after the start time.")
    return end


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _human(dt: datetime) -> str:
    return dt.strftime("%a %b %d, %Y %I:%M %p").replace(" 0", " ")


def _event_time(raw: dict) -> str:
    """Event start/end -> readable local time (all-day events show the date)."""
    if "dateTime" in raw:
        return _human(datetime.fromisoformat(raw["dateTime"].replace("Z", "+00:00")).astimezone(TZ))
    return f"{raw.get('date', '?')} (all day)"


def _format_event(item: dict) -> dict:
    return {
        "event_id": item.get("id"),
        "summary": item.get("summary", "(no title)"),
        "start": _event_time(item.get("start", {})),
        "end": _event_time(item.get("end", {})),
        "location": item.get("location", ""),
        "attendees": [a.get("email") for a in item.get("attendees", []) if a.get("email")],
        "link": item.get("htmlLink", ""),
    }


def _is_blocking(item: dict) -> bool:
    """All-day, 'free' (transparent) and declined events don't block time."""
    if "dateTime" not in item.get("start", {}):
        return False
    if item.get("transparency") == "transparent":
        return False
    for attendee in item.get("attendees", []):
        if attendee.get("self") and attendee.get("responseStatus") == "declined":
            return False
    return True


def _find_conflicts(service, start: datetime, end: datetime,
                    exclude_event_id: Optional[str] = None) -> List[dict]:
    """Events on the user's calendar that overlap [start, end)."""
    items = service.events().list(
        calendarId=CALENDAR_ID, timeMin=_iso(start), timeMax=_iso(end),
        singleEvents=True, orderBy="startTime", maxResults=MAX_RESULTS_CAP,
    ).execute().get("items", [])
    return [_format_event(i) for i in items
            if _is_blocking(i) and i.get("id") != exclude_event_id]


def _busy_intervals(service, start: datetime, end: datetime,
                    attendees: List[str]) -> tuple[list[tuple[datetime, datetime]], list[str]]:
    """Merged busy intervals for the user + attendees, and calendars we couldn't read."""
    body = {
        "timeMin": _iso(start), "timeMax": _iso(end), "timeZone": TZ_NAME,
        "items": [{"id": CALENDAR_ID}] + [{"id": e} for e in attendees],
    }
    calendars = service.freebusy().query(body=body).execute().get("calendars", {})
    busy, unreadable = [], []
    for cal_id, info in calendars.items():
        if info.get("errors"):
            unreadable.append(cal_id)
            continue
        for b in info.get("busy", []):
            busy.append((_parse_datetime(b["start"]), _parse_datetime(b["end"])))
    busy.sort()
    merged: list[tuple[datetime, datetime]] = []
    for s, e in busy:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged, unreadable


def _round_up(dt: datetime, minutes: int) -> datetime:
    dt = dt.replace(second=0, microsecond=0)
    remainder = dt.minute % minutes
    return dt if remainder == 0 else dt + timedelta(minutes=minutes - remainder)


# =============================================================================
# Tools exposed to the agent
# =============================================================================
def get_current_datetime() -> dict:
    """Get today's date, the current time, and the weekday in the user's time zone.

    Use this FIRST whenever the user says a relative date such as "today",
    "tomorrow", "next Monday" or "next week", so you can convert it to an exact
    YYYY-MM-DD date before calling other calendar tools.

    Returns:
        dict with 'status', 'now' (YYYY-MM-DD HH:MM), 'date', 'weekday' and 'timezone'.
    """
    try:
        now = datetime.now(TZ)
        return {
            "status": "success",
            "now": now.strftime("%Y-%m-%d %H:%M"),
            "date": now.strftime("%Y-%m-%d"),
            "weekday": now.strftime("%A"),
            "timezone": TZ_NAME,
        }
    except Exception as e:  # extremely unlikely, but never crash the agent
        return _error(f"Could not read the current time: {e}")


def list_upcoming_events(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_results: int = 10,
    search_text: Optional[str] = None,
) -> dict:
    """List calendar events in a date range (what's on my calendar / my schedule).

    Use this when the user asks what meetings or events they have, e.g.
    "What do I have tomorrow?" or "Show my meetings next week". Also use it to
    find an event's event_id before rescheduling it.

    Args:
        start_date: First day to include, YYYY-MM-DD. Defaults to right now.
        end_date: Last day to include (inclusive), YYYY-MM-DD. Defaults to no limit.
        max_results: Maximum number of events to return (1-50). Default 10.
        search_text: Optional keyword to match in event titles/descriptions,
            e.g. "dentist" or "team sync".

    Returns:
        dict with 'status', 'count' and 'events' (each with event_id, summary,
        start, end, location, attendees, link), or 'status': 'error' and 'message'.
    """
    try:
        if not 1 <= int(max_results) <= MAX_RESULTS_CAP:
            raise _InputError(f"max_results must be between 1 and {MAX_RESULTS_CAP}.")
        now = datetime.now(TZ)
        time_min = (datetime.combine(_parse_date(start_date, "start date"), time(0, 0), TZ)
                    if start_date else now)
        params = {
            "calendarId": CALENDAR_ID, "timeMin": _iso(time_min),
            "maxResults": int(max_results), "singleEvents": True, "orderBy": "startTime",
        }
        if end_date:
            time_max = datetime.combine(_parse_date(end_date, "end date") + timedelta(days=1),
                                        time(0, 0), TZ)
            if time_max <= time_min:
                raise _InputError("The end date must be on or after the start date.")
            params["timeMax"] = _iso(time_max)
        if search_text:
            params["q"] = search_text

        service = get_calendar_service()
        items = service.events().list(**params).execute().get("items", [])
        events = [_format_event(i) for i in items]
        return {
            "status": "success",
            "count": len(events),
            "events": events,
            "message": "No events found in that range." if not events else "",
        }
    except Exception as e:
        return _api_error(e, "list your events")


def find_available_slots(
    start_date: str,
    end_date: Optional[str] = None,
    duration_minutes: int = 60,
    attendee_emails: Optional[List[str]] = None,
    work_day_start: str = "09:00",
    work_day_end: str = "17:00",
    include_weekends: bool = False,
    max_slots: int = 8,
) -> dict:
    """Find free time slots when the user (and optional attendees) are all available.

    Use this when the user wants to know when they are free, or asks to find a
    time for a meeting, e.g. "Find a free hour next week" or "When can Alice
    and I meet on Friday?". Do NOT use it to create the event; call
    create_event after the user picks a slot.

    Args:
        start_date: First day to search, YYYY-MM-DD.
        end_date: Last day to search (inclusive), YYYY-MM-DD. Defaults to start_date.
        duration_minutes: Length of the meeting in minutes. Default 60.
        attendee_emails: Optional other people's emails whose calendars should
            also be free (only works if their free/busy is shared with the user).
        work_day_start: Earliest start time each day, 24-hour HH:MM. Default 09:00.
        work_day_end: Latest end time each day, 24-hour HH:MM. Default 17:00.
        include_weekends: Whether Saturday/Sunday may be used. Default False.
        max_slots: Maximum number of slots to return (1-20). Default 8.

    Returns:
        dict with 'status', 'slots' (each with 'start' and 'end' as YYYY-MM-DD HH:MM
        plus a readable 'label'), and 'unchecked_attendees' for calendars that
        could not be read; or 'status': 'error' and 'message'.
    """
    try:
        first = _parse_date(start_date, "start date")
        last = _parse_date(end_date, "end date") if end_date else first
        if last < first:
            raise _InputError("The end date must be on or after the start date.")
        if (last - first).days >= MAX_SEARCH_DAYS:
            raise _InputError(f"Please search at most {MAX_SEARCH_DAYS} days at a time.")
        if not 5 <= int(duration_minutes) <= 12 * 60:
            raise _InputError("Duration must be between 5 minutes and 12 hours.")
        if not 1 <= int(max_slots) <= 20:
            raise _InputError("max_slots must be between 1 and 20.")
        day_start = _parse_clock(work_day_start, "work day start")
        day_end = _parse_clock(work_day_end, "work day end")
        if day_end <= day_start:
            raise _InputError("work_day_end must be later than work_day_start.")
        attendees = _validate_emails(attendee_emails)

        window_start = datetime.combine(first, day_start, TZ)
        window_end = datetime.combine(last, day_end, TZ)
        now = datetime.now(TZ)
        if window_end <= now:
            raise _InputError("That date range is already in the past.",
                              "Call get_current_datetime to check today's date.")

        service = get_calendar_service()
        busy, unreadable = _busy_intervals(service, window_start, window_end, attendees)

        length = timedelta(minutes=int(duration_minutes))
        step = timedelta(minutes=SLOT_STEP_MINUTES)
        slots: list[dict] = []
        day = first
        while day <= last and len(slots) < int(max_slots):
            if include_weekends or day.weekday() < 5:
                cursor = datetime.combine(day, day_start, TZ)
                if cursor < now:
                    cursor = _round_up(now, SLOT_STEP_MINUTES)
                closing = datetime.combine(day, day_end, TZ)
                while cursor + length <= closing and len(slots) < int(max_slots):
                    end = cursor + length
                    clash = next((b for b in busy if b[0] < end and cursor < b[1]), None)
                    if clash:
                        cursor = _round_up(clash[1], SLOT_STEP_MINUTES)
                        continue
                    slots.append({
                        "start": cursor.strftime("%Y-%m-%d %H:%M"),
                        "end": end.strftime("%Y-%m-%d %H:%M"),
                        "label": f"{_human(cursor)} - {end.strftime('%I:%M %p').lstrip('0')}",
                    })
                    cursor += length if length >= step else step
            day += timedelta(days=1)

        result = {
            "status": "success",
            "timezone": TZ_NAME,
            "slots": slots,
            "unchecked_attendees": unreadable,
        }
        if not slots:
            result["message"] = "No free slots found. Try a wider date range, a shorter meeting, or include weekends."
        if unreadable:
            result["note"] = ("Could not see free/busy for: " + ", ".join(unreadable) +
                              ". Their availability was not checked.")
        return result
    except Exception as e:
        return _api_error(e, "find free time")


def check_conflicts(
    start_time: str,
    end_time: Optional[str] = None,
    duration_minutes: int = 60,
    attendee_emails: Optional[List[str]] = None,
) -> dict:
    """Check whether a proposed time overlaps existing events (scheduling conflicts).

    Use this when the user asks "Am I free at ...?", "Does 3pm Tuesday clash
    with anything?", or before suggesting a specific time.

    Args:
        start_time: Proposed start, YYYY-MM-DD HH:MM (24-hour, user's time zone).
        end_time: Proposed end, YYYY-MM-DD HH:MM. If omitted, duration_minutes is used.
        duration_minutes: Length in minutes when end_time is not given. Default 60.
        attendee_emails: Optional emails of other people to check as well.

    Returns:
        dict with 'status', 'has_conflict' (bool), 'conflicts' (overlapping events
        on the user's calendar) and 'busy_attendees'; or 'status': 'error'.
    """
    try:
        start = _parse_datetime(start_time, "start time")
        end = _resolve_end(start, end_time, duration_minutes)
        attendees = _validate_emails(attendee_emails)

        service = get_calendar_service()
        conflicts = _find_conflicts(service, start, end)

        busy_attendees, unchecked = [], []
        if attendees:
            body = {"timeMin": _iso(start), "timeMax": _iso(end), "timeZone": TZ_NAME,
                    "items": [{"id": e} for e in attendees]}
            cals = service.freebusy().query(body=body).execute().get("calendars", {})
            for email in attendees:
                info = cals.get(email, {})
                if info.get("errors"):
                    unchecked.append(email)
                elif info.get("busy"):
                    busy_attendees.append(email)

        has_conflict = bool(conflicts or busy_attendees)
        return {
            "status": "success",
            "requested": f"{_human(start)} - {end.strftime('%I:%M %p').lstrip('0')}",
            "has_conflict": has_conflict,
            "conflicts": conflicts,
            "busy_attendees": busy_attendees,
            "unchecked_attendees": unchecked,
            "message": "That time is free." if not has_conflict else "That time overlaps existing commitments.",
        }
    except Exception as e:
        return _api_error(e, "check for conflicts")


def create_event(
    summary: str,
    start_time: str,
    end_time: Optional[str] = None,
    duration_minutes: int = 60,
    attendee_emails: Optional[List[str]] = None,
    description: str = "",
    location: str = "",
    allow_conflicts: bool = False,
) -> dict:
    """Create (schedule) a new calendar event and email invitations to attendees.

    Use this ONLY after the user has confirmed the title, date and time, e.g.
    "Yes, schedule the team sync Monday at 2pm". It checks for conflicts first
    and will NOT double-book unless allow_conflicts is True.

    Args:
        summary: Event title, e.g. "Team sync".
        start_time: Start, YYYY-MM-DD HH:MM (24-hour, user's time zone).
        end_time: End, YYYY-MM-DD HH:MM. If omitted, duration_minutes is used.
        duration_minutes: Length in minutes when end_time is not given. Default 60.
        attendee_emails: Optional guest emails; each guest receives an invitation.
        description: Optional notes/agenda.
        location: Optional place or video link.
        allow_conflicts: Set True only if the user explicitly agreed to overlap
            an existing event.

    Returns:
        dict with 'status': 'success' and the created 'event'; 'status': 'conflict'
        with the overlapping events (nothing was created); or 'status': 'error'.
    """
    try:
        if not summary or not summary.strip():
            raise _InputError("The event needs a title.", "Ask the user what to call the event.")
        start = _parse_datetime(start_time, "start time")
        end = _resolve_end(start, end_time, duration_minutes)
        if start < datetime.now(TZ) - timedelta(minutes=5):
            raise _InputError("That start time is in the past.",
                              "Call get_current_datetime and choose a future time.")
        attendees = _validate_emails(attendee_emails)

        service = get_calendar_service()
        if not allow_conflicts:
            conflicts = _find_conflicts(service, start, end)
            if conflicts:
                return {
                    "status": "conflict",
                    "message": "Not created: this time overlaps existing events. Ask the user "
                               "to pick another time or confirm the double-booking.",
                    "conflicts": conflicts,
                }

        body = {
            "summary": summary.strip(),
            "start": {"dateTime": _iso(start), "timeZone": TZ_NAME},
            "end": {"dateTime": _iso(end), "timeZone": TZ_NAME},
        }
        if description:
            body["description"] = description
        if location:
            body["location"] = location
        if attendees:
            body["attendees"] = [{"email": e} for e in attendees]

        created = service.events().insert(
            calendarId=CALENDAR_ID, body=body,
            sendUpdates="all" if attendees else "none",
        ).execute()
        return {
            "status": "success",
            "message": "Event created." + (" Invitations sent." if attendees else ""),
            "event": _format_event(created),
        }
    except Exception as e:
        return _api_error(e, "create the event")


def reschedule_event(
    event_id: str,
    new_start_time: str,
    new_end_time: Optional[str] = None,
    allow_conflicts: bool = False,
) -> dict:
    """Move an existing event to a new time (reschedule), keeping everything else.

    Use this when the user wants to move, postpone or change the time of an
    existing meeting. Get the event_id from list_upcoming_events first. If no
    new end time is given, the original duration is kept. Attendees are
    notified of the change.

    Args:
        event_id: The event_id returned by list_upcoming_events.
        new_start_time: New start, YYYY-MM-DD HH:MM (24-hour, user's time zone).
        new_end_time: Optional new end, YYYY-MM-DD HH:MM.
        allow_conflicts: Set True only if the user explicitly agreed to overlap
            another event.

    Returns:
        dict with 'status': 'success' and the updated 'event'; 'status': 'conflict'
        with overlapping events (nothing changed); or 'status': 'error'.
    """
    try:
        if not event_id or not event_id.strip():
            raise _InputError("Missing event_id.", "Call list_upcoming_events to find the event first.")
        new_start = _parse_datetime(new_start_time, "new start time")
        if new_start < datetime.now(TZ) - timedelta(minutes=5):
            raise _InputError("The new start time is in the past.",
                              "Call get_current_datetime and choose a future time.")

        service = get_calendar_service()
        event = service.events().get(calendarId=CALENDAR_ID, eventId=event_id.strip()).execute()
        old_start_raw, old_end_raw = event.get("start", {}), event.get("end", {})
        if "dateTime" not in old_start_raw:
            raise _InputError("All-day events can't be moved with this tool yet.",
                              "Edit the all-day event in Google Calendar directly.")
        old_duration = (_parse_datetime(old_end_raw["dateTime"]) -
                        _parse_datetime(old_start_raw["dateTime"]))
        new_end = (_parse_datetime(new_end_time, "new end time") if new_end_time
                   else new_start + old_duration)
        if new_end <= new_start:
            raise _InputError("The new end time must be after the new start time.")

        if not allow_conflicts:
            conflicts = _find_conflicts(service, new_start, new_end, exclude_event_id=event["id"])
            if conflicts:
                return {
                    "status": "conflict",
                    "message": "Not moved: the new time overlaps other events. Ask the user to "
                               "pick another time or confirm the double-booking.",
                    "conflicts": conflicts,
                }

        updated = service.events().patch(
            calendarId=CALENDAR_ID, eventId=event["id"],
            body={
                "start": {"dateTime": _iso(new_start), "timeZone": TZ_NAME},
                "end": {"dateTime": _iso(new_end), "timeZone": TZ_NAME},
            },
            sendUpdates="all",
        ).execute()
        return {
            "status": "success",
            "message": f"Moved '{updated.get('summary', 'event')}' from "
                       f"{_event_time(old_start_raw)} to {_human(new_start)}.",
            "event": _format_event(updated),
        }
    except Exception as e:
        return _api_error(e, "reschedule the event")


calendar_tools = [
    get_current_datetime,
    list_upcoming_events,
    find_available_slots,
    check_conflicts,
    create_event,
    reschedule_event,
]
