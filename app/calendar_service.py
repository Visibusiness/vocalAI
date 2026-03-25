"""
calendar_service.py

Handles Google Calendar integration.

Setup:
  1. Go to https://console.cloud.google.com
  2. Create a project → enable "Google Calendar API"
  3. Create a Service Account → download the JSON key as "credentials.json"
  4. Share your Google Calendar with the service account email (give it "Make changes to events" permission)
  5. Set CALENDAR_ID below to your calendar's ID (found in Google Calendar settings)

Install dependencies:
  pip install google-api-python-client google-auth
"""

import os
from datetime import datetime, timedelta

from google.oauth2 import service_account
from googleapiclient.discovery import build

# Path to the service account credentials file (place it in the project root)
CREDENTIALS_FILE = os.path.join(os.path.dirname(__file__), "..", "credentials.json")

# Your Google Calendar ID — find it in:
# Google Calendar → Settings → [your calendar] → "Calendar ID"
# For your primary calendar it is just your Gmail address.
CALENDAR_ID = os.environ.get("GOOGLE_CALENDAR_ID", "your_calendar_id@gmail.com")

SCOPES = ["https://www.googleapis.com/auth/calendar"]


def _get_service():
    """Authenticate and return a Google Calendar API service object."""
    credentials = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE, scopes=SCOPES
    )
    return build("calendar", "v3", credentials=credentials)


def check_conflict(date_str: str, time_str: str) -> bool:
    """
    Return True if there is already an event overlapping the requested 1-hour slot.

    Args:
        date_str : date in "YYYY-MM-DD" format
        time_str : time in "HH:MM" format
    """
    try:
        start_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except ValueError:
        return False

    end_dt = start_dt + timedelta(hours=1)

    # Google Calendar expects RFC3339 with timezone offset
    # We query with UTC bounds that cover the Europe/Bucharest slot
    # Using isoformat with Z suffix after converting; simplest: query with timeMin/timeMax
    time_min = start_dt.strftime("%Y-%m-%dT%H:%M:%S") + "+02:00"
    time_max = end_dt.strftime("%Y-%m-%dT%H:%M:%S") + "+02:00"

    service = _get_service()
    events_result = service.events().list(
        calendarId=CALENDAR_ID,
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True,
    ).execute()

    events = events_result.get("items", [])
    return len(events) > 0


def cancel_appointment(date_str: str, time_str: str) -> bool:
    """
    Delete the event at the given date/time slot.

    Returns True if an event was found and deleted, False if nothing was found.
    """
    try:
        start_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except ValueError:
        return False

    end_dt = start_dt + timedelta(hours=1)
    time_min = start_dt.strftime("%Y-%m-%dT%H:%M:%S") + "+02:00"
    time_max = end_dt.strftime("%Y-%m-%dT%H:%M:%S") + "+02:00"

    service = _get_service()
    events_result = service.events().list(
        calendarId=CALENDAR_ID,
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True,
    ).execute()

    events = events_result.get("items", [])
    if not events:
        return False

    for event in events:
        service.events().delete(calendarId=CALENDAR_ID, eventId=event["id"]).execute()
        print(f"[calendar_service] Deleted event: {event.get('summary', '')} at {date_str} {time_str}")

    return True


def create_appointment(name: str, date_str: str, time_str: str) -> str:
    """
    Create a 1-hour appointment in Google Calendar.

    Args:
        name     : patient name (used in event title)
        date_str : date in "YYYY-MM-DD" format
        time_str : time in "HH:MM" format

    Returns:
        The URL link to the created Google Calendar event.

    Raises:
        ValueError  : if date/time format is invalid
        Exception   : if the Google API call fails
    """
    # Parse and build start/end datetimes
    try:
        start_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except ValueError:
        raise ValueError(f"Invalid date/time format: date='{date_str}' time='{time_str}'")

    end_dt = start_dt + timedelta(hours=1)

    # Format as RFC3339 with Romanian timezone (UTC+3 in summer, UTC+2 in winter)
    # Using Europe/Bucharest timezone name for correctness
    timezone = "Europe/Bucharest"

    event = {
        "summary": f"Programare - {name}",
        "description": f"Programare creată automat de TestRec pentru {name}.",
        "start": {
            "dateTime": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "timeZone": timezone,
        },
        "end": {
            "dateTime": end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "timeZone": timezone,
        },
    }

    service = _get_service()
    created_event = service.events().insert(calendarId=CALENDAR_ID, body=event).execute()

    event_link = created_event.get("htmlLink", "")
    print(f"[calendar_service] Event created: {event_link}")
    return event_link
