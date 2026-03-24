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
