"""
demo_seed.py

Pre-populates Google Calendar with realistic appointments for the demo.
Run this once before the presentation to seed the calendar.

Usage:
    GOOGLE_CALENDAR_ID="your@gmail.com" python demo_seed.py

What it creates:
    - Dr. Ionescu (Cardiologie)  — 2 Apr 10:00  (will conflict if client tries to book here)
    - Dr. Popescu (Medicină internă) — 2 Apr 14:00
    - Dr. Marinescu (Pediatrie) — 3 Apr 11:00

Demo scenarios this enables:
    1. "Vreau să mă programez la Dr. Ionescu pe 2 aprilie la 10" → conflict detected
    2. "Ce programări sunt pe 2 aprilie?" → full-day scan returns 2 events
    3. "Anulează programarea lui Gheorghe Munteanu" → cancellation flow
    4. Book a free slot → success
"""

import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(__file__))

from app.calendar_service import create_appointment

APPOINTMENTS = [
    {
        "name": "Alexandru Dima",
        "date": "2026-04-02",
        "time": "10:00",
        "phone": "+40721111001",
        "doctor": "Ionescu",
    },
    {
        "name": "Gheorghe Munteanu",
        "date": "2026-04-02",
        "time": "14:00",
        "phone": "+40721111002",
        "doctor": "Popescu",
    },
    {
        "name": "Elena Constantin",
        "date": "2026-04-03",
        "time": "11:00",
        "phone": "+40721111003",
        "doctor": "Marinescu",
    },
]


def main():
    print("Seeding demo calendar...\n")
    for appt in APPOINTMENTS:
        try:
            link = create_appointment(
                name=appt["name"],
                date_str=appt["date"],
                time_str=appt["time"],
                phone=appt["phone"],
                doctor=appt["doctor"],
            )
            print(f"  Created: Dr. {appt['doctor']} - {appt['name']} on {appt['date']} at {appt['time']}")
            print(f"           {link}\n")
        except Exception as e:
            print(f"  ERROR for {appt['name']}: {e}\n")

    print("Done. Demo calendar is ready.")
    print()
    print("Demo scenarios:")
    print("  1. Book a CONFLICTING slot: 'Vreau la Dr. Ionescu pe 2 aprilie la 10'")
    print("     → AI should detect conflict and propose another time")
    print()
    print("  2. Full-day scan: 'Ce programări sunt pe 2 aprilie?'")
    print("     → AI should list both appointments on that day")
    print()
    print("  3. Cancel: 'Anulează programarea pe 2 aprilie la 14'")
    print("     → AI should confirm and delete Gheorghe Munteanu's slot")
    print()
    print("  4. Book a FREE slot: 'Vreau la Dr. Popescu pe 3 aprilie la 15'")
    print("     → AI should confirm and create the event")


if __name__ == "__main__":
    main()
