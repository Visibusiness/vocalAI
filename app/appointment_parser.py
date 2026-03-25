"""
appointment_parser.py

Extracts a structured appointment JSON block from the LLM reply.

The LLM is instructed to append a ```json ... ``` block when an appointment
is confirmed. This module finds that block, validates it, and returns:
  - clean_text : the reply with the JSON block removed (sent to TTS)
  - appointment : a dict with keys action/name/date/time, or None
"""

import re
import json

# Matches a fenced ```json ... ``` block (multiline)
_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)

REQUIRED_KEYS_SCHEDULE = {"action", "name", "date", "time"}
REQUIRED_KEYS_CANCEL = {"action", "date", "time"}
REQUIRED_KEYS_CHECK = {"action", "date", "time"}


def extract_appointment(ai_reply: str) -> tuple[str, dict | None]:
    """
    Parse the LLM reply and extract an appointment JSON block if present.

    Handles two actions:
      - "schedule": requires name, date, time
      - "cancel":   requires date, time

    Returns:
        clean_text    - the reply text with the JSON block stripped out
        appointment   - dict with appointment data, or None if not found / invalid
    """
    match = _JSON_BLOCK_RE.search(ai_reply)

    if not match:
        return ai_reply.strip(), None

    # Remove the JSON block from the text the user will hear
    clean_text = _JSON_BLOCK_RE.sub("", ai_reply).strip()

    raw_json = match.group(1)

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        print(f"[appointment_parser] JSON decode error: {e}")
        return clean_text, None

    # Reject any JSON that still has null values — AI sent it too early
    if any(v is None for v in data.values()):
        print(f"[appointment_parser] Rejected JSON with null values: {data}")
        return clean_text, None

    action = data.get("action")

    if action == "schedule":
        if not REQUIRED_KEYS_SCHEDULE.issubset(data.keys()):
            missing = REQUIRED_KEYS_SCHEDULE - data.keys()
            print(f"[appointment_parser] Missing keys for schedule: {missing}")
            return clean_text, None

    elif action == "cancel":
        if not REQUIRED_KEYS_CANCEL.issubset(data.keys()):
            missing = REQUIRED_KEYS_CANCEL - data.keys()
            print(f"[appointment_parser] Missing keys for cancel: {missing}")
            return clean_text, None

    elif action == "check":
        if not REQUIRED_KEYS_CHECK.issubset(data.keys()):
            missing = REQUIRED_KEYS_CHECK - data.keys()
            print(f"[appointment_parser] Missing keys for check: {missing}")
            return clean_text, None

    else:
        print(f"[appointment_parser] Unknown action: {action}")
        return clean_text, None

    return clean_text, data
