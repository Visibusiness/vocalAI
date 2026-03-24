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

REQUIRED_KEYS = {"action", "name", "date", "time"}


def extract_appointment(ai_reply: str) -> tuple[str, dict | None]:
    """
    Parse the LLM reply and extract an appointment JSON block if present.

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

    # Validate required keys and action type
    if not REQUIRED_KEYS.issubset(data.keys()):
        missing = REQUIRED_KEYS - data.keys()
        print(f"[appointment_parser] Missing keys: {missing}")
        return clean_text, None

    if data.get("action") != "schedule":
        return clean_text, None

    return clean_text, data
