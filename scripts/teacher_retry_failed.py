import json
import sys
import time
from pathlib import Path

import requests


# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "phi3.5"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ERROR_FILE = PROJECT_ROOT / "results" / "teacher_errors.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "results" / "teacher_outputs.jsonl"

TEMPERATURE = 0
SEED = 42

# Give failed requests more time than the original generation.
TIMEOUT = 300

# Required failed IDs
FAILED_IDS = {
    "C0177",
    "C1296",
    "C1298",
    "C1374",
}


# ============================================================
# Output schema
# ============================================================

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": [
                "alarm_set",
                "alarm_query",
                "calendar_set",
                "calendar_query",
                "calendar_remove",
                "datetime_query",
                "weather_query",
                "recommendation_events",
            ],
        },
        "date": {
            "type": ["string", "null"],
        },
        "time": {
            "type": ["string", "null"],
        },
        "place": {
            "type": ["string", "null"],
        },
    },
    "required": [
        "intent",
        "date",
        "time",
        "place",
    ],
    "additionalProperties": False,
}


# ============================================================
# Teacher prompt
# ============================================================

SYSTEM_PROMPT = """
You are a command-parsing language model.

Your task is to convert a natural-language user command into a
strict JSON object containing an intent and three optional slots.

Allowed intents:

1. alarm_set
2. alarm_query
3. calendar_set
4. calendar_query
5. calendar_remove
6. datetime_query
7. weather_query
8. recommendation_events

Return ONLY the JSON object.

The JSON must contain exactly these keys:

{
  "intent": "...",
  "date": "... or null",
  "time": "... or null",
  "place": "... or null"
}

Rules:

- Use JSON null when a slot is not explicitly present.
- NEVER return the string "null".
- Do not invent information.
- Do not add explanations.
- Do not add additional keys.
- Preserve the user's temporal wording when it is explicitly present.
- "right now", "now", "currently", and similar expressions do NOT
  belong in the time slot.
- For datetime_query, a location belongs in the place slot.
- For weather_query, a location belongs in the place slot.
- A calendar command asking about an existing event is calendar_query.
- A command asking to remove/delete/cancel an event is calendar_remove.
- A command asking to create/schedule/set a calendar event is calendar_set.
- A command asking to create/set an alarm is alarm_set.
- A command asking what alarms exist is alarm_query.
- A command asking for event recommendations is recommendation_events.

Important:
If a slot is absent, output actual JSON null, not "null".

Examples:

Input:
"please schedule meeting with ri at ten am on april first"

Output:
{
  "intent": "calendar_set",
  "date": "april first",
  "time": "ten am",
  "place": null
}

Input:
"delete the birthday party tomorrow at two pm"

Output:
{
  "intent": "calendar_remove",
  "date": "tomorrow",
  "time": "two pm",
  "place": null
}

Input:
"how is weather in new york city"

Output:
{
  "intent": "weather_query",
  "date": null,
  "time": null,
  "place": "new york city"
}

Input:
"every other wednesday remind me to go grocery shopping"

Output:
{
  "intent": "calendar_set",
  "date": "every other wednesday",
  "time": null,
  "place": null
}
"""


# ============================================================
# Validation
# ============================================================

ALLOWED_INTENTS = {
    "alarm_set",
    "alarm_query",
    "calendar_set",
    "calendar_query",
    "calendar_remove",
    "datetime_query",
    "weather_query",
    "recommendation_events",
}


def validate_output(output):

    if not isinstance(output, dict):
        return False, "output is not an object"

    expected_keys = {
        "intent",
        "date",
        "time",
        "place",
    }

    if set(output.keys()) != expected_keys:
        return False, "incorrect JSON keys"

    if output["intent"] not in ALLOWED_INTENTS:
        return False, "invalid intent"

    for slot in ["date", "time", "place"]:

        value = output[slot]

        if value is not None and not isinstance(value, str):
            return False, f"{slot} must be string or null"

        if isinstance(value, str) and value.lower() == "null":
            return False, (
                f"{slot} incorrectly returned as string 'null'"
            )

    return True, None


# ============================================================
# Load failed commands
# ============================================================

def load_failed_commands():

    if not ERROR_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {ERROR_FILE}"
        )

    failed = []

    with ERROR_FILE.open("r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if record.get("id") in FAILED_IDS:

                failed.append({
                    "id": record["id"],
                    "text": record["text"],
                    "original_error": record["error"],
                })

    return failed


# ============================================================
# Call teacher
# ============================================================

def call_teacher(command):

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": command,
            },
        ],
        "stream": False,
        "format": OUTPUT_SCHEMA,
        "options": {
            "temperature": TEMPERATURE,
            "seed": SEED,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=TIMEOUT,
    )

    response.raise_for_status()

    response_json = response.json()

    raw_response = (
        response_json
        .get("message", {})
        .get("content", "")
        .strip()
    )

    if not raw_response:
        raise ValueError("empty teacher response")

    parsed = json.loads(raw_response)

    valid, error = validate_output(parsed)

    if not valid:
        raise ValueError(error)

    return raw_response, parsed


# ============================================================
# Append successful result
# ============================================================

def append_result(record, raw_response, parsed):

    result = {
        "id": record["id"],
        "text": record["text"],
        "teacher_raw_response": raw_response,
        "teacher_output": parsed,
    }

    with OUTPUT_FILE.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                result,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Retry Failed Teacher Examples")
    print("=" * 70)

    failed = load_failed_commands()

    print(f"Failed examples found: {len(failed)}")
    print()

    success = 0
    remaining_failures = []

    for index, record in enumerate(failed, start=1):

        print(
            f"[{index}/{len(failed)}] "
            f"{record['id']}: {record['text']}"
        )

        print(
            f"Original error: {record['original_error']}"
        )

        try:

            start = time.time()

            raw_response, parsed = call_teacher(
                record["text"]
            )

            elapsed = time.time() - start

            append_result(
                record,
                raw_response,
                parsed,
            )

            success += 1

            print(
                f"PASS ({elapsed:.1f}s): "
                f"{json.dumps(parsed, ensure_ascii=False)}"
            )

        except Exception as exc:

            remaining_failures.append({
                "id": record["id"],
                "text": record["text"],
                "error": str(exc),
            })

            print(
                f"FAILED: {exc}",
                file=sys.stderr,
            )

        print()

    print("=" * 70)
    print("Retry Summary")
    print("=" * 70)

    print(f"Attempted : {len(failed)}")
    print(f"Success   : {success}")
    print(f"Failed    : {len(remaining_failures)}")

    if remaining_failures:

        retry_error_file = (
            PROJECT_ROOT
            / "results"
            / "teacher_retry_errors.jsonl"
        )

        with retry_error_file.open(
            "w",
            encoding="utf-8",
        ) as f:

            for record in remaining_failures:

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        print()
        print(
            f"Remaining failures saved to:\n"
            f"{retry_error_file}"
        )

        sys.exit(1)

    print()
    print(
        "ALL 4 FAILED EXAMPLES WERE RECOVERED."
    )


if __name__ == "__main__":
    main()
