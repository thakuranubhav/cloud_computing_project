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

INPUT_FILE = PROJECT_ROOT / "data" / "commands_pool.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_FILE = RESULTS_DIR / "teacher_outputs.jsonl"
ERROR_FILE = RESULTS_DIR / "teacher_errors.jsonl"

MAX_EXAMPLES = 1500

TEMPERATURE = 0
SEED = 42
TIMEOUT = 180


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

- Use null when a slot is not explicitly present in the command.
- Do not invent information.
- Do not add explanations.
- Do not add additional keys.
- Preserve the user's temporal wording when it is explicitly present.
- "right now", "now", "currently", and similar expressions do NOT
  belong in the time slot.
- For datetime_query, a location such as "Washington", "New York",
  "California", or "Beijing" belongs in the place slot.
- For weather_query, a location belongs in the place slot.
- A calendar command asking about an existing event is calendar_query.
- A command asking to remove/delete/cancel an event is calendar_remove.
- A command asking to create/schedule/set a calendar event is calendar_set.
- A command asking to create/set an alarm is alarm_set.
- A command asking what alarms exist is alarm_query.
- A command asking for event recommendations is recommendation_events.

Examples:

Input:
"set an alarm for sunday at five"

Output:
{
  "intent": "alarm_set",
  "date": "sunday",
  "time": "five",
  "place": null
}

Input:
"what alarms are on right now"

Output:
{
  "intent": "alarm_query",
  "date": null,
  "time": null,
  "place": null
}

Input:
"remind me of the meeting on tuesday"

Output:
{
  "intent": "calendar_set",
  "date": "tuesday",
  "time": null,
  "place": null
}

Input:
"do i have any meetings scheduled the weekend after next"

Output:
{
  "intent": "calendar_query",
  "date": "the weekend after next",
  "time": null,
  "place": null
}

Input:
"remove the upcoming calendar event"

Output:
{
  "intent": "calendar_remove",
  "date": null,
  "time": null,
  "place": null
}

Input:
"what is the weather in paris tomorrow"

Output:
{
  "intent": "weather_query",
  "date": "tomorrow",
  "time": null,
  "place": "paris"
}

Input:
"give me the time at washington right now"

Output:
{
  "intent": "datetime_query",
  "date": null,
  "time": null,
  "place": "washington"
}

Input:
"what events should i attend this weekend"

Output:
{
  "intent": "recommendation_events",
  "date": "this weekend",
  "time": null,
  "place": null
}
"""


# ============================================================
# Utility functions
# ============================================================

def load_commands():
    """Load commands from commands_pool.jsonl."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    records = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"WARNING: Invalid JSON at line {line_number}: {exc}",
                    file=sys.stderr,
                )
                continue

            # IMPORTANT:
            # commands_pool.jsonl uses the "text" field.
            command_text = record.get("text")

            if not isinstance(command_text, str):
                continue

            command_text = command_text.strip()

            if not command_text:
                continue

            records.append(
                {
                    "id": record.get("id"),
                    "text": command_text,
                }
            )

    return records


def validate_output(parsed):
    """Validate the parsed teacher output."""

    if not isinstance(parsed, dict):
        return False, "output is not an object"

    expected_keys = {
        "intent",
        "date",
        "time",
        "place",
    }

    if set(parsed.keys()) != expected_keys:
        return False, "incorrect JSON keys"

    allowed_intents = {
        "alarm_set",
        "alarm_query",
        "calendar_set",
        "calendar_query",
        "calendar_remove",
        "datetime_query",
        "weather_query",
        "recommendation_events",
    }

    if parsed["intent"] not in allowed_intents:
        return False, f"invalid intent: {parsed['intent']}"

    for slot in ["date", "time", "place"]:
        value = parsed[slot]

        if value is not None and not isinstance(value, str):
            return False, f"{slot} must be string or null"

        # The teacher must return JSON null, not the string "null".
        if isinstance(value, str) and value.lower() == "null":
            return False, f"{slot} incorrectly returned as string 'null'"

    return True, None


def parse_teacher_response(response_json):
    """
    Extract the assistant response from Ollama and parse it as JSON.
    """

    message = response_json.get("message", {})

    raw_response = message.get("content", "")

    if not isinstance(raw_response, str):
        raise ValueError("Ollama response content is not a string")

    raw_response = raw_response.strip()

    if not raw_response:
        raise ValueError("Ollama returned an empty response")

    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Teacher response is not valid JSON: {exc}"
        ) from exc

    valid, error = validate_output(parsed)

    if not valid:
        raise ValueError(error)

    return raw_response, parsed


# ============================================================
# Ollama request
# ============================================================

def call_teacher(command_text):
    """Send one command to the Phi-3.5 teacher."""

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": command_text,
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

    return response.json()


# ============================================================
# Main generation
# ============================================================

def main():

    print("=" * 70)
    print("Teacher Data Generation")
    print("=" * 70)

    print(f"Teacher model : {MODEL}")
    print(f"Ollama URL    : {OLLAMA_URL}")
    print(f"Temperature   : {TEMPERATURE}")
    print(f"Seed          : {SEED}")
    print(f"Input         : {INPUT_FILE}")
    print(f"Output        : {OUTPUT_FILE}")
    print(f"Max examples  : {MAX_EXAMPLES}")
    print()

    # --------------------------------------------------------
    # Load commands
    # --------------------------------------------------------

    records = load_commands()

    print(f"Loaded {len(records)} cleaned commands.")

    if len(records) < MAX_EXAMPLES:
        print(
            f"ERROR: Expected at least {MAX_EXAMPLES} commands, "
            f"but found {len(records)}."
        )
        sys.exit(1)

    records = records[:MAX_EXAMPLES]

    print(
        f"Using first {len(records)} commands for teacher generation."
    )
    print()

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Generate teacher outputs
    # --------------------------------------------------------

    success_count = 0
    error_count = 0

    start_time = time.time()

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as output_file, ERROR_FILE.open(
        "w",
        encoding="utf-8",
    ) as error_file:

        for index, record in enumerate(records, start=1):

            command_id = record["id"]
            command_text = record["text"]

            print(
                f"[{index}/{len(records)}] "
                f"{command_id}: {command_text}"
            )

            try:

                response_json = call_teacher(command_text)

                raw_response, parsed_output = parse_teacher_response(
                    response_json
                )

                # Store the exact teacher reply together with
                # the original command.
                result = {
                    "id": command_id,
                    "text": command_text,
                    "teacher_raw_response": raw_response,
                    "teacher_output": parsed_output,
                }

                output_file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                output_file.flush()

                success_count += 1

                print(
                    f"    PASS -> {json.dumps(parsed_output, ensure_ascii=False)}"
                )

            except Exception as exc:

                error_record = {
                    "id": command_id,
                    "text": command_text,
                    "error": str(exc),
                }

                error_file.write(
                    json.dumps(
                        error_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                error_file.flush()

                error_count += 1

                print(
                    f"    ERROR -> {exc}",
                    file=sys.stderr,
                )

    elapsed = time.time() - start_time

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Teacher Generation Complete")
    print("=" * 70)

    print(f"Total commands : {len(records)}")
    print(f"Successful     : {success_count}")
    print(f"Failed         : {error_count}")
    print(f"Elapsed time   : {elapsed / 60:.2f} minutes")

    if success_count:
        print(
            f"Average time   : "
            f"{elapsed / success_count:.2f} seconds/example"
        )

    print()
    print(f"Teacher output : {OUTPUT_FILE}")
    print(f"Teacher errors : {ERROR_FILE}")
    print()

    if error_count == 0:
        print("ALL TEACHER EXAMPLES GENERATED SUCCESSFULLY.")
    else:
        print(
            f"WARNING: {error_count} examples failed. "
            "Check teacher_errors.jsonl."
        )


if __name__ == "__main__":
    main()
