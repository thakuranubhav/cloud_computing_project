import json
import urllib.request
import urllib.error


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "phi3.5"


OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent", "date", "time", "place"],
    "properties": {
        "intent": {
            "type": "string",
            "enum": [
                "alarm_set",
                "alarm_query",
                "calendar_set",
                "calendar_query",
                "calendar_remove",
                "weather_query",
                "datetime_query",
                "recommendation_events",
            ],
        },
        "date": {"type": ["string", "null"]},
        "time": {"type": ["string", "null"]},
        "place": {"type": ["string", "null"]},
    },
}


SYSTEM_PROMPT = r"""
You are a strict command parser.

Parse the user's command into exactly four fields:

intent
date
time
place

Return ONLY one JSON object.

Do not return markdown.
Do not return explanations.
Do not return reasoning.
Do not add additional fields.

ALLOWED INTENTS:

alarm_set
alarm_query
calendar_set
calendar_query
calendar_remove
weather_query
datetime_query
recommendation_events

SLOT RULES:

date, time, and place must be either a string or JSON null.

If a slot is not explicitly mentioned, use JSON null.

NEVER output the string "null".

Copy slot values exactly as they appear in the command.

Do not:
- capitalize slot values
- normalize slot values
- translate slot values
- invent information
- add information that is not present

The slot must be the shortest phrase that carries the meaning.

Do not include grammatical connector words such as:
at
on
in
for

inside the slot value.

IMPORTANT TEMPORAL RULE:

"now"
"right now"
"currently"
"at the moment"

are temporal context expressions.

For datetime_query, they are NOT date values
and NOT time values.

For example:

Command:
give me the time at washington right now

Output:
{"intent":"datetime_query","date":null,"time":null,"place":"washington"}

INTENT RULES:

alarm_set:
The user wants to create or set an alarm.

alarm_query:
The user asks about an existing alarm.

calendar_set:
The user wants to create or schedule a calendar event.

calendar_query:
The user asks about existing calendar events.

calendar_remove:
The user wants to remove/delete a calendar event.

weather_query:
The user asks about weather.

datetime_query:
The user asks for the current date/time or time at a location.

recommendation_events:
The user asks for recommendations about events.

EXAMPLES:

Command:
what time is it in london

Output:
{"intent":"datetime_query","date":null,"time":null,"place":"london"}

Command:
what is the weather in paris tomorrow

Output:
{"intent":"weather_query","date":"tomorrow","time":null,"place":"paris"}

Command:
set an alarm for sunday at five

Output:
{"intent":"alarm_set","date":"sunday","time":"five","place":null}

Command:
show my meetings next monday

Output:
{"intent":"calendar_query","date":"next monday","time":null,"place":null}

Command:
remove the meeting on friday

Output:
{"intent":"calendar_remove","date":"friday","time":null,"place":null}

Command:
remind me of the meeting on tuesday

Output:
{"intent":"calendar_set","date":"tuesday","time":null,"place":null}

Apply the rules to the new command.

Return ONLY the JSON object.
"""


TEST_CASES = [
    {
        "command": "set an alarm for sunday at five",
        "expected_intent": "alarm_set",
    },
    {
        "command": "what alarms are on right now",
        "expected_intent": "alarm_query",
    },
    {
        "command": "remind me of the meeting on tuesday",
        "expected_intent": "calendar_set",
    },
    {
        "command": "do i have any meetings scheduled the weekend after next",
        "expected_intent": "calendar_query",
    },
    {
        "command": "remove the upcoming calendar event",
        "expected_intent": "calendar_remove",
    },
    {
        "command": "what is the weather in paris tomorrow",
        "expected_intent": "weather_query",
    },
    {
        "command": "give me the time at washington right now",
        "expected_intent": "datetime_query",
    },
    {
        "command": "what events should i attend this weekend",
        "expected_intent": "recommendation_events",
    },
]


def validate_output(obj):
    expected_keys = {"intent", "date", "time", "place"}

    if not isinstance(obj, dict):
        raise ValueError("Output is not a JSON object.")

    if set(obj.keys()) != expected_keys:
        raise ValueError(
            f"Wrong keys: {set(obj.keys())}"
        )

    allowed_intents = {
        "alarm_set",
        "alarm_query",
        "calendar_set",
        "calendar_query",
        "calendar_remove",
        "weather_query",
        "datetime_query",
        "recommendation_events",
    }

    if obj["intent"] not in allowed_intents:
        raise ValueError(
            f"Invalid intent: {obj['intent']}"
        )

    for slot in ["date", "time", "place"]:
        if obj[slot] is not None and not isinstance(obj[slot], str):
            raise ValueError(
                f"{slot} must be string or null."
            )

        if obj[slot] == "null":
            raise ValueError(
                f"{slot} contains string 'null'."
            )

    return obj


def parse_command(command):
    payload = {
        "model": MODEL,
        "stream": False,
        "format": OUTPUT_SCHEMA,
        "options": {
            "temperature": 0,
            "seed": 42,
        },
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Command: {command}",
            },
        ],
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=180
        ) as response:

            raw = response.read().decode("utf-8")

    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama. "
            "Make sure Ollama is running."
        ) from exc

    result = json.loads(raw)

    content = result["message"]["content"]

    parsed = json.loads(content)

    return validate_output(parsed)


def main():

    passed = 0
    failed = 0

    print("=" * 70)
    print("PHI-3.5 TEACHER MULTI-INTENT TEST")
    print("=" * 70)
    print(f"Model       : {MODEL}")
    print("Temperature : 0")
    print("Seed        : 42")
    print("=" * 70)

    for index, test in enumerate(TEST_CASES, start=1):

        command = test["command"]
        expected = test["expected_intent"]

        print(f"\n[{index}/{len(TEST_CASES)}]")
        print(f"Command : {command}")

        try:
            output = parse_command(command)

            actual = output["intent"]

            print(f"Expected: {expected}")
            print(f"Actual  : {actual}")
            print(
                "Output  : "
                + json.dumps(output, ensure_ascii=False)
            )

            if actual == expected:
                print("STATUS  : PASS")
                passed += 1
            else:
                print("STATUS  : FAIL")
                failed += 1

        except Exception as exc:

            print("STATUS  : ERROR")
            print(f"ERROR   : {exc}")

            failed += 1

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Passed : {passed}")
    print(f"Failed : {failed}")

    if failed == 0:
        print("\nALL TEACHER TESTS PASSED.")
    else:
        print("\nSOME TEACHER TESTS FAILED.")


if __name__ == "__main__":
    main()
