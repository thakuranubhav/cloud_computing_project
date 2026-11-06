import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "results" / "teacher_outputs.jsonl"
ERROR_FILE = PROJECT_ROOT / "results" / "teacher_data_validation_errors.jsonl"

EXPECTED_COUNT = 1500

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

REQUIRED_KEYS = {
    "intent",
    "date",
    "time",
    "place",
}

SLOT_KEYS = {
    "date",
    "time",
    "place",
}


def validate_record(record, line_number):

    errors = []

    if not isinstance(record, dict):
        return ["record is not a JSON object"]

    # --------------------------------------------------------
    # Required top-level fields
    # --------------------------------------------------------

    for key in [
        "id",
        "text",
        "teacher_raw_response",
        "teacher_output",
    ]:
        if key not in record:
            errors.append(f"missing top-level field: {key}")

    if errors:
        return errors

    # --------------------------------------------------------
    # Basic field validation
    # --------------------------------------------------------

    if not isinstance(record["id"], str):
        errors.append("id is not a string")

    if not isinstance(record["text"], str):
        errors.append("text is not a string")

    if not record["text"].strip():
        errors.append("text is empty")

    if not isinstance(record["teacher_raw_response"], str):
        errors.append("teacher_raw_response is not a string")

    # --------------------------------------------------------
    # Teacher output validation
    # --------------------------------------------------------

    output = record["teacher_output"]

    if not isinstance(output, dict):
        errors.append("teacher_output is not an object")
        return errors

    if set(output.keys()) != REQUIRED_KEYS:
        errors.append(
            f"incorrect teacher_output keys: "
            f"{sorted(output.keys())}"
        )
        return errors

    # --------------------------------------------------------
    # Intent
    # --------------------------------------------------------

    intent = output["intent"]

    if intent not in ALLOWED_INTENTS:
        errors.append(f"invalid intent: {intent}")

    # --------------------------------------------------------
    # Slots
    # --------------------------------------------------------

    for slot in SLOT_KEYS:

        value = output[slot]

        if value is not None and not isinstance(value, str):
            errors.append(
                f"{slot} must be string or null"
            )

        # Reject literal "null"
        if isinstance(value, str) and value.lower() == "null":
            errors.append(
                f"{slot} contains string 'null' instead of JSON null"
            )

    return errors


def main():

    print("=" * 70)
    print("Teacher Dataset Validation")
    print("=" * 70)

    if not INPUT_FILE.exists():
        print(f"ERROR: File not found: {INPUT_FILE}")
        sys.exit(1)

    records = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:

                records.append({
                    "line": line_number,
                    "errors": [
                        f"invalid JSON: {exc}"
                    ],
                })

                continue

            errors = validate_record(
                record,
                line_number,
            )

            if errors:
                records.append({
                    "line": line_number,
                    "id": record.get("id"),
                    "errors": errors,
                })

    # --------------------------------------------------------
    # Count records
    # --------------------------------------------------------

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        total_lines = sum(
            1
            for line in f
            if line.strip()
        )

    print(f"Expected records : {EXPECTED_COUNT}")
    print(f"Actual records   : {total_lines}")
    print(f"Invalid records  : {len(records)}")
    print()

    # --------------------------------------------------------
    # Save validation errors
    # --------------------------------------------------------

    if records:

        with ERROR_FILE.open(
            "w",
            encoding="utf-8",
        ) as f:

            for error in records:

                f.write(
                    json.dumps(
                        error,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        print(
            f"Validation errors saved to:\n"
            f"{ERROR_FILE}"
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    if total_lines != EXPECTED_COUNT:

        print(
            f"FAIL: Expected {EXPECTED_COUNT} records "
            f"but found {total_lines}."
        )

        sys.exit(1)

    if records:

        print(
            f"FAIL: Found {len(records)} invalid records."
        )

        sys.exit(1)

    print("PASS: All 1500 teacher examples are valid.")
    print("=" * 70)


if __name__ == "__main__":
    main()