import json
import sys
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "results" / "teacher_outputs.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "student_train.jsonl"

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

REQUIRED_OUTPUT_KEYS = {
    "intent",
    "date",
    "time",
    "place",
}


# ============================================================
# Validate teacher output
# ============================================================

def validate_teacher_output(output):

    if not isinstance(output, dict):
        return False, "teacher_output is not an object"

    if set(output.keys()) != REQUIRED_OUTPUT_KEYS:
        return False, "incorrect output keys"

    if output["intent"] not in ALLOWED_INTENTS:
        return False, "invalid intent"

    for slot in ["date", "time", "place"]:

        value = output[slot]

        if value is not None and not isinstance(value, str):
            return False, f"{slot} must be string or null"

        if isinstance(value, str) and value.lower() == "null":
            return False, (
                f"{slot} contains string 'null'"
            )

    return True, None


# ============================================================
# Convert one example
# ============================================================

def convert_example(record):

    if "id" not in record:
        raise ValueError("missing id")

    if "text" not in record:
        raise ValueError("missing text")

    if "teacher_output" not in record:
        raise ValueError("missing teacher_output")

    command = record["text"]

    if not isinstance(command, str) or not command.strip():
        raise ValueError("invalid command text")

    teacher_output = record["teacher_output"]

    valid, error = validate_teacher_output(
        teacher_output
    )

    if not valid:
        raise ValueError(error)

    # Compact JSON is preferable for training.
    target = json.dumps(
        teacher_output,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    # Instruction-style causal LM training example.
    prompt = (
        "### Command:\n"
        f"{command.strip()}\n\n"
        "### Response:\n"
    )

    return {
        "id": record["id"],
        "text": prompt,
        "target": target,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Preparing Pythia-70M Student Dataset")
    print("=" * 70)

    if not INPUT_FILE.exists():

        print(
            f"ERROR: Input file not found:\n{INPUT_FILE}"
        )

        sys.exit(1)

    examples = []
    errors = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(line)

                example = convert_example(record)

                examples.append(example)

            except Exception as exc:

                errors.append({
                    "line": line_number,
                    "error": str(exc),
                })

    print(f"Input examples : {len(examples)}")
    print(f"Invalid examples: {len(errors)}")

    if len(examples) != EXPECTED_COUNT:

        print()
        print(
            f"ERROR: Expected exactly "
            f"{EXPECTED_COUNT} valid examples."
        )

        if errors:

            print()
            print("First errors:")

            for error in errors[:10]:
                print(error)

        sys.exit(1)

    if errors:

        print(
            "ERROR: Some examples failed validation."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Check duplicate IDs
    # --------------------------------------------------------

    ids = [example["id"] for example in examples]

    duplicate_ids = {
        item
        for item in ids
        if ids.count(item) > 1
    }

    if duplicate_ids:

        print(
            "ERROR: Duplicate IDs detected:"
        )

        for item in sorted(duplicate_ids):
            print(item)

        sys.exit(1)

    # --------------------------------------------------------
    # Write dataset
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        for example in examples:

            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print(
        f"Student dataset written to:\n"
        f"{OUTPUT_FILE}"
    )

    print()
    print("Dataset format:")

    print(
        json.dumps(
            examples[0],
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("=" * 70)
    print("PASS: Student dataset prepared successfully.")
    print(f"Examples: {len(examples)}")
    print("=" * 70)


if __name__ == "__main__":
    main()