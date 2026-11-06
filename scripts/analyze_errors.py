import json

# Load predictions
predictions = {}

with open("predictions.jsonl", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        record = json.loads(line)

        try:
            predictions[record["id"]] = json.loads(record["prediction"])
        except json.JSONDecodeError:
            predictions[record["id"]] = None


# Load validation data
validation = []

with open("data/validation.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            validation.append(json.loads(line))


# Compare
wrong_count = 0
invalid_count = 0

for item in validation:
    vid = item["id"]
    pred = predictions.get(vid)

    if pred is None:
        invalid_count += 1
        print(f"\n{vid} | {item['text']}")
        print("EXPECTED:")
        print(
            item["intent"],
            "|", item["date"],
            "|", item["time"],
            "|", item["place"]
        )
        print("PREDICTED: INVALID JSON")
        print("-" * 80)
        continue

    differences = []

    for field in ["intent", "date", "time", "place"]:
        if pred.get(field) != item[field]:
            differences.append(field)

    if differences:
        wrong_count += 1

        print(f"\n{vid} | {item['text']}")
        print("DIFFERENCES:", ", ".join(differences))

        print(
            "EXPECTED:  ",
            item["intent"],
            "|", item["date"],
            "|", item["time"],
            "|", item["place"]
        )

        print(
            "PREDICTED: ",
            pred.get("intent"),
            "|", pred.get("date"),
            "|", pred.get("time"),
            "|", pred.get("place")
        )

        print("-" * 80)


print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("Wrong examples:", wrong_count)
print("Invalid JSON:", invalid_count)
print("Total validation examples:", len(validation))