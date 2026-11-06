"""Generate raw student-model predictions for validation data."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = """### Command:
{command}

### Response:
"""


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: line {line_number} is not valid JSON"
                ) from exc

    return rows


def generate_prediction(
    model,
    tokenizer,
    command: str,
    max_new_tokens: int,
) -> str:
    prompt = PROMPT_TEMPLATE.format(command=command)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing the downloaded student model",
    )

    parser.add_argument(
        "--validation",
        default="data/validation.jsonl",
        help="Validation JSONL file",
    )

    parser.add_argument(
        "--output",
        default="predictions.jsonl",
        help="Output prediction JSONL file",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    validation_path = Path(args.validation)
    output_path = Path(args.output)

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory does not exist: {model_dir}"
        )

    validation_rows = read_jsonl(validation_path)

    print(f"Loading model from: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
    )

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        f"Generating predictions for {len(validation_rows)} examples..."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for index, row in enumerate(validation_rows, 1):
            prediction = generate_prediction(
                model=model,
                tokenizer=tokenizer,
                command=row["text"],
                max_new_tokens=args.max_new_tokens,
            )

            result = {
                "id": row["id"],
                "prediction": prediction,
            }

            output_file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

            if index % 10 == 0 or index == len(validation_rows):
                print(
                    f"Generated {index}/{len(validation_rows)}"
                )

    print(f"Predictions written to: {output_path}")


if __name__ == "__main__":
    main()
