import os
import re
import json
from pathlib import Path
from typing import Dict, List

INPUT_DIR = "workplace/MONOPII/pii_outputs_600m"
OUTPUT_DIR = "pii_cleaned_email_600m"

os.makedirs(OUTPUT_DIR, exist_ok=True)

EMAIL_REGEX = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)

def extract_emails(text: str) -> List[str]:
    return EMAIL_REGEX.findall(text)


def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    email_counts: Dict[str, Dict[str, int]] = {}


    jsonl_files = sorted(input_dir.glob("mgpt*_email.jsonl"))
    print(f"Found {len(jsonl_files)} email jsonl files.")

    for path in jsonl_files:
        fname = path.name
        print(f"Processing {fname}...")

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except:
                    continue

                lang = obj.get("language")
                if not lang:
                    continue

                completion = obj.get("completion", "")
                if not completion:
                    continue

                emails = extract_emails(completion)
                if not emails:
                    continue

                d = email_counts.setdefault(lang, {})
                for e in emails:
                    key = e.lower()
                    d[key] = d.get(key, 0) + 1


    for lang, counts in email_counts.items():
        out_path = output_dir / f"emails_{lang}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for pii, dup in sorted(counts.items(), key=lambda x: -x[1]):
                record = {
                    "language": lang,
                    "pii_type": "email",
                    "pii": pii,
                    "duplicate": dup,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[email] {lang}: {len(counts)} unique emails -> {out_path}")

    print("Done (email).")


if __name__ == "__main__":
    main()
