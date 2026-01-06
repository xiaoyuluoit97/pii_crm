import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional

INPUT_DIR = "workplace/MONOPII/pii_outputs_600m"
OUTPUT_DIR = "pii_cleaned_email_600m"

os.makedirs(OUTPUT_DIR, exist_ok=True)


PROMPT_LANG_TO_CODE = {
    "af": "27",
    "ar": "20",
    "az": "994",
    "be": "375",
    "bg": "359",
    "da": "45",
    "de": "49",
    "el": "30",
    "en": "1",
    "es": "34",
    "fi": "358",
    "fr": "33",
    "hi": "91",
    "hu": "36",
    "it": "39",
    "ko": "82",
    "lt": "370",
    "lv": "371",
    "ml": "91",
    "nl": "31",
    "pl": "48",
    "pt": "55",
    "ro": "40",
    "ru": "7",
    "sv": "46",
    "sw": "255",
    "ta": "91",
    "th": "66",
    "tr": "90",
    "uk": "380",
    "vi": "84",
    "zh": "86",
}


LANG_TO_COUNTRY_CODES = {
    "zh": ["86", "852", "853", "886"],
    "th": ["66"],
    "da": ["45"],
    "de": ["49", "43", "41"],
    "ru": ["7"],
    "uk": ["380"],
    "be": ["375"],
    "es": ["34", "54", "52", "56", "57", "58", "502", "503", "504", "505", "506"],
    "it": ["39", "378", "3906"],
    "hi": ["91", "977", "975"],
    "en": ["1", "44", "61", "64", "353", "27", "65", "60", "63", "91"],
    "ja": ["81"],
    "ar": ["20", "212", "213", "218", "249", "962", "963", "964", "965", "966",
           "967", "968", "970", "971", "973", "974"],
    "he": ["972"],
    "ur": ["92", "971"],
    "af": ["27", "264", "268", "266", "267", "260"],
    "sv": ["46", "358", "47", "45"],
    "pt": ["351", "55", "258", "244", "238", "239", "290", "595"],
    "ro": ["40", "373"],
    "fr": ["33", "32", "41", "261", "596", "590", "687"],
    "ko": ["82"],
    "nl": ["31", "32", "599", "597"],
    "tr": ["90"],
    "tk": ["993"],
    "az": ["994"],
    "ta": ["91", "94", "65", "60"],
    "te": ["91"],
    "ml": ["91", "971"],
    "lv": ["371"],
    "lt": ["370"],
    "fi": ["358"],
    "hu": ["36"],
    "sw": ["254", "255", "256", "257", "260", "263"],
    "yo": ["234", "229"],
    "et": ["372"],
    "el": ["30"],
    "vi": ["84"],
    "pl": ["48"],
    "bg": ["359"],
}


PHONE_RAW_REGEX = re.compile(
    r"""
    (?:
        (?:(?:\+|00)\d{1,3}[\s\-\.]?)?   
        (?:\(?\d{2,4}\)?[\s\-\.]?)?     
        (?:\d[\s\-\.]?){6,12}           
    )
    """,
    re.VERBOSE,
)


def looks_like_repeated(digits: str, threshold: float = 0.8) -> bool:
    if not digits:
        return True
    counts = {}
    for ch in digits:
        counts[ch] = counts.get(ch, 0) + 1
    max_freq = max(counts.values()) / len(digits)
    return max_freq >= threshold


def looks_like_sequential(digits: str) -> bool:
    if len(digits) < 4:
        return False
    inc = all((int(digits[i + 1]) - int(digits[i]) == 1) for i in range(len(digits) - 1))
    dec = all((int(digits[i]) - int(digits[i + 1]) == 1) for i in range(len(digits) - 1))
    return inc or dec


def _valid_range(s: str, lo: int, hi: int) -> bool:
    v = int(s)
    return lo <= v <= hi


def looks_like_date_digits(digits: str) -> bool:
    n = len(digits)

    if n == 8:
        y1 = digits[0:4]
        m1 = digits[4:6]
        d1 = digits[6:8]

        d2 = digits[0:2]
        m2 = digits[2:4]
        y2 = digits[4:8]

        m3 = digits[0:2]
        d3 = digits[2:4]
        y3 = digits[4:8]

        try:
            if _valid_range(y1, 1900, 2100) and _valid_range(m1, 1, 12) and _valid_range(d1, 1, 31):
                return True
            if _valid_range(d2, 1, 31) and _valid_range(m2, 1, 12) and _valid_range(y2, 1900, 2100):
                return True
            if _valid_range(m3, 1, 12) and _valid_range(d3, 1, 31) and _valid_range(y3, 1900, 2100):
                return True
        except ValueError:
            pass

    if n == 6:
        y1 = digits[0:2]
        m1 = digits[2:4]
        d1 = digits[4:6]

        d2 = digits[0:2]
        m2 = digits[2:4]
        y2 = digits[4:6]

        m3 = digits[0:2]
        d3 = digits[2:4]
        y3 = digits[4:6]

        try:
            if _valid_range(m1, 1, 12) and _valid_range(d1, 1, 31):
                return True
            if _valid_range(d2, 1, 31) and _valid_range(m2, 1, 12):
                return True
            if _valid_range(m3, 1, 12) and _valid_range(d3, 1, 31):
                return True
        except ValueError:
            pass

    return False


def matches_country_prefix(lang: str, digits: str) -> bool:
    codes = LANG_TO_COUNTRY_CODES.get(lang)
    if not codes:

        return True

    variants = [digits]
    if digits.startswith("00"):
        variants.append(digits[2:])
    if digits.startswith("0"):
        variants.append(digits[1:])

    for v in variants:
        for code in codes:
            if v.startswith(code):
                return True
    return False


def normalize_phone(raw: str, lang: Optional[str] = None) -> Optional[str]:

    cleaned = re.sub(r"[()\s\-\.]", "", raw)
    digits = re.sub(r"\D", "", cleaned)

    if len(digits) < 7 or len(digits) > 15:
        return None
    if looks_like_repeated(digits):
        return None
    if looks_like_sequential(digits):
        return None
    if looks_like_date_digits(digits):
        return None

    if lang is not None:
        if not matches_country_prefix(lang, digits):
            return None

    return cleaned


def extract_phones(text: str, lang: Optional[str] = None) -> List[str]:
    matches = PHONE_RAW_REGEX.findall(text)
    res = []
    for m in matches:
        p = normalize_phone(m, lang=lang)
        if p:
            res.append(p)
    return res


def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    phone_counts: Dict[str, Dict[str, int]] = {}

    jsonl_files = sorted(input_dir.glob("*_phone.jsonl"))
    print(f"Found {len(jsonl_files)} phone jsonl files.")

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


                code = PROMPT_LANG_TO_CODE.get(lang)
                if code:
                    prefix = f"+{code} "
                    text_for_extract = prefix + completion
                else:
                    text_for_extract = completion

                phones = extract_phones(text_for_extract, lang=lang)
                if not phones:
                    continue

                d = phone_counts.setdefault(lang, {})
                for p in phones:
                    d[p] = d.get(p, 0) + 1


    for lang, counts in phone_counts.items():
        out_path = output_dir / f"phones_{lang}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for pii, dup in sorted(counts.items(), key=lambda x: -x[1]):
                record = {
                    "language": lang,
                    "pii_type": "phone",
                    "pii": pii,
                    "duplicate": dup,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[phone] {lang}: {len(counts)} unique phones -> {out_path}")

    print("Done (phone).")


if __name__ == "__main__":
    main()
