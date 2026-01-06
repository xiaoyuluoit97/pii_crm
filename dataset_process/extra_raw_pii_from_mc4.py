import os
import re
import json
import hashlib
import unicodedata
from datasets import load_dataset
from tqdm import tqdm
import cld3

# ============ Config ============
OUTPUT_BASE_PATH_PHONE = "dataset/MLLM_MEM/PII/phone_raw"
OUTPUT_BOTH_PATH_URL   = "dataset/MLLM_MEM/PII/double_pii_raw"
OUTPUT_BASE_PATH_EMAIL = "dataset/MLLM_MEM/PII/email_raw"
OUTPUT_BASE_PATH_URL   = "dataset/MLLM_MEM/PII/url_raw"

SAMPLES_PER_LANGUAGE_BOTH = 5000    
SPLIT = "validation"
MAX_RECORDS_PER_LANG = None          
#SPLIT = "train"
os.makedirs(OUTPUT_BASE_PATH_PHONE, exist_ok=True)
os.makedirs(OUTPUT_BOTH_PATH_URL, exist_ok=True)
os.makedirs(OUTPUT_BASE_PATH_EMAIL, exist_ok=True)
os.makedirs(OUTPUT_BASE_PATH_URL, exist_ok=True)

# ============ Regex ============
URL_RE = re.compile(r"https?://[^\s<>'\"\]\)]+", re.UNICODE)
IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
DATE_LIKE_RE = re.compile(r"^\d{1,4}([\-./])\d{1,2}\1\d{1,4}$")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,5}", re.UNICODE)


EN_SIMPLE_PHONE_RE = re.compile(r"[0-9][0-9][0-9][-.()][0-9][0-9][0-9][-.()][0-9][0-9][0-9][0-9]")

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
    "ar": ["20", "212", "213", "218", "249", "962", "963", "964", "965", "966", "967", "968", "970", "971", "973", "974"],
    "he": ["972"],
    "ur": ["92", "971"],
    "af": ["27", "264", "268", "266", "267", "260"],
    "sv": ["46", "358", "47", "45"],
    "pt": ["351", "55", "258", "244", "238", "239", "290", "595"],
    "ro": ["40", "373"],
    "fr": ["33", "32", "41", "261", "596", "590", "687"],
    "ko": ["82"],
    "nl": ["31", "32", "599", "597"],
    "tr": ["90", "90"],
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
    "et": ["372"],   # Estonian - Estonia
    "el": ["30"],    # Greek - Greece
    "vi": ["84"],    # Vietnamese - Vietnam
    "pl": ["48"],    # Polish - Poland
    "bg": ["359"],   # Bulgarian - Bulgaria
}

def build_phone_re_for_lang(codes):
    if not codes:
        return None
    cc_group = "(?:" + "|".join(re.escape(cc) for cc in codes) + ")"
    pattern = rf"(?<!\w)\+{cc_group}(?:[ \t.\-()]*\d){{6,12}}(?!\w)"
    return re.compile(pattern)

PHONE_RE_BY_LANG = {lang: build_phone_re_for_lang(codes) for lang, codes in LANG_TO_COUNTRY_CODES.items()}

# ============ Utils ============
def get_hash(text: str) -> str:
    return hashlib.md5((text or "").strip().lower().encode("utf-8")).hexdigest()

def _base_normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", (text or "")).strip()

def spans_overlap(a, b):
    return not (a[1] <= b[0] or a[0] >= b[1])

def digit_count(s: str) -> int:
    return sum(ch.isdigit() for ch in s)

def strip_cc_and_separators(s: str, lang_code: str) -> str:
    if lang_code not in LANG_TO_COUNTRY_CODES:
        return s
    plus_digits = "+" + "".join(ch for ch in s if ch.isdigit())
    for cc in LANG_TO_COUNTRY_CODES[lang_code]:
        if plus_digits.startswith("+" + cc):
            cc_parts = r"\s*".join(list(cc))
            prefix = re.compile(rf"^\+\s*{cc_parts}", re.UNICODE)
            return prefix.sub("", s, count=1).strip()
    return s

def contains_url(text: str) -> bool:
    return bool(URL_RE.search(text or ""))

def contains_email(text: str) -> bool:
    return bool(EMAIL_RE.search(text or ""))

def contains_phone_candidate(text: str, lang_code: str) -> bool:

    url_spans = [(m.start(), m.end()) for m in URL_RE.finditer(text)]
    pre = EN_SIMPLE_PHONE_RE if lang_code == "en" else PHONE_RE_BY_LANG.get(lang_code)
    if not pre:
        return False
    for m in pre.finditer(text):
        span = (m.start(), m.end())
        if any(spans_overlap(span, us) for us in url_spans):
            continue
        s = m.group(0)
        if not (7 <= digit_count(s) <= 15):
            continue
        if IPV4_RE.search(s):
            continue
        tail = strip_cc_and_separators(s, lang_code) if lang_code != "en" else s
        if DATE_LIKE_RE.match(tail):
            parts = re.split(r"[-./]", tail)
            if any(p.isdigit() and 1900 <= int(p) <= 2099 for p in parts if len(p) == 4):
                continue
        return True
    return False

def _lang_ok(text: str, expected_lang: str, min_lang_ratio: float = 0.8, min_prob: float = 0.8) -> bool:
    try:
        langs = cld3.get_frequent_languages(text, 2)
    except Exception:
        return False
    if not langs:
        return False
    top = langs[0]
    if top.language != expected_lang:
        return False
    if getattr(top, "probability", 0.0) < min_prob:
        return False
    if getattr(top, "proportion", 0.0) < min_lang_ratio:
        return False
    return True

def check_clean_phone(text: str, expected_lang: str = "en") -> tuple[bool, str]:
    text = _base_normalize(text)
    if not text:
        return False, "empty"
    if not contains_phone_candidate(text, expected_lang):
        return False, "no_phone"
    if len(text) < 50:
        return False, "too_short"
    if re.search(r"[�]{2,}", text):
        return False, "garbled"
    if re.search(r"(.)\1{10,}", text):
        return False, "repeat_char"
    if not _lang_ok(text, expected_lang):
        return False, "lang_not_ok"
    return True, None

def check_clean_email(text: str, expected_lang: str = "en") -> tuple[bool, str]:
    text = _base_normalize(text)
    if not text:
        return False, "empty"
    if not contains_email(text):
        return False, "no_email"
    if len(text) < 50:
        return False, "too_short"
    if re.search(r"[�]{2,}", text):
        return False, "garbled"
    if re.search(r"(.)\1{10,}", text):
        return False, "repeat_char"
    if not _lang_ok(text, expected_lang):
        return False, "lang_not_ok"
    return True, None

def check_clean_url(text: str, expected_lang: str = "en") -> tuple[bool, str]:
    text = _base_normalize(text)
    if not text:
        return False, "empty"
    if not contains_url(text):
        return False, "no_url"
    if len(text) < 50:
        return False, "too_short"
    if re.search(r"[�]{2,}", text):
        return False, "garbled"
    if re.search(r"(.)\1{10,}", text):
        return False, "repeat_char"
    if not _lang_ok(text, expected_lang):
        return False, "lang_not_ok"
    return True, None

# ============ Main ============
def process_language(lang_code, lang_name, samples_limit_both):
    print(f"\n[{lang_code}] start")

    try:
        dataset = load_dataset("allenai/c4", lang_code, streaming=True, split=SPLIT)
    except Exception as e:
        print(f"[{lang_code}] load failed: {e}")
        return

    seen_hashes_phone = set()
    seen_hashes_both  = set()
    seen_hashes_email = set()
    seen_hashes_url   = set()
    selected_samples_phone = []
    selected_samples_both  = []
    selected_samples_email = []
    selected_samples_url   = []

    subdir = "test" if SPLIT == "validation" else SPLIT
    os.makedirs(os.path.join(OUTPUT_BASE_PATH_PHONE, subdir), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_BOTH_PATH_URL, subdir), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_BASE_PATH_EMAIL, subdir), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_BASE_PATH_URL,   subdir), exist_ok=True)

    phone_save_path = os.path.join(OUTPUT_BASE_PATH_PHONE, subdir, f"{lang_code}.jsonl")
    both_save_path  = os.path.join(OUTPUT_BOTH_PATH_URL,   subdir, f"{lang_code}.jsonl")
    email_save_path = os.path.join(OUTPUT_BASE_PATH_EMAIL, subdir, f"{lang_code}.jsonl")
    url_save_path   = os.path.join(OUTPUT_BASE_PATH_URL,   subdir, f"{lang_code}.jsonl")

    processed = 0
    pbar = tqdm(total=samples_limit_both, desc=f"📞 {lang_code}", leave=False)  # 只跟踪 double PII 进度

    for sample in dataset:

        if len(selected_samples_both) >= samples_limit_both:
            break

        processed += 1
        if MAX_RECORDS_PER_LANG is not None and processed > MAX_RECORDS_PER_LANG:
            break

        text = sample.get("text", "")
        text_norm = _base_normalize(text)
        if not text_norm:
            continue
        h = get_hash(text_norm)


        ok_p, _ = check_clean_phone(text_norm, expected_lang=lang_code)
        if not ok_p:
            pass
        else:
            if h not in seen_hashes_phone:
                selected_samples_phone.append(text_norm)
                seen_hashes_phone.add(h)


            if len(selected_samples_both) < samples_limit_both and contains_email(text_norm):
                if h not in seen_hashes_both:
                    selected_samples_both.append(text_norm)
                    seen_hashes_both.add(h)
                    pbar.update(1)


        ok_e, _ = check_clean_email(text_norm, expected_lang=lang_code)
        if ok_e and h not in seen_hashes_email:
            selected_samples_email.append(text_norm)
            seen_hashes_email.add(h)


        ok_u, _ = check_clean_url(text_norm, expected_lang=lang_code)
        if ok_u and h not in seen_hashes_url:
            selected_samples_url.append(text_norm)
            seen_hashes_url.add(h)

    pbar.close()


    with open(phone_save_path, "w", encoding="utf-8") as f:
        for record in selected_samples_phone:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(both_save_path, "w", encoding="utf-8") as f:
        for record in selected_samples_both:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(email_save_path, "w", encoding="utf-8") as f:
        for record in selected_samples_email:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(url_save_path, "w", encoding="utf-8") as f:
        for record in selected_samples_url:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"[{lang_code}] phone_raw={len(selected_samples_phone)} | "
        f"double_pii_raw={len(selected_samples_both)} | "
        f"email_raw={len(selected_samples_email)} | "
        f"url_raw={len(selected_samples_url)} | "
        f"saved: {phone_save_path} | {both_save_path} | {email_save_path} | {url_save_path}"
    )

def main(languages: dict, samples_limit_both: int = SAMPLES_PER_LANGUAGE_BOTH):
    for lang_code, lang_name in languages.items():
        process_language(lang_code, lang_name, samples_limit_both)

if __name__ == "__main__":
#     languages = {
#    "zh": "Chinese",
#    "th": "Thai",
#    "da": "Danish",
#    "de": "German",
#    "es": "Spanish",
#    "it": "Italian",
#    "hi": "Hindi",
#    "en": "English",
#    "fr": "French",
#    "nl": "Dutch",
#    "pt": "Portuguese",
#    "ru": "Russian",
#    "uk": "Ukrainian",
#    "be": "Belarusian",
#    "ar": "Arabic",
#    "he": "Hebrew",
#    "af": "Afrikaans",
#    "ur": "Urdu",
#    "ro": "Romanian",
#    "sv": "Swedish",
#   "ro": "Romanian",
#    "ko": "Korean",
#    "tr": "Turkish",
#    "tk": "Turkmen",
#    "az": "Azerbaijani",
#    "ta": "Tamil",
#    "te": "Telugu",
#    "ml": "Malayalam",
#    "lv": "Latvian",
#    "lt": "Lithuanian",
#    "fi": "Finnish",
#    "hu": "Hungarian",
#    "sw": "Swahili",
#    "yo": "Yoruba",
#  }

    main(languages)
