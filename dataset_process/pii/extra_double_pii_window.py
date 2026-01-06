import os
import re
import json
import unicodedata
from tqdm import tqdm

# ============ Config ============
INPUT_BASE = "dataset/MLLM_MEM/PII/double_pii_raw/train"
OUTPUT_BASE = "dataset/MLLM_MEM/PII/double_pii_windows/train"


LANGS = ["et"]
LEFT_CTX_CHARS  = 200   # 前文窗口
RIGHT_CTX_CHARS = 100   # 后文窗口

os.makedirs(OUTPUT_BASE, exist_ok=True)

# ============ Regex ============
URL_RE = re.compile(r"https?://[^\s<>'\"\]\)]+", re.UNICODE)
IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
DATE_LIKE_RE = re.compile(r"^\d{1,4}([\-./])\d{1,2}\1\d{1,4}$")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", re.UNICODE)
EN_SIMPLE_PHONE_RE = re.compile(
    r"[0-9][0-9][0-9][-.()][0-9][0-9][0-9][-.()][0-9][0-9][0-9][0-9]"
)

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
def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", (text or "")).strip()

def spans_overlap(a, b):
    return not (a[1] <= b[0] or a[0] >= b[1])

def digit_count(s: str) -> int:
    return sum(ch.isdigit() for ch in s)

def strip_cc_for_date_check(s: str, lang_code: str) -> str:
    if lang_code not in LANG_TO_COUNTRY_CODES:
        return s
    plus_digits = "+" + "".join(ch for ch in s if ch.isdigit())
    for cc in LANG_TO_COUNTRY_CODES[lang_code]:
        if plus_digits.startswith("+" + cc):
            cc_parts = r"\s*".join(list(cc))
            prefix = re.compile(rf"^\+\s*{cc_parts}", re.UNICODE)
            return prefix.sub("", s, count=1).strip()
    return s

def find_all_emails(text: str):
    if not text:
        return []
    url_spans = [(m.start(), m.end()) for m in URL_RE.finditer(text)]
    spans = []
    for m in EMAIL_RE.finditer(text):
        span = (m.start(), m.end())
        if any(spans_overlap(span, us) for us in url_spans):
            continue
        spans.append((m.start(), m.end(), m.group(0)))
    return spans

def find_all_phones(text: str, lang_code: str):
    if not text:
        return []
    url_spans = [(m.start(), m.end()) for m in URL_RE.finditer(text)]
    pre = EN_SIMPLE_PHONE_RE if lang_code == "en" else PHONE_RE_BY_LANG.get(lang_code)
    if not pre:
        return []
    spans = []
    for m in pre.finditer(text):
        span = (m.start(), m.end())
        if any(spans_overlap(span, us) for us in url_spans):
            continue
        s = m.group(0)
        if not (7 <= digit_count(s) <= 15):
            continue
        if IPV4_RE.search(s):
            continue
        tail = strip_cc_for_date_check(s, lang_code) if lang_code != "en" else s
        if DATE_LIKE_RE.match(tail):
            parts = re.split(r"[-./]", tail)
            if any(p.isdigit() and 1900 <= int(p) <= 2099 for p in parts if len(p) == 4):
                continue
        spans.append((m.start(), m.end(), s))
    return spans

def phone_norm_key(s: str) -> str:
    plus = "+" if s.strip().startswith("+") else ""
    digits = "".join(ch for ch in s if ch.isdigit())
    return plus + digits

def choose_closest_pair(email_spans, phone_spans):

    best = None
    best_dist = None
    for e in email_spans:
        for p in phone_spans:
            e_s, e_e, _ = e
            p_s, p_e, _ = p
            if e_e <= p_s:
                dist = p_s - e_e
            elif p_e <= e_s:
                dist = e_s - p_e
            else:
                dist = 0  # overlap
            if best is None or dist < best_dist:
                best = (e, p)
                best_dist = dist
    return best

def extract_window(text: str, a_span, b_span, L: int, R: int):

    a_s, a_e, _ = a_span
    b_s, b_e, _ = b_span
    left_idx  = min(a_s, b_s)
    right_idx = max(a_e, b_e)
    start = max(0, left_idx - L)
    end   = min(len(text), right_idx + R)
    window = text[start:end]
    between_start = min(a_e, b_e)
    between_end   = max(a_s, b_s)
    between = text[between_start:between_end] if between_start < between_end else ""
    return {
        "window": window,
        "window_start": start,
        "window_end": end,
        "between": between,
        "between_start": between_start,
        "between_end": between_end,
        "first_span_start": left_idx,
        "last_span_end": right_idx,
    }

# ============ Main ============
def process_language(lang_code: str):
    in_path = os.path.join(INPUT_BASE, f"{lang_code}.jsonl")
    if not os.path.exists(in_path):
        print(f"[{lang_code}] missing: {in_path}")
        return

    out_dir = os.path.join(OUTPUT_BASE)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{lang_code}.jsonl")

    seen_emails = set()
    seen_phones = set()

    total_lines = sum(1 for _ in open(in_path, "r", encoding="utf-8"))
    kept = 0
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        pbar = tqdm(fin, total=total_lines, desc=f"🧹 {lang_code}")
        for line in pbar:
            try:
                text = json.loads(line)
                if not isinstance(text, str):
                    continue
            except Exception:
                continue

            text = _norm(text)
            if not text:
                continue

            emails = find_all_emails(text)
            phones = find_all_phones(text, lang_code)

            if not emails or not phones:
                continue

            pair = choose_closest_pair(emails, phones)
            if not pair:
                continue
            e_span, p_span = pair
            email_text = e_span[2].lower()
            phone_text = p_span[2]
            phone_key = phone_norm_key(phone_text)

            if email_text in seen_emails or phone_key in seen_phones:
                continue

            seen_emails.add(email_text)
            seen_phones.add(phone_key)

            win = extract_window(text, e_span, p_span, LEFT_CTX_CHARS, RIGHT_CTX_CHARS)

            record = {
                "lang": lang_code,
                "email": email_text,         
                "phone": phone_text,         
                "phone_norm": phone_key,
                "window": win["window"],
                "between": win["between"],
                "positions": {
                    "email_start": e_span[0], "email_end": e_span[1],
                    "phone_start": p_span[0], "phone_end": p_span[1],
                    "window_start": win["window_start"], "window_end": win["window_end"],
                    "between_start": win["between_start"], "between_end": win["between_end"],
                },
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

        pbar.close()

    print(f"[{lang_code}] kept={kept} -> {out_path}")

def main():
    for lang in LANGS:
        process_language(lang)

if __name__ == "__main__":
    main()
