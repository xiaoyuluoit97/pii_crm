import os
import re
import json
from tqdm import tqdm
from transformers import AutoTokenizer

# ---------- Paths & Settings ----------
INPUT_DIR  = "dataset/MLLM_MEM/PII/phone_raw/train"
OUTPUT_DIR = "dataset/MLLM_MEM/PII/phone_prompts_mGPT/train"
LANGS = ["en"]

PROMPT_TOKEN_LENS = [100]

# Sampling cap (controlled independently per K; None means no limit)
MAX_SAMPLES_PER_K = {50: None, 100: None, 200: None}

# Batch size for tokenization (larger = faster but more memory usage)
BATCH_SIZE = 5000

# Thresholds for "numeric-heavy" text and large-volume hits
MAX_DIGIT_RATIO = 0.4           # Maximum ratio of digit characters in text
MAX_LONG_NUM_GROUPS = 2         # Max number of continuous digit groups (≥5 digits)
MAX_PHONE_SPANS_PER_TEXT = 2    # Max allowed phone matches per text

# New: minimum local-digit length to be considered a leakage
MIN_LOCAL_DIGITS_FOR_LEAK = 5

# Local HuggingFace model base directory
MODEL_BASE_DIR = "model"
MODEL_ID = "ai-forever/mGPT"

# Deduplication strategy
# Default: deduplicate only within the same K
# If set to True, deduplicate globally across all K
DEDUP_ACROSS_K = False

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ.setdefault("TOKENIZER_PARALLELISM", "false")

# ---------- Regex Patterns & Language Mapping ----------
URL_RE = re.compile(r"https?://[^\s<>'\"\]\)]+", re.UNICODE)
IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
DATE_LIKE_RE = re.compile(r"^\d{1,4}([\-./])\d{1,2}\1\d{1,4}$")

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
    # Require +[country code], allow separators; total length validated later
    pattern = rf"(?<!\w)\+{cc_group}(?:[ \t.\-()]*\d){{6,12}}(?!\w)"
    return re.compile(pattern)

PHONE_RE_BY_LANG = {
    lang: build_phone_re_for_lang(codes)
    for lang, codes in LANG_TO_COUNTRY_CODES.items()
}

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

# Normalization key for deduplication:
# keep digits only; preserve a leading '+' if present
def normalize_phone_for_dedup(s: str) -> str:
    digits = re.sub(r"\D+", "", s)
    has_plus = s.strip().startswith("+")
    return ("+" if has_plus else "") + digits

# Numeric-density detection
LONG_NUM_RE = re.compile(r"\d{5,}")

def is_numeric_heavy(text: str) -> bool:
    if not text:
        return False
    total = len(text)
    digits = sum(ch.isdigit() for ch in text)
    ratio = digits / max(1, total)
    if ratio >= MAX_DIGIT_RATIO:
        return True
    long_groups = len(LONG_NUM_RE.findall(text))
    if long_groups >= MAX_LONG_NUM_GROUPS:
        return True
    return False

# ---------- Local digit leakage detection utilities ----------
def normalize_digits(s: str) -> str:
    return re.sub(r"\D+", "", s)

def get_lang_codes_sorted(lang_code: str):
    codes = LANG_TO_COUNTRY_CODES.get(lang_code, [])
    # Prefer longer country codes to avoid ambiguity (e.g., 39 vs 3906)
    return sorted(codes, key=len, reverse=True)

def extract_local_digits(phone: str, lang_code: str) -> str:
    """
    Extract local digits (country code removed) from a phone number.
    If country code cannot be identified, return all digits.
    """
    digits = normalize_digits(phone)
    for cc in get_lang_codes_sorted(lang_code):
        if digits.startswith(cc):
            return digits[len(cc):]
    return digits

def build_sep_tolerant_regex(digits: str) -> re.Pattern:
    """
    Build a regex allowing arbitrary separators between digits.
    Separators include whitespace, newline, -, ., (, ).
    Uses (?<!\d) and (?!\d) to avoid being embedded in longer digit strings.
    """
    if not digits:
        return None
    body = "".join(re.escape(d) + r"[\s\-\.\(\)]*" for d in digits)
    body = body.rstrip(r"[\s\-\.\(\)]*")
    pattern = rf"(?<!\d){body}(?!\d)"
    return re.compile(pattern, re.UNICODE)

def prompt_has_local_digits(prompt_text: str, phone: str, lang_code: str) -> bool:
    local_digits = extract_local_digits(phone, lang_code)
    if len(local_digits) < MIN_LOCAL_DIGITS_FOR_LEAK:
        return False
    rx = build_sep_tolerant_regex(local_digits)
    return bool(rx.search(prompt_text)) if rx else False

# ---------- Phone span iterator ----------
def iter_phone_spans(text: str, lang_code: str):
    if not text:
        return
    url_spans = [(m.start(), m.end()) for m in URL_RE.finditer(text)]
    pre = EN_SIMPLE_PHONE_RE if lang_code == "en" else PHONE_RE_BY_LANG.get(lang_code)
    if not pre:
        return
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
        yield m.start(), m.end(), s

def find_token_index_for_char(offsets, char_pos: int) -> int:
    last_valid = 0
    for i, (s, e) in enumerate(offsets):
        if s <= char_pos < e:
            return i
        if s <= char_pos:
            last_valid = i
    return last_valid

def resolve_local_repo_path(model_base_dir: str, repo_id: str):
    parts = repo_id.split("/")
    local_path = os.path.join(model_base_dir, *parts)
    return local_path if os.path.isdir(local_path) else None

def load_tokenizer_with_local_fallback(repo_id: str, base_dir: str):
    local_repo_path = resolve_local_repo_path(base_dir, repo_id)
    if local_repo_path:
        try:
            return AutoTokenizer.from_pretrained(local_repo_path, use_fast=True)
        except Exception as e:
            print(f"⚠️ Failed to load tokenizer from local path '{local_repo_path}': {e}")
    try:
        return AutoTokenizer.from_pretrained(
            repo_id, use_fast=True, cache_dir=base_dir, local_files_only=True
        )
    except Exception as e:
        print(f"ℹ️ Local cache-only load failed (may not exist in cache): {e}")
    print("🌐 Downloading tokenizer from Hugging Face and caching to:", base_dir)
    return AutoTokenizer.from_pretrained(
        repo_id, use_fast=True, cache_dir=base_dir, local_files_only=False
    )

tokenizer = load_tokenizer_with_local_fallback(MODEL_ID, MODEL_BASE_DIR)
ADD_SPECIAL_TOKENS = False


def all_limits_reached(counters):
    for K, limit in MAX_SAMPLES_PER_K.items():
        if limit is not None and counters.get(K, 0) < limit:
            return False
    return True

def process_lang(lang_code: str):
    in_path = os.path.join(INPUT_DIR, f"{lang_code}.jsonl")
    if not os.path.exists(in_path):
        print(f"❌ Missing input file: {in_path}")
        return

    total_lines = sum(1 for _ in open(in_path, "r", encoding="utf-8"))
    print(f"\n🌍 {lang_code} | Reading: {in_path}  (lines: {total_lines:,})")

    out_files = {K: open(os.path.join(OUTPUT_DIR, f"{lang_code}_{K}.jsonl"), "w", encoding="utf-8")
                 for K in PROMPT_TOKEN_LENS}

    skipped_due_to_phone_in_prompt = {K: 0 for K in PROMPT_TOKEN_LENS}
    skipped_due_to_local_norm_in_prompt = {K: 0 for K in PROMPT_TOKEN_LENS}  
    skipped_due_to_duplicate = {K: 0 for K in PROMPT_TOKEN_LENS}
    collected_per_K = {K: 0 for K in PROMPT_TOKEN_LENS}

    seen_phones_per_K = {K: set() for K in PROMPT_TOKEN_LENS}
    seen_phones_global = set() if DEDUP_ACROSS_K else None

    with open(in_path, "r", encoding="utf-8") as fin:
        pbar = tqdm(total=total_lines, desc=f"✂️ {lang_code}")
        batch_raw = []
        batch_candidates = []

        def flush_batch():
            if not batch_raw:
                return
            enc = tokenizer(
                batch_raw,
                add_special_tokens=ADD_SPECIAL_TOKENS,
                return_offsets_mapping=True,
                padding=False,
                truncation=False
            )
            stop_all = False
            for text_str, spans, offsets in zip(batch_raw, batch_candidates, enc["offset_mapping"]):
                if all_limits_reached(collected_per_K):
                    stop_all = True
                    break
                if not spans:
                    continue

                for (start_char, end_char, phone) in spans:
                    tok_idx = find_token_index_for_char(offsets, start_char)
                    phone_key = normalize_phone_for_dedup(phone)

                    for K in PROMPT_TOKEN_LENS:
                        max_cap = MAX_SAMPLES_PER_K.get(K)
                        if max_cap is not None and collected_per_K[K] >= max_cap:
                            continue

                        if DEDUP_ACROSS_K:
                            if phone_key in seen_phones_global:
                                skipped_due_to_duplicate[K] += 1
                                continue
                        else:
                            if phone_key in seen_phones_per_K[K]:
                                skipped_due_to_duplicate[K] += 1
                                continue

                        left_tok = max(0, tok_idx - K)
                        if tok_idx - left_tok < K:
                            continue

                        prompt_begin_char = offsets[left_tok][0]
                        prompt_end_char = start_char
                        if prompt_begin_char is None or prompt_end_char is None:
                            continue
                        if not (0 <= prompt_begin_char <= prompt_end_char <= len(text_str)):
                            continue

                        prompt_text = text_str[prompt_begin_char:prompt_end_char]

                        if phone in prompt_text:
                            skipped_due_to_phone_in_prompt[K] += 1
                            continue
                        if prompt_has_local_digits(prompt_text, phone, lang_code):
                            skipped_due_to_local_norm_in_prompt[K] += 1
                            continue

                        original_text = text_str[prompt_begin_char:end_char]
                        record = {"phone": phone, "prompt": prompt_text, "original": original_text}
                        out_files[K].write(json.dumps(record, ensure_ascii=False) + "\n")
                        collected_per_K[K] += 1

                        if DEDUP_ACROSS_K:
                            seen_phones_global.add(phone_key)
                        else:
                            seen_phones_per_K[K].add(phone_key)

                        if all_limits_reached(collected_per_K):
                            stop_all = True
                            break
                    if stop_all:
                        break
                if stop_all:
                    break

            batch_raw.clear()
            batch_candidates.clear()

        processed_lines = 0
        for line in fin:
            processed_lines += 1
            pbar.update(1)
            if all_limits_reached(collected_per_K):
                break

            try:
                text = json.loads(line)
                if not isinstance(text, str) or not text.strip():
                    continue
            except Exception:
                continue

            text_str = text.strip()

            if is_numeric_heavy(text_str):
                continue

            spans = list(iter_phone_spans(text_str, lang_code))
            if not spans:
                continue

            if len(spans) > MAX_PHONE_SPANS_PER_TEXT:
                continue

            batch_raw.append(text_str)
            batch_candidates.append(spans)

            if len(batch_raw) >= BATCH_SIZE:
                flush_batch()

        if not all_limits_reached(collected_per_K):
            flush_batch()
        pbar.close()

    for f in out_files.values():
        f.close()

    print("✅ Saved:", [os.path.join(OUTPUT_DIR, f"{lang_code}_{K}.jsonl") for K in PROMPT_TOKEN_LENS])
    for K in PROMPT_TOKEN_LENS:
        cap = MAX_SAMPLES_PER_K.get(K)
        cap_str = f"{collected_per_K[K]}/{cap}" if cap is not None else f"{collected_per_K[K]}"
        print(
            f"ℹ️ Collected for K={K}: {cap_str} | "
            f"Skipped(leak_exact_phone)={skipped_due_to_phone_in_prompt[K]} | "
            f"Skipped(leak_local_norm)={skipped_due_to_local_norm_in_prompt[K]} | "
            f"Skipped(duplicate)={skipped_due_to_duplicate[K]}"
        )

def main():
    for lang in LANGS:
        process_lang(lang)

if __name__ == "__main__":
    main()
