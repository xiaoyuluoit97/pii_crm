import os
import glob
import json
import unicodedata
from typing import Tuple, Set


# ----------------------------
# Normalization + LCS-substring
# ----------------------------

# Common dash variants you may see in scraped text:
DASH_CHARS = {
    "-",        # hyphen-minus
    "‐", "-",   # hyphen variants
    "–", "—",   # en/em dash
    "−",        # minus sign
}

PLUS_CHARS = {"+", "＋"}  # ASCII plus + fullwidth plus

def normalize_for_phone(s: str) -> str:
    """
    Phone normalization (prefix prompt and target suffix):
      - Unicode NFKC
      - remove all whitespace characters
      - remove '+' and '-' (incl. common dash variants)
    Keep everything else (e.g., digits, parentheses, dots), as requested.
    """
    s = unicodedata.normalize("NFKC", s)

    out = []
    for ch in s:
        # drop whitespace (space/newline/tab/other unicode spaces)
        if ch.isspace():
            continue
        # drop plus / dash variants
        if ch in PLUS_CHARS or ch in DASH_CHARS:
            continue
        out.append(ch)
    return "".join(out)


def lcs_substring_len(a: str, b: str) -> int:
    """
    Longest Common Substring length (contiguous), rolling DP for memory efficiency.
    """
    if not a or not b:
        return 0
    if len(b) > len(a):
        a, b = b, a

    prev = [0] * (len(b) + 1)
    curr = [0] * (len(b) + 1)
    best = 0

    for i in range(1, len(a) + 1):
        ai = a[i - 1]
        curr[0] = 0
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best:
                    best = curr[j]
            else:
                curr[j] = 0
        prev, curr = curr, prev

    return best


def compute_phone_overlap(prompt: str, phone: str) -> float:
    """
    r_phone = LCS_substring_len(norm(phone), norm(prompt)) / len(norm(phone))

    norm(prompt): NFKC + remove whitespace + remove '+' and '-'
    norm(phone):  NFKC + remove whitespace + remove '+' and '-'
    """
    norm_prompt = normalize_for_phone(prompt)
    norm_phone = normalize_for_phone(phone)

    if not norm_phone:
        return 0.0

    l = lcs_substring_len(norm_phone, norm_prompt)
    return l / len(norm_phone)


# ----------------------------
# Hit matching: (prompt_suffix, normalized_phone)
# ----------------------------

PROMPT_SUFFIX_N = 250  # adjust if needed

def canon_prompt_suffix(prompt: str, n: int = PROMPT_SUFFIX_N) -> str:
    """
    prompt: NFKC + collapse whitespace + take last n chars
    (Used only for robust prompt matching across truncation windows.)
    """
    p = unicodedata.normalize("NFKC", prompt)
    p = " ".join(p.split())
    return p if n <= 0 else p[-n:]


def load_hit_pairs(hit_file: str, max_samples: int = 2000) -> Set[Tuple[str, str]]:
    """
    Read up to max_samples lines from hits jsonl, return a set of:
      (prompt_suffix, normalized_phone)
    Assumes each line has 'prompt' and 'target' (phone) fields.
    """
    hit_pairs: Set[Tuple[str, str]] = set()
    if not os.path.exists(hit_file):
        print(f"[WARN] hit file not found: {hit_file}")
        return hit_pairs

    with open(hit_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            prompt = obj.get("prompt", "")
            phone = obj.get("target", "")

            pkey = canon_prompt_suffix(prompt, PROMPT_SUFFIX_N)
            pnorm = normalize_for_phone(phone)
            hit_pairs.add((pkey, pnorm))

    return hit_pairs


# ----------------------------
# Main pipeline
# ----------------------------

BASE_DATA_DIR = "dataset/MLLM_MEM/PII/new_verbatim/phone"
HITS_DIR = "mllm_pii_memorization/mem_pii/mem_result_2000/phone/exact_mem_sample"
OUTPUT_DIR = "dataset/MLLM_MEM/PII/new_verbatim/phone_with_overlap_1B"

MAX_PER_FILE = 2000


def process_language_file(input_path: str):
    """
    For each *_100.jsonl file:
      - read up to MAX_PER_FILE
      - compute r_phone
      - mark hit via (prompt_suffix, normalized_phone)
      - write to OUTPUT_DIR
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    basename = os.path.basename(input_path)
    lang = basename.split("_")[0]
    hits_file = os.path.join(HITS_DIR, f"hits_mGPT_phone_{lang}_100.jsonl")

    print(f"[INFO] Processing lang={lang} input={basename}")
    print(f"[INFO] Using hits file: {hits_file}")
    print(f"[INFO] Matching rule: (prompt_suffix_last_{PROMPT_SUFFIX_N}_chars, phone_norm_remove_space_plus_dash)")

    hit_pairs = load_hit_pairs(hits_file, max_samples=MAX_PER_FILE)
    print(f"[INFO] Loaded hit_pairs: {len(hit_pairs)}")

    output_path = os.path.join(OUTPUT_DIR, f"{lang}_100_overlap.jsonl")

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for idx, line in enumerate(fin):
            if idx >= MAX_PER_FILE:
                break
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            prompt = obj.get("prompt", "")
            phone = obj.get("phone", "")  # base data field: 'phone'

            r_phone = compute_phone_overlap(prompt, phone)

            # hit matching
            pkey = canon_prompt_suffix(prompt, PROMPT_SUFFIX_N)
            pnorm = normalize_for_phone(phone)
            hit_flag = 1 if (pkey, pnorm) in hit_pairs else 0

            obj["r_phone"] = r_phone
            obj["hit"] = hit_flag

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[INFO] Done lang={lang}, wrote to {output_path}")


def main():
    pattern = os.path.join(BASE_DATA_DIR, "*_100.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[WARN] No *_100.jsonl found in {BASE_DATA_DIR}")
        return

    for path in files:
        process_language_file(path)


if __name__ == "__main__":
    main()
