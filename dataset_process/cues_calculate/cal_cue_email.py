import os
import glob
import json
import unicodedata
from typing import Tuple, Set


# ----------------------------
# Normalization + LCS
# ----------------------------

def norm_alnum_nfkc(s: str) -> str:
    """
    NFKC + lower + keep only Unicode letters/digits.
    Works for Latin (incl. diacritics), Chinese, Vietnamese, etc.
    """
    s = unicodedata.normalize("NFKC", s).lower()
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            out.append(ch)
    return "".join(out)


def norm_digits_nfkc(s: str) -> str:
    """NFKC + keep only digits (for phone-number style matching if needed)."""
    s = unicodedata.normalize("NFKC", s)
    return "".join(ch for ch in s if ch.isdigit())


def lcs_substring_len(a: str, b: str) -> int:
    """
    Longest Common Substring length (contiguous).
    Rolling DP to save memory.
    """
    if not a or not b:
        return 0
    # Make b the shorter one for smaller DP buffer
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


def lcs_ratio(suffix: str, prefix: str, *, norm_fn=norm_alnum_nfkc) -> float:
    """
    c(suffix, prefix) = LCS(norm(suffix), norm(prefix)) / |norm(suffix)|
    Directional: measures what fraction of the TARGET SUFFIX is revealed by the PREFIX.
    """
    ns = norm_fn(suffix)
    if not ns:
        return 0.0
    np = norm_fn(prefix)
    l = lcs_substring_len(ns, np)
    return l / len(ns)


# ----------------------------
# Domain: remove TLD
# ----------------------------

# Optional: extend if you see lots of 2-part public suffixes in your data.
COMMON_2PART_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk",
    "com.au", "net.au", "org.au",
    "co.za", "org.za",
}

def strip_tld(domain: str) -> str:
    """
    Remove top-level domain (TLD) for overlap computation.

    Examples:
      brianvisagie.com     -> brianvisagie
      foo.bar.co.uk        -> foo.bar   (drops 'co.uk' if in COMMON_2PART_SUFFIXES)
      example              -> example
    """
    d = unicodedata.normalize("NFKC", domain).strip().lower().strip(".")
    parts = [p for p in d.split(".") if p]
    if len(parts) <= 1:
        return d

    last2 = ".".join(parts[-2:])
    if last2 in COMMON_2PART_SUFFIXES and len(parts) >= 3:
        return ".".join(parts[:-2])
    return ".".join(parts[:-1])


# ----------------------------
# Email overlap: local/domain + length-weighted overall
# ----------------------------

def compute_email_overlap(prompt: str, local: str, domain_no_tld: str) -> Tuple[float, float, float]:
    """
    Returns (c_local, c_domain, c_overall)

    c_local  = LCS(norm(local),  norm(prompt)) / |norm(local)|
    c_domain = LCS(norm(domain_no_tld), norm(prompt)) / |norm(domain_no_tld)|

    overall = (|norm(local)| * c_local + |norm(domain_no_tld)| * c_domain) / (|norm(local)| + |norm(domain_no_tld)|)
    """
    c_local = lcs_ratio(local, prompt, norm_fn=norm_alnum_nfkc)
    c_domain = lcs_ratio(domain_no_tld, prompt, norm_fn=norm_alnum_nfkc)

    nl = norm_alnum_nfkc(local)
    nd = norm_alnum_nfkc(domain_no_tld)
    denom = len(nl) + len(nd)
    if denom == 0:
        c_overall = 0.0
    else:
        c_overall = (len(nl) * c_local + len(nd) * c_domain) / denom

    return c_local, c_domain, c_overall


# ----------------------------
# Hit matching (optional)
# ----------------------------

PROMPT_SUFFIX_N = 250  # adjust if needed


def canon_email(s: str) -> str:
    """NFKC + strip + lower."""
    return unicodedata.normalize("NFKC", s).strip().lower()


def canon_prompt_suffix(prompt: str, n: int = PROMPT_SUFFIX_N) -> str:
    """
    NFKC + collapse whitespace + take last n chars.
    Used to robustly match prompts across different truncation windows.
    """
    p = unicodedata.normalize("NFKC", prompt)
    p = " ".join(p.split())
    return p if n <= 0 else p[-n:]


def load_hit_pairs(hit_file: str, max_samples: int) -> Set[Tuple[str, str]]:
    """
    Read up to max_samples lines from a JSONL hit file.
    Each line must contain 'prompt' and 'target' fields.
    Returns a set of (prompt_suffix, email).
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
            target = obj.get("target", "")
            hit_pairs.add((canon_prompt_suffix(prompt, PROMPT_SUFFIX_N), canon_email(target)))

    return hit_pairs


# ----------------------------
# Main pipeline
# ----------------------------

BASE_DATA_DIR = "dataset/MLLM_MEM/PII/new_verbatim_mgpt600m/email"
HITS_DIR = "mllm_pii_memorization/mem_pii/mem_result_2000/email/exact_mem_sample"
OUTPUT_DIR = "dataset/MLLM_MEM/PII/new_verbatim/email_with_overlap_13B"

MAX_PER_FILE = 2000
ALL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "all")
ALL_OUTPUT_FILE = os.path.join(ALL_OUTPUT_DIR, "all_100_overlap.jsonl")


def process_language_file(input_path: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ALL_OUTPUT_DIR, exist_ok=True)

    basename = os.path.basename(input_path)           # e.g. "zh_100.jsonl"
    lang = basename.split("_")[0]
    hits_file = os.path.join(HITS_DIR, f"hits_mGPT-13B_email_{lang}_100.jsonl")

    print(f"[INFO] Processing lang={lang} input={basename}")
    print(f"[INFO] Using hits file: {hits_file}")
    print(f"[INFO] Matching rule: (prompt_suffix_last_{PROMPT_SUFFIX_N}_chars, email)")

    hit_pairs = load_hit_pairs(hits_file, max_samples=MAX_PER_FILE)
    print(f"[INFO] Loaded hit_pairs: {len(hit_pairs)}")

    # 单语言输出
    output_path = os.path.join(OUTPUT_DIR, f"{lang}_100_overlap.jsonl")

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout_lang, \
         open(ALL_OUTPUT_FILE, "a", encoding="utf-8") as fout_all:

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
            email_target = obj.get("email", "")

            local, domain = "", ""
            if isinstance(email_target, str) and "@" in email_target:
                local, domain = email_target.split("@", 1)

            domain_no_tld = strip_tld(domain) if domain else ""

            # overlap
            c_local, c_domain, c_email = compute_email_overlap(
                prompt, local, domain_no_tld
            )

            # hit
            pkey = canon_prompt_suffix(prompt, PROMPT_SUFFIX_N)
            hit_flag = 1 if (pkey, canon_email(email_target)) in hit_pairs else 0

            obj["c_local"] = c_local
            obj["c_domain"] = c_domain
            obj["c_email"] = c_email
            obj["hit"] = hit_flag
            obj["lang"] = lang   

            line_out = json.dumps(obj, ensure_ascii=False) + "\n"
            fout_lang.write(line_out)
            fout_all.write(line_out)

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
