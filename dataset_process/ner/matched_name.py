import os
import re
import json
import unicodedata
from typing import List
from tqdm import tqdm

# ===== Paths =====
INPUT_BASE  = "dataset/MLLM_MEM/PII/email_windows_qwen/test"
OUTPUT_BASE = "dataset/MLLM_MEM/PII/email_windows_qwen_onemapone/test"

LANGS = [
    "pl", "tr", "pt", "af", "ru", "fr", "sw", "vi", "es", "ta",
    "az", "hu", "it", "en", "be", "sv", "lt", "de", "da", "ar", 
    "fi", "zh", "uk", "lv", "hi", "nl", "ro", "bg", "ko", "el", 
    "th", "ml"
]
os.makedirs(OUTPUT_BASE, exist_ok=True)

# --- utils ---
SPLIT_NAME_RE = re.compile(r"[ \t\u00A0\u2009\u202F\-·'’\.]+") 
MIN_UNIT_LEN = 3  

def fold(s: str) -> str:

    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()

def email_local_norm(email: str) -> str:

    if not email:
        return ""
    local = email.split("@", 1)[0]
    local = fold(local)
    local = re.sub(r"[._+\-]", "", local)
    return local

def name_units(name: str) -> List[str]:

    raw = name or ""
    parts = [p for p in SPLIT_NAME_RE.split(raw) if p]
    parts_fold = [fold(p) for p in parts]

    tokens = [t for t in parts_fold if len(t) >= MIN_UNIT_LEN]
    joined = "".join(parts_fold)
    full_compact = fold(re.sub(r"[ \t\-.·'’]+", "", raw))

    units = set(tokens)
    if len(joined) >= MIN_UNIT_LEN + 2:
        units.add(joined)
    if len(full_compact) >= MIN_UNIT_LEN + 2:
        units.add(full_compact)
    return list(units)

def score_name_against_email_local(name: str, email_local_folded: str):

    units = name_units(name)
    hits = [u for u in units if u and u in email_local_folded]
    if not hits:
        return 0.0, []
    total_len = sum(len(h) for h in hits)
    score = len(hits) + total_len / 100.0  #
    return score, hits

def pick_best_name(email: str, candidates: List[str]):

    em_local = email_local_norm(email)
    best = None
    best_score = 0.0
    for nm in candidates:
        sc, _ = score_name_against_email_local(nm, em_local)
        if sc > best_score or (sc == best_score and best is not None and len(nm) > len(best)):
            best, best_score = nm, sc
        elif best is None and sc > 0:
            best, best_score = nm, sc
    if best_score > 0:
        return best
    return None  # ❗

def collect_names(obj: dict) -> List[str]:

    if "name" in obj and obj["name"]:
        return [obj["name"]]
    items = []
    for k, v in obj.items():
        if k.startswith("name_") and v:
            try:
                idx = int(k.split("_", 1)[1])
            except:
                idx = 1_000_000
            items.append((idx, v))
    items.sort(key=lambda x: x[0])
    return [v for _, v in items]

def process_language(lang: str):
    in_path = os.path.join(INPUT_BASE, f"{lang}.jsonl")
    if not os.path.exists(in_path):
        print(f"[{lang}] missing: {in_path}")
        return
    out_path = os.path.join(OUTPUT_BASE, f"{lang}.jsonl")

    total = sum(1 for _ in open(in_path, "r", encoding="utf-8"))
    kept = 0
    direct_count = 0       # 1→1
    mapped_count = 0       # N→1
    skipped_no_match = 0   # N→1 

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        pbar = tqdm(fin, total=total, desc=f"pick {lang}")
        for line in pbar:
            try:
                obj = json.loads(line)
            except:
                continue
            email = obj.get("email")
            text  = obj.get("text") or obj.get("window")
            if not (email and text):
                continue

            names = collect_names(obj)
            if not names:
                continue


            if len(names) == 1:
                final_name = names[0]
                direct_count += 1
            else:
                chosen = pick_best_name(email, names)
                if not chosen:
                    skipped_no_match += 1
                    continue  
                final_name = chosen
                mapped_count += 1

            out = {
                "name": final_name,
                "email": email,
                "text": text
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1
        pbar.close()

    print(f"[{lang}] kept={kept} -> {out_path}")
    print(f"[{lang}] direct (1→1) = {direct_count}")
    print(f"[{lang}] mapped  (N→1) = {mapped_count}")
    print(f"[{lang}] skipped_no_match = {skipped_no_match}")
    
def main():
    for lang in LANGS:
        process_language(lang)

if __name__ == "__main__":
    main()
