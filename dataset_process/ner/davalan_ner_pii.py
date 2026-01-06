import os
import re
import json
import unicodedata
from collections import OrderedDict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ===== Paths =====
INPUT_BASE  = "dataset/MLLM_MEM/PII/double_pii_windows/test"
OUTPUT_BASE = "dataset/MLLM_MEM/PII/structure_pii/test"

LANGS = [""] 
MODEL_BASE_DIR = "model"

os.makedirs(OUTPUT_BASE, exist_ok=True)

# ===== Model mapping =====
davlan_langs = ["en","de","zh","nl","fr","it","es","lv","pt","ar"]


MODEL_BY_LANG = {}
for lang in davlan_langs:
    MODEL_BY_LANG[lang] = "Davlan/xlm-roberta-base-ner-hrl"
MODEL_BY_LANG["_default"] = "wrong"  # only accept which could be use by Davalan's NER

# ===== Tracking =====
SKIPPED_NER = []       
MISSING_INPUT = []     
PROCESSED = []         

# ===== Text utils =====
def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", (text or "")).strip()

def is_valid_name(name: str, lang: str) -> bool:
    if not name:
        return False
    s = name.strip(" ,.;:()[]{}<>“”\"'`")
    if not s:
        return False
    if any(ch.isdigit() for ch in s) or "@" in s:
        return False
    return len(s) >= 2

# ===== Robust build_pipeline =====
def build_pipeline(lang: str):
    repo_id = MODEL_BY_LANG.get(lang, MODEL_BY_LANG["_default"])

    tok = None
    for local_only in (True, False):
        try:
            tok = AutoTokenizer.from_pretrained(
                repo_id, use_fast=True, cache_dir=MODEL_BASE_DIR, local_files_only=local_only
            )
            break
        except Exception:
            if not local_only:
                raise  
    if tok is None:
        raise RuntimeError(f"Tokenizer load failed for {repo_id}")

    last_err = None
    for local_only in (True, False):
        try:
            mdl = AutoModelForTokenClassification.from_pretrained(
                repo_id, cache_dir=MODEL_BASE_DIR, local_files_only=local_only
            )
            return pipeline(
                "token-classification", model=mdl, tokenizer=tok, aggregation_strategy="simple"
            )
        except Exception as e:
            last_err = e
            if not local_only:
                pass  

    for local_only in (True, False):
        try:
            mdl = AutoModelForTokenClassification.from_pretrained(
                repo_id, cache_dir=MODEL_BASE_DIR, local_files_only=local_only, from_tf=True
            )
            return pipeline(
                "token-classification", model=mdl, tokenizer=tok, aggregation_strategy="simple"
            )
        except Exception as e:
            last_err = e
            if not local_only:
                raise RuntimeError(
                    f"Failed to load '{repo_id}' for lang '{lang}'. Last error: {last_err}"
                )

def extract_names(nlp, text: str, lang: str):
    text = _norm(text)
    if not text:
        return []
    ents = nlp(text)
    names = []
    for e in ents:
        label = (e.get("entity_group") or e.get("entity") or "").upper()
        if label in {"PER", "PERSON"}:
            name = e.get("word") or e.get("entity") or ""
            name = name.replace(" ##", "").replace("▁", " ").strip()
            name = re.sub(r"\s{2,}", " ", name)
            if is_valid_name(name, lang):
                names.append(name)
    deduped = list(OrderedDict((n.lower(), n) for n in names).values())
    return deduped

def process_language(lang: str):
    in_path = os.path.join(INPUT_BASE, f"{lang}.jsonl")
    if not os.path.exists(in_path):
        print(f"[{lang}] missing: {in_path}")
        MISSING_INPUT.append(lang)
        return

    try:
        nlp = build_pipeline(lang)
    except Exception as e:
        print(f"[{lang}] NER unavailable -> skip. Reason: {e}")
        SKIPPED_NER.append(lang)
        return

    out_dir = os.path.join(OUTPUT_BASE)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{lang}.jsonl")

    total = sum(1 for _ in open(in_path, "r", encoding="utf-8"))
    kept = 0
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        pbar = tqdm(fin, total=total, desc=f"👤 NER {lang}")
        for line in pbar:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue

            email = obj.get("email", "")
            phone = obj.get("phone", "")
            text  = _norm(obj.get("window", "")) 

            if not text or not email or not phone:
                continue

            names = extract_names(nlp, text, lang)
            if not names:
                continue

            out = {
                "email": email,
                "phone": phone,
                "text":  text,
            }
            if len(names) == 1:
                out["name"] = names[0]
            else:
                for i, n in enumerate(names, 1):
                    out[f"name_{i}"] = n

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1

        pbar.close()

    print(f"[{lang}] kept={kept} -> {out_path}")
    PROCESSED.append(lang)

def main():
    for lang in LANGS:
        process_language(lang)

    print("\n===== SUMMARY =====")
    if PROCESSED:
        print(f"Processed ({len(PROCESSED)}): {sorted(PROCESSED)}")
    else:
        print("Processed: []")
    if SKIPPED_NER:
        print(f"Skipped due to NER unavailable ({len(SKIPPED_NER)}): {sorted(SKIPPED_NER)}")
    else:
        print("Skipped due to NER unavailable: []")
    if MISSING_INPUT:
        print(f"Missing input file ({len(MISSING_INPUT)}): {sorted(MISSING_INPUT)}")
    else:
        print("Missing input file: []")

if __name__ == "__main__":
    main()
