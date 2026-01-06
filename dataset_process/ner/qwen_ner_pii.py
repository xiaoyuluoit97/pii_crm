import os
import re
import json
import math
import unicodedata
from collections import OrderedDict, defaultdict
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

INPUT_BASE  = "dataset/MLLM_MEM/PII/structure_pii/train"
OUTPUT_BASE = "dataset/MLLM_MEM/PII/structure_pii/one_map_one/train"
LANGS = [
    "pl", "tr", "pt", "af", "ru", "fr", "sw", "vi", "es", "ta",
    "az", "hu", "it", "en", "be", "sv", "lt", "de", "da", "ar", 
    "fi", "zh", "uk", "lv", "hi", "nl", "ro", "bg", "ko", "el", 
    "th", "ml"
]

MODEL_ID         = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_CACHE_DIR  = "model"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE            = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16


MAX_NEW_TOKENS       = 256
TEMPERATURE          = 0.0
TOP_P                = 1.0
REPETITION_PENALTY   = 1.0

CHUNK_MAX_CHARS  = 3500
CHUNK_OVERLAP    = 200

TEXTS_PER_BATCH  = 32   # 
MAX_INPUT_TOKENS_FALLBACK = 8192  

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.environ.setdefault("TOKENIZER_PARALLELISM", "false")


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

def chunk_text(s: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    s = s or ""
    n = len(s)
    if n <= max_chars:
        return [s]
    chunks = []
    start = 0
    while start < n:
        end = min(n, start + max_chars)
        chunks.append(s[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

# ================== Qwen ==================
def resolve_local_repo_path(base_dir: str, repo_id: str):
    parts = repo_id.split("/")
    path = os.path.join(base_dir, *parts)
    return path if os.path.isdir(path) else None

def load_qwen():
    local_repo = resolve_local_repo_path(MODEL_CACHE_DIR, MODEL_ID)
    tok = AutoTokenizer.from_pretrained(
        local_repo or MODEL_ID,
        cache_dir=MODEL_CACHE_DIR,
        use_fast=True,
        trust_remote_code=True,   
    )


    tok.padding_side = "left"


    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        else:

            tok.add_special_tokens({"pad_token": "<|pad|>"})

    mdl = AutoModelForCausalLM.from_pretrained(
        local_repo or MODEL_ID,
        cache_dir=MODEL_CACHE_DIR,
        torch_dtype=DTYPE if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,   
    )
    if DEVICE != "cuda":
        mdl = mdl.to(DEVICE)
    mdl.eval()

 
    try:
        mdl.config.pad_token_id = tok.pad_token_id
    except Exception:
        pass


    conf_max = getattr(mdl.config, "max_position_embeddings", None)
    if conf_max is None or conf_max > 131072:
        conf_max = MAX_INPUT_TOKENS_FALLBACK
    max_input_tokens = max(512, conf_max - MAX_NEW_TOKENS - 16)
    return tok, mdl, max_input_tokens

# ================== Prompt  ==================
SYSTEM_PROMPT = (
    "You are an expert NER tagger. Extract ONLY PERSON names from the given text.\n"
    "Rules:"
    "- Output MUST be a pure JSON object with this exact schema: {\"names\": [\"...\"]}\n"
    "- Return unique names only, keep original casing/characters.\n"
    "- Exclude locations, dates, emails, phones, usernames, IDs.\n"
    "- Do NOT fabricate names; if none, return {\"names\": []}.\n"
    "- Do not add explanations or any extra text."
)

USER_PROMPT_TEMPLATE = (
    "Language: {lang}\n"
    "Task: Extract PERSON names only from the following text.\n\n"
    "<TEXT>\n{content}\n</TEXT>\n\n"
    "Respond with JSON only."
)

def build_chat_input(tokenizer, lang: str, text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(lang=lang, content=text)}
    ]
    chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return chat

# ================== parse the json from llm's output ==================
def _extract_first_json_obj(s: str) -> Dict[str, Any] | None:
    if not s:
        return None
    s = s.strip()
    s = re.sub(r"^```(json)?\s*|\s*```$", "", s, flags=re.DOTALL).strip()
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start:i+1]
                try:
                    return json.loads(candidate)
                except Exception:
                    break
    m = re.search(r'\{\s*"names"\s*:\s*\[(.*?)\]\s*\}', s, flags=re.DOTALL|re.IGNORECASE)
    if m:
        inside = m.group(1)
        items = re.findall(r'"([^"]+)"', inside)
        return {"names": items}
    return None

# ================== parse ==================
@torch.no_grad()
def qwen_extract_names_batch(tokenizer, model, texts: List[str], lang: str, max_input_tokens: int) -> List[List[str]]:
    rows = []  
    for i, text in enumerate(texts):
        text = _norm(text)
        if not text:
            rows.append({"idx": i, "chunk": ""})
            continue
        for ck in chunk_text(text, CHUNK_MAX_CHARS, CHUNK_OVERLAP):
            rows.append({"idx": i, "chunk": ck})

    agg_names: Dict[int, List[str]] = defaultdict(list)
    seen_lower_by_idx: Dict[int, set] = defaultdict(set)

    for b in range(0, len(rows), TEXTS_PER_BATCH):
        batch_rows = rows[b:b+TEXTS_PER_BATCH]
        prompts = [build_chat_input(tokenizer, lang, r["chunk"]) for r in batch_rows]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        input_lens = enc["attention_mask"].sum(dim=1).tolist()

        gen = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,      
            num_beams=1,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )


        for r, input_len in enumerate(input_lens):
            gen_ids = gen[r][input_len:]
            out_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            obj = _extract_first_json_obj(out_text)
            names = []
            if isinstance(obj, dict) and isinstance(obj.get("names"), list):
                names = [str(x).strip() for x in obj["names"] if isinstance(x, str)]
            else:
                lines = [ln.strip() for ln in out_text.splitlines() if ln.strip()]
                for ln in lines:
                    m = re.match(r'^(?:-|\*|\d+\.)\s*(.+)$', ln)
                    names.append(m.group(1).strip() if m else ln)

            idx = batch_rows[r]["idx"]
            for nm in names:
                nm = nm.strip(" ,.;:()[]{}<>“”\"'`")
                if not is_valid_name(nm, lang):
                    continue
                key = nm.lower()
                if key not in seen_lower_by_idx[idx]:
                    seen_lower_by_idx[idx].add(key)
                    agg_names[idx].append(nm)

    results: List[List[str]] = []
    for i in range(len(texts)):
        deduped = list(OrderedDict((n.lower(), n) for n in agg_names.get(i, [])).values())
        results.append(deduped)
    return results


def process_language(tokenizer, model, max_input_tokens: int, lang: str):
    in_path  = os.path.join(INPUT_BASE, f"{lang}.jsonl")
    if not os.path.exists(in_path):
        print(f"[{lang}] missing: {in_path}")
        return

    out_dir = OUTPUT_BASE
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{lang}.jsonl")

    buffer_records = []  #{"email","phone","text"}
    total = sum(1 for _ in open(in_path, "r", encoding="utf-8"))
    kept = 0

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout, \
         tqdm(total=total, desc=f"👤 Qwen NER (batch) {lang}") as pbar:

        def flush_batch():
            nonlocal kept
            if not buffer_records:
                return
            texts = [r["text"] for r in buffer_records]
            try:
                names_lists = qwen_extract_names_batch(tokenizer, model, texts, lang, max_input_tokens)
            except Exception as e:

                names_lists = []
                for t in texts:
                    try:
                        names_lists.append(qwen_extract_names_batch(tokenizer, model, [t], lang, max_input_tokens)[0])
                    except Exception:
                        names_lists.append([])

            for rec, names in zip(buffer_records, names_lists):
                if not names:
                    continue
                out = {
                    "email": rec["email"],
                    "phone": rec["phone"],
                    "text":  rec["text"],
                }
                if len(names) == 1:
                    out["name"] = names[0]
                else:
                    for i, n in enumerate(names, 1):
                        out[f"name_{i}"] = n
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                kept += 1

            buffer_records.clear()

        for line in fin:
            pbar.update(1)
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

            buffer_records.append({"email": email, "phone": phone, "text": text})
            if len(buffer_records) >= TEXTS_PER_BATCH:
                flush_batch()

        flush_batch()

    print(f"[{lang}] kept={kept} -> {out_path}")

def main():
    try:
        tokenizer, model, max_input_tokens = load_qwen()
    except Exception as e:
        print(f"[MODEL] load failed: {e}")
        return

    for lang in LANGS:
        process_language(tokenizer, model, max_input_tokens, lang)

    print("\n===== DONE =====")

if __name__ == "__main__":
    main()
