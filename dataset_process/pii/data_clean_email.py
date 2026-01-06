import os
import re
import json
from tqdm import tqdm
from transformers import AutoTokenizer

# ---------- 路径与设置 ----------
INPUT_DIR  = "dataset/MLLM_MEM/PII/email_raw/test"
OUTPUT_DIR = "dataset/MLLM_MEM/PII/email_prompts_mGPT/test"

LANGS = ["zh","th","da","de","es","it","hi","en","fr","nl",
         "pt","ru","uk","be","ja","ar","he","af","ur","ro",
         "sv","ko","tr","tk","az","ta","te","ml","lv","lt","fi","hu","sw","yo"]

PROMPT_TOKEN_LENS = [100]

# 采样上限（每个 K 单独控制；None 表示不设上限）
MAX_SAMPLES_PER_K = {50: None, 100: 20_000, 200: 20_000}

# 批量 tokenization 大小
BATCH_SIZE = 5000

# ✅ 数字“异常密集”与批量命中的阈值
MAX_DIGIT_RATIO = 0.35
MAX_LONG_NUM_GROUPS = 5
MAX_EMAIL_SPANS_PER_TEXT = 5  # 单条文本允许的邮箱数量上限

# 本地 HuggingFace 模型仓库根目录
MODEL_BASE_DIR = "model"
MODEL_ID = "ai-forever/mGPT"

# ✅ 去重策略（默认：仅在同一 K 内去重；设为 True 则跨 K 全局去重）
DEDUP_ACROSS_K = False

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ.setdefault("TOKENIZER_PARALLELISM", "false")

# ---------- 正则 ----------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", re.UNICODE)
LONG_NUM_RE = re.compile(r"\d{5,}")
# URL/域名匹配（支持 http(s)://、ftp://、www.、以及裸域名）
URL_OR_DOMAIN_RE = re.compile(
    r"(?i)\b(?:https?://|ftp://|www\.)?((?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})(?::\d{2,5})?(?:/[^\s]*)?\b"
)

# 🚫 伪邮箱后缀黑名单（文件类型等）
EXT_BLOCKLIST = {
    "png","jpg","jpeg","gif","webp","svg","bmp","tif","tiff","ico",
    "pdf","doc","docx","xls","xlsx","ppt","pptx",
    "zip","rar","7z","tar","gz","bz2",
    "mp3","wav","flac","mp4","mov","avi","mkv","webm","heic",
    "apk","exe","dmg","iso","psd","ai","sketch"
}

# -------- 工具函数 --------
def is_numeric_heavy(text: str) -> bool:
    """判断文本是否数字异常密集"""
    if not text:
        return False
    total = len(text)
    digits = sum(ch.isdigit() for ch in text)
    ratio = digits / max(1, total)
    if ratio >= MAX_DIGIT_RATIO:
        return True
    if len(LONG_NUM_RE.findall(text)) >= MAX_LONG_NUM_GROUPS:
        return True
    return False

def appears_inside_url_or_path(text: str, start: int, end: int) -> bool:

    if not text:
        return False
    L = start
    while L > 0 and not text[L-1].isspace():
        L -= 1
    R = end
    n = len(text)
    while R < n and not text[R].isspace():
        R += 1
    token = text[L:R].lower()

    if "://" in token or token.startswith("www."):
        return True
    if "/" in token or "\\" in token:
        return True

    prev_ch = text[start-1] if start > 0 else ""
    next_ch = text[end] if end < n else ""
    if prev_ch in "/\\" or next_ch in "/\\?#&":
        return True

    return False

def is_likely_email(candidate: str, context: str, start: int, end: int) -> bool:

    if not candidate:
        return False
    tld = candidate.rsplit(".", 1)[-1].lower()
    if tld in EXT_BLOCKLIST:
        return False
    if appears_inside_url_or_path(context, start, end):
        return False
    return True

def iter_email_spans(text: str):

    for m in EMAIL_RE.finditer(text or ""):
        s, e = m.start(), m.end()
        cand = m.group(0)
        if is_likely_email(cand, text, s, e):
            yield s, e, cand

def find_token_index_for_char(offsets, char_pos: int) -> int:
    """给定 offsets_mapping，返回覆盖 char_pos 的 token 索引；若找不到，返回最接近的左侧 token"""
    last_valid = 0
    for i, (s, e) in enumerate(offsets):
        if s is None or e is None:
            continue
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
        print(f"ℹ️ Local cache-only load failed: {e}")
    print("🌐 Downloading tokenizer from Hugging Face:", base_dir)
    return AutoTokenizer.from_pretrained(
        repo_id, use_fast=True, cache_dir=base_dir, local_files_only=False
    )

def normalize_domain(d: str) -> str:
    """规范化域名：去尾点、转小写、去 www. 前缀、转 IDNA"""
    if not d:
        return ""
    d = d.strip().strip(".").lower()
    if d.startswith("www."):
        d = d[4:]
    try:
        d = d.encode("idna").decode("ascii")
    except Exception:
        pass
    return d

def extract_domains(text: str):
    """从文本中提取所有 URL/裸域名的主机部分（已规范化）"""
    domains = set()
    if not text:
        return domains
    for m in URL_OR_DOMAIN_RE.finditer(text):
        dom = normalize_domain(m.group(1))
        if dom:
            domains.add(dom)
    return domains

def get_email_domain(addr: str) -> str:
    """从邮箱中取域名并规范化"""
    try:
        dom = addr.split("@", 1)[1]
    except Exception:
        return ""
    return normalize_domain(dom)

def domain_equivalent_or_sub(a: str, b: str) -> bool:
    """域名是否相同或互为子域（a==b 或 a 以 .b 结尾或 b 以 .a 结尾）"""
    if not a or not b:
        return False
    return a == b or a.endswith("." + b) or b.endswith("." + a)

def all_limits_reached(counters):
    for K, limit in MAX_SAMPLES_PER_K.items():
        if limit is not None and counters.get(K, 0) < limit:
            return False
    return True

# Tokenizer
tokenizer = load_tokenizer_with_local_fallback(MODEL_ID, MODEL_BASE_DIR)
ADD_SPECIAL_TOKENS = False

# -------- 主处理函数 --------
def process_lang(lang_code: str):
    in_path = os.path.join(INPUT_DIR, f"{lang_code}.jsonl")
    if not os.path.exists(in_path):
        print(f"❌ Missing input file: {in_path}")
        return

    total_lines = sum(1 for _ in open(in_path, "r", encoding="utf-8"))
    print(f"\n🌍 {lang_code} | Reading: {in_path}  (lines: {total_lines:,})")

    out_files = {K: open(os.path.join(OUTPUT_DIR, f"{lang_code}_{K}.jsonl"), "w", encoding="utf-8")
                 for K in PROMPT_TOKEN_LENS}

    skipped_due_to_email_in_prompt = {K: 0 for K in PROMPT_TOKEN_LENS}
    skipped_due_to_domain_in_prompt = {K: 0 for K in PROMPT_TOKEN_LENS}
    skipped_due_to_duplicate = {K: 0 for K in PROMPT_TOKEN_LENS}  # ✅ 新增：重复邮箱跳过计数
    collected_per_K = {K: 0 for K in PROMPT_TOKEN_LENS}

    # ✅ 去重集合
    seen_emails_per_K = {K: set() for K in PROMPT_TOKEN_LENS}
    seen_emails_global = set() if DEDUP_ACROSS_K else None

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

                for (start_char, end_char, email) in spans:
                    tok_idx = find_token_index_for_char(offsets, start_char)

                    for K in PROMPT_TOKEN_LENS:
                        max_cap = MAX_SAMPLES_PER_K.get(K)
                        if max_cap is not None and collected_per_K[K] >= max_cap:
                            continue

                        # ✅ 去重判断
                        if DEDUP_ACROSS_K:
                            if email in seen_emails_global:
                                skipped_due_to_duplicate[K] += 1
                                continue
                        else:
                            if email in seen_emails_per_K[K]:
                                skipped_due_to_duplicate[K] += 1
                                continue

                        left_tok = max(0, tok_idx - K)
                        # 需要至少 K 个 token 的左上下文
                        if tok_idx - left_tok < K:
                            continue

                        prompt_begin_char = offsets[left_tok][0]
                        prompt_end_char = start_char
                        if prompt_begin_char is None or prompt_end_char is None:
                            continue
                        if not (0 <= prompt_begin_char <= prompt_end_char <= len(text_str)):
                            continue

                        prompt_text = text_str[prompt_begin_char:prompt_end_char]
                        prompt_text_l = prompt_text.lower()

                        # 规则1：邮箱串本身出现在 prompt 里，跳过
                        if email in prompt_text:
                            skipped_due_to_email_in_prompt[K] += 1
                            continue

                        # 规则2：域名泄露（URL/裸域名与邮箱域相同或互为子域），跳过
                        email_dom = get_email_domain(email)
                        prompt_domains = extract_domains(prompt_text)
                        has_domain_leak = any(domain_equivalent_or_sub(email_dom, d) for d in prompt_domains)

                        # 兜底：prompt 里直接出现裸域名字符串
                        if not has_domain_leak and email_dom and email_dom in prompt_text_l:
                            has_domain_leak = True

                        if has_domain_leak:
                            skipped_due_to_domain_in_prompt[K] += 1
                            continue

                        original_text = text_str[prompt_begin_char:end_char]
                        record = {"email": email, "prompt": prompt_text, "original": original_text}
                        out_files[K].write(json.dumps(record, ensure_ascii=False) + "\n")
                        collected_per_K[K] += 1

                        # ✅ 写入后登记为已见
                        if DEDUP_ACROSS_K:
                            seen_emails_global.add(email)
                        else:
                            seen_emails_per_K[K].add(email)

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

            spans = list(iter_email_spans(text_str))  # 已含后验过滤
            if not spans:
                continue
            if len(spans) > MAX_EMAIL_SPANS_PER_TEXT:
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
            f"Skipped(leak_email)={skipped_due_to_email_in_prompt[K]} | "
            f"Skipped(leak_domain)={skipped_due_to_domain_in_prompt[K]} | "
            f"Skipped(duplicate)={skipped_due_to_duplicate[K]}"
        )

def main():
    for lang in LANGS:
        process_lang(lang)

if __name__ == "__main__":
    main()
