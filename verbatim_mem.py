import os
import json
import math
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers import MT5Tokenizer, GPT2LMHeadModel


TARGET_FIELD = "email"  # "email" or "phone"


PII_BASE_DIR = "dataset/MLLM_MEM/PII/new_verbatim"
PROMPT_DIR = os.path.join(PII_BASE_DIR, f"{TARGET_FIELD}")  # {lang}_{50|100|200}.jsonl

if TARGET_FIELD == "phone":
    RESULT_DIR = "./mem_result_new2000/phone"
elif TARGET_FIELD == "email":
    RESULT_DIR = "./mem_result_new2000/email"
else:
    RESULT_DIR = "./mem_result_new2000/name"

# ====== language selection ======
LANGUAGES = [
    "af", "ar", "az", "be", "bg", "da",
    "de", "el", "en", "es", "fi", "fr",
    "hi", "hu", "it", "ko", "lt", "lv",
    "ml", "nl", "pl", "pt", "ro", "ru",
    "sv", "sw", "ta", "th", "tr", "uk", "vi", "zh"
]

PROMPT_TOKEN_LENS = [100]
MODEL_ID = "ai-forever/mGPT-13B"
MODEL_CACHE_DIR = "model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 50

FIXED_GEN_TOKENS = 15

SAMPLE_LIMIT = 2000

VERBOSE_LOG_NON_HITS = False
NON_HIT_LOG_EVERY_N = 100
LOG_GEN_TRUNCATE_CHARS = None

# ====== only calculate log-likelihood ì======
ONLY_EVAL_LOG_LIKELIHOOD = False 

# ====== LL debug======
ENABLE_LL_DEBUG = True      
LL_DEBUG_TO_FILE = True      
LL_DEBUG_TO_CONSOLE = True   
LL_DEBUG_EVERY = 100         


WRITE_LL_SAMPLES = True

os.makedirs(RESULT_DIR, exist_ok=True)
os.environ.setdefault("TOKENIZER_PARALLELISM", "false")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    cache_dir=MODEL_CACHE_DIR,
    use_fast=True,
    local_files_only=True
)

model = GPT2LMHeadModel.from_pretrained(
    MODEL_ID,
    cache_dir=MODEL_CACHE_DIR,
    torch_dtype=torch.float16,
    local_files_only=True
).to(DEVICE)
model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
pad_id = tokenizer.pad_token_id
eos_id = tokenizer.eos_token_id

@torch.no_grad()
def batch_memorization_check(prompts: List[str], targets: List[str]):
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False,
        max_length=2048
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    max_ctx = getattr(model.config, "n_positions", 2048)
    if input_ids.size(1) + FIXED_GEN_TOKENS > max_ctx:
        keep = max_ctx - FIXED_GEN_TOKENS
        keep = max(1, keep)
        input_ids = input_ids[:, -keep:]
        attention_mask = attention_mask[:, -keep:]

    gen_out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=FIXED_GEN_TOKENS,
        do_sample=False,
        num_beams=1,
        pad_token_id=pad_id,
        eos_token_id=eos_id
    )

    gen_texts = []
    for i in range(gen_out.size(0)):
        prompt_len = (attention_mask[i] != 0).sum().item()
        cont_ids = gen_out[i][prompt_len:]
        gen_text = tokenizer.decode(
            cont_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        gen_texts.append(gen_text)

    hits = [(t in gen_str) for gen_str, t in zip(gen_texts, targets)]
    return hits, gen_texts


@torch.no_grad()
def compute_suffix_loglik_batch(
    model,
    full_input_ids_batch: torch.Tensor,     # [B, L_pad]
    prefix_lens: torch.Tensor,              # [B] 
    return_trace: bool = False,
    total_lens: Optional[torch.Tensor] = None, 
    attention_mask: Optional[torch.Tensor] = None,
):

    device = (
        full_input_ids_batch.device
        if isinstance(full_input_ids_batch, torch.Tensor) and full_input_ids_batch.device.type != "cpu"
        else (model.device if hasattr(model, "device") else torch.device("cpu"))
    )
    full_input_ids_batch = full_input_ids_batch.to(device)
    prefix_lens = prefix_lens.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    B, L_pad = full_input_ids_batch.shape

    if total_lens is None:
        total_lens = torch.full((B,), L_pad, device=device, dtype=torch.long)
    else:
        total_lens = total_lens.to(device)

    outputs = model(input_ids=full_input_ids_batch, attention_mask=attention_mask)
    logits = outputs.logits.to(torch.float32)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    log_probs_shifted = log_probs[:, :-1, :]
    next_tokens = full_input_ids_batch[:, 1:]
    all_lp = torch.gather(log_probs_shifted, 2, next_tokens.unsqueeze(-1)).squeeze(-1)  # [B, L_pad-1]

    pos = torch.arange(1, L_pad, device=device).unsqueeze(0).expand(B, -1)
    mask = (pos >= prefix_lens.unsqueeze(1)) & (pos < total_lens.unsqueeze(1))

    log_probs_per_token_list = [all_lp[i, mask[i]] for i in range(B)]
    total_logprobs = (all_lp * mask).sum(dim=1)

    if not return_trace:
        return log_probs_per_token_list, total_logprobs

    trace_list: List[Dict[str, Any]] = []
    for i in range(B):
        idxs = torch.nonzero(mask[i], as_tuple=False).squeeze(-1)
        suffix_positions = pos[i, idxs].detach().cpu().tolist()
        suffix_token_ids = next_tokens[i, idxs].detach().cpu().tolist()
        suffix_logprobs = all_lp[i, idxs].detach().cpu().tolist()
        cumsum_logprob = float(sum(suffix_logprobs))
        trace_list.append({
            'prefix_len': int(prefix_lens[i].item()),
            'total_len': int(total_lens[i].item()),
            'suffix_positions': suffix_positions,
            'suffix_token_ids': suffix_token_ids,
            'suffix_logprobs': suffix_logprobs,
            'cumsum_logprob': cumsum_logprob,
        })
    return log_probs_per_token_list, total_logprobs, trace_list


# ========= 逐 token LL 调试日志 =========
def write_ll_debug(
    vtrace, to_console: bool, every: int, idx: int, key: str,
    prompt: str, target: str, tok, trace: Dict[str, Any]
):
    if trace is None:
        return

    prefix_len = trace['prefix_len']
    total_len = trace['total_len']
    suffix_pos = trace['suffix_positions']
    suffix_ids = trace['suffix_token_ids']
    suffix_lp = trace['suffix_logprobs']
    cumsum_lp = trace['cumsum_logprob']

    try:
        token_strs = tok.convert_ids_to_tokens(suffix_ids)
    except Exception:
        token_strs = [tok.decode([tid]) for tid in suffix_ids]

    header = []
    header.append(f"[{key}] idx={idx}  (total_len={total_len}, prefix_len={prefix_len}, target_len={total_len - prefix_len})")
    header.append("[SEGMENT]")
    header.append(f"Prompt str : {prompt}")
    header.append(f"Target str : {target}")
    header.append(f"Cut ranges : prompt=[0, {prefix_len})  target=[{prefix_len}, {total_len})")
    header.append("Note: logits[t] predicts token at position t+1; suffix positions below are in tokens[:,1:] coords.")
    header.append("")

    rows = []
    rows.append("t_pos(next) | token_id | token_str           | logprob           | cumsum")
    running = 0.0
    for p, tid, tstr, lp in zip(suffix_pos, suffix_ids, token_strs, suffix_lp):
        running += lp
        tdisp = tstr if isinstance(tstr, str) else str(tstr)
        if len(tdisp) > 22:
            tdisp = tdisp[:21] + "…"
        rows.append(f"{p:11d} | {tid:8d} | {tdisp:22s} | {lp:16.6f} | {running: .6f}")

    rows.append("")
    rows.append(f"SUM logprob (suffix) = {cumsum_lp:.6f}")
    rows.append("")

    blob = "\n".join(header + rows)
    if ENABLE_LL_DEBUG and LL_DEBUG_TO_FILE and vtrace is not None:
        vtrace.write(blob + "\n")
    if ENABLE_LL_DEBUG and to_console and (every <= 1 or (idx % every == 0)):
        print(blob)


def maybe_truncate(s: str, limit: int | None):
    if limit is None or s is None:
        return s
    return s if len(s) <= limit else s[:limit] + "…[TRUNC]"


# ====== LL batch ======
def build_ll_batch(prompts: List[str], targets: List[str], max_length: int = 2048):
    full_ids_list = []
    prefix_lens = []
    total_lens = []
    valid_idx = []
    meta = []

    for i, (p, t) in enumerate(zip(prompts, targets)):
        enc_full = tokenizer(
            p + t,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_length
        )
        ids_1d = enc_full["input_ids"][0].cpu()
        tot_len = int(ids_1d.size(0))

        with tokenizer.as_target_tokenizer():
            t_ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        tlen = len(t_ids)

        if tlen > 0 and tlen < tot_len:
            pre_len = tot_len - tlen
            full_ids_list.append(ids_1d)
            prefix_lens.append(pre_len)
            total_lens.append(tot_len)
            valid_idx.append(i)
            meta.append({"prompt": p, "target": t})

    if not full_ids_list:
        return None, None, None, [], [], None

    pad_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    full_batch = pad_sequence(full_ids_list, batch_first=True, padding_value=pad_val).to(DEVICE)

    attn_mask = (full_batch != pad_val).long()
    prefix_lens_t = torch.tensor(prefix_lens, dtype=torch.long, device=full_batch.device)
    total_lens_t = torch.tensor(total_lens, dtype=torch.long, device=full_batch.device)
    return full_batch, prefix_lens_t, total_lens_t, valid_idx, meta, attn_mask

ì
def eval_one_file(jsonl_path, lang, k, overall_path_jsonl, ll_samples_path_jsonl):
    if not os.path.exists(jsonl_path):
        print(f"⚠️ Missing file: {jsonl_path}")
        return None

    # 读样本
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if TARGET_FIELD in obj and "prompt" in obj:
                    samples.append(obj)
            except Exception:
                continue

    if not samples:
        print(f"⚠️ No valid samples in {jsonl_path}")
        return None

    if SAMPLE_LIMIT is not None:
        samples = samples[:SAMPLE_LIMIT]
    total = len(samples)
    hits_total = 0  

    if not ONLY_EVAL_LOG_LIKELIHOOD:
        hit_examples_path = os.path.join(RESULT_DIR, "exact_mem_sample")
        hit_examples_path = os.path.join(
            hit_examples_path, f"hits_{MODEL_ID.split('/')[-1]}_{TARGET_FIELD}_{lang}_{k}.jsonl"
        )
        os.makedirs(os.path.dirname(hit_examples_path), exist_ok=True)
        fout = open(hit_examples_path, "w", encoding="utf-8")
    else:
        hit_examples_path = None
        fout = None

    non_hit_seen = 0

    file_ll_sums: List[float] = []

    vtrace = None
    if ENABLE_LL_DEBUG and LL_DEBUG_TO_FILE:
        vtrace_path = os.path.join(RESULT_DIR, f"lltrace_{MODEL_ID.split('/')[-1]}_{TARGET_FIELD}_{lang}_{k}.log")
        vtrace = open(vtrace_path, "w", encoding="utf-8")

    fl_samples = None
    if WRITE_LL_SAMPLES:
        fl_samples = open(ll_samples_path_jsonl, "a", encoding="utf-8")

    try:
        for i in tqdm(range(0, total, BATCH_SIZE), desc=f"🔍 {lang}-{k}-{TARGET_FIELD}"):
            batch = samples[i:i+BATCH_SIZE]
            prompts = [s["prompt"] for s in batch]
            targets = [s[TARGET_FIELD] for s in batch]

            # ====== calculate target  log-likelihood ======
            full_batch, prefix_lens_t, total_lens_t, valid_idx, meta, attn_mask = build_ll_batch(
                prompts, targets, max_length=2048
            )

            ll_list: List[Optional[float]] = [None] * len(batch)
            traces: Dict[int, Dict[str, Any]] = {}

            if full_batch is not None:
                if ENABLE_LL_DEBUG:
                    _, total_logprobs, trace_list = compute_suffix_loglik_batch(
                        model=model,
                        full_input_ids_batch=full_batch,
                        prefix_lens=prefix_lens_t,
                        return_trace=True,
                        total_lens=total_lens_t,
                        attention_mask=attn_mask
                    )
                else:
                    _, total_logprobs = compute_suffix_loglik_batch(
                        model=model,
                        full_input_ids_batch=full_batch,
                        prefix_lens=prefix_lens_t,
                        return_trace=False,
                        total_lens=total_lens_t,
                        attention_mask=attn_mask
                    )
                    trace_list = [None] * len(valid_idx)  # occ

                for k_valid, b_index in enumerate(valid_idx):
                    val = float(total_logprobs[k_valid].item())
                    ll_list[b_index] = val
                    file_ll_sums.append(val)
                    if ENABLE_LL_DEBUG and trace_list[k_valid] is not None:
                        traces[b_index] = trace_list[k_valid]

            # ====== evaluate both ======
            if not ONLY_EVAL_LOG_LIKELIHOOD:
                hits, gen_texts = batch_memorization_check(prompts, targets)
            else:
                hits = [None] * len(batch)
                gen_texts = [None] * len(batch)

            # ====== log / hit / result ======
            for j, (s, hit, gen_str, ll_val) in enumerate(zip(batch, hits, gen_texts, ll_list)):
                if not ONLY_EVAL_LOG_LIKELIHOOD:
                    if hit:
                        hits_total += 1
                        hit_record = {
                            "lang": lang,
                            "prompt_tokens": k,
                            "prompt": s.get("prompt", ""),
                            "target": s.get(TARGET_FIELD, ""),
                            "generated": gen_str,
                            "original": s.get("original", ""),
                            "hit": True,
                            "loglik_sum": ll_val
                        }
                        fout.write(json.dumps(hit_record, ensure_ascii=False) + "\n")
                    else:
                        if VERBOSE_LOG_NON_HITS:
                            non_hit_seen += 1
                            if NON_HIT_LOG_EVERY_N >= 1 and (non_hit_seen % NON_HIT_LOG_EVERY_N == 0):
                                print("----- NON-HIT -----")
                                print("PROMPT:")
                                print(maybe_truncate(s.get('prompt', ''), LOG_GEN_TRUNCATE_CHARS))
                                print(f"{TARGET_FIELD.upper()}:")
                                print(s.get(TARGET_FIELD, ''))
                                print("GENERATED:")
                                print(maybe_truncate(gen_str, LOG_GEN_TRUNCATE_CHARS))
                                if ll_val is not None:
                                    print(f"[LL] loglik_sum={ll_val:.6f}")
                                else:
                                    print("[LL] NA")
                                print("-------------------", flush=True)
                else:

                    if (j % NON_HIT_LOG_EVERY_N == 0) and ll_val is not None:
                        print(f"[LL-ONLY] idx={i + j} loglik_sum={ll_val:.6f}")

                if ENABLE_LL_DEBUG and j in traces:
                    write_ll_debug(
                        vtrace=vtrace,
                        to_console=LL_DEBUG_TO_CONSOLE,
                        every=LL_DEBUG_EVERY,
                        idx=i + j,
                        key=f"{lang}_{k}",
                        prompt=prompts[j],
                        target=targets[j],
                        tok=tokenizer,
                        trace=traces[j],
                    )

                if WRITE_LL_SAMPLES and fl_samples is not None:
                    rec = {
                        "model": MODEL_ID,
                        "target_field": TARGET_FIELD,
                        "lang": lang,
                        "prompt_tokens": k,
                        "prompt": s.get("prompt", ""),
                        "target": s.get(TARGET_FIELD, ""),
                        "original": s.get("original", ""),
                        "r_local": s.get("r_local", None),
                        "r_domain": s.get("r_domain", None),
                        "generated": gen_str if not ONLY_EVAL_LOG_LIKELIHOOD else None,
                        "hit": bool(hit) if not ONLY_EVAL_LOG_LIKELIHOOD else None,
                        "loglik_sum": ll_val
                    }
                    fl_samples.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if not ONLY_EVAL_LOG_LIKELIHOOD:
            rate = hits_total / total if total > 0 else 0.0
        else:
            rate = None

        n_ll = len(file_ll_sums)
        if n_ll > 0:
            avg_ll_sum = sum(file_ll_sums) / n_ll
            if n_ll > 1:
                var = sum((x - avg_ll_sum) ** 2 for x in file_ll_sums) / (n_ll - 1)
                std = math.sqrt(var)
                se = std / math.sqrt(n_ll)
                ci_lo, ci_hi = avg_ll_sum - 1.96 * se, avg_ll_sum + 1.96 * se
            else:
                ci_lo = ci_hi = avg_ll_sum
        else:
            avg_ll_sum, ci_lo, ci_hi = None, None, None

        summary = {
            "model": MODEL_ID,
            "target_field": TARGET_FIELD,
            "lang": lang,
            "prompt_tokens": k,
            "sample_limit": SAMPLE_LIMIT,
            "total_evaluated": total,
            "hits": hits_total if not ONLY_EVAL_LOG_LIKELIHOOD else None,
            "hit_rate": rate,
            "mode": "loglik_only" if ONLY_EVAL_LOG_LIKELIHOOD else "full",
            "avg_target_loglik_sum": avg_ll_sum,
            "loglik_sum_ci95": [
                round(ci_lo, 6) if ci_lo is not None else None,
                round(ci_hi, 6) if ci_hi is not None else None
            ],
            "n_ll_samples": n_ll
        }

        print(f"[SUMMARY] {summary}", flush=True)
        with open(overall_path_jsonl, "a", encoding="utf-8") as fo:
            fo.write(json.dumps(summary, ensure_ascii=False) + "\n")

        if not ONLY_EVAL_LOG_LIKELIHOOD:
            print(f"[{lang} | {k} | {TARGET_FIELD}] total={total}, hits={hits_total}, rate={rate:.4f}")
            print(f"Hits saved to: {hit_examples_path}")
        if WRITE_LL_SAMPLES:
            print(f"Per-sample LL stream appended to: {ll_samples_path_jsonl}")

        return summary

    finally:
        if not ONLY_EVAL_LOG_LIKELIHOOD and fout is not None:
            fout.close()
        if ENABLE_LL_DEBUG and vtrace is not None:
            vtrace.close()
            print(f"LL trace log saved to: {os.path.join(RESULT_DIR, os.path.basename(vtrace.name))}")
        if fl_samples is not None:
            fl_samples.close()


# ====== main ======
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_path_jsonl = os.path.join(
        RESULT_DIR, f"overall_{MODEL_ID.split('/')[-1]}_{TARGET_FIELD}_{ts}.jsonl"
    )
    print(f"Overall stream file: {overall_path_jsonl}")

    ll_samples_path_jsonl = os.path.join(
        RESULT_DIR, f"ll_samples_{MODEL_ID.split('/')[-1]}_{TARGET_FIELD}_{ts}.jsonl"
    )

    header = {
        "model": MODEL_ID,
        "target_field": TARGET_FIELD,
        "timestamp": ts,
        "type": "header",
        "mode": "loglik_only" if ONLY_EVAL_LOG_LIKELIHOOD else "full"
    }
    with open(overall_path_jsonl, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(header, ensure_ascii=False) + "\n")
    if WRITE_LL_SAMPLES:
        with open(ll_samples_path_jsonl, "a", encoding="utf-8") as fl:
            fl.write(json.dumps(header, ensure_ascii=False) + "\n")

    for lang in LANGUAGES:
        for k in PROMPT_TOKEN_LENS:
            jsonl_path = os.path.join(PROMPT_DIR, f"{lang}_{k}.jsonl")
            _ = eval_one_file(jsonl_path, lang, k, overall_path_jsonl, ll_samples_path_jsonl)

    print(f"\nOverall stream appended to: {overall_path_jsonl}")


if __name__ == "__main__":
    main()
