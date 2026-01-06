import os
import json
from pathlib import Path
from typing import Dict, Optional, List

import argparse
from tqdm import tqdm
import torch

from transformers import MT5Tokenizer, GPT2LMHeadModel

MODEL_ID = "ai-forever/mGPT-13B"


# =======================
# Defaults (can be overridden by CLI)
# =======================

DEFAULT_PII_TYPE = "email"  # "email" or "phone"

# Prompt template file (jsonl). Each line:
# {"language": "en", "email": "...", "phone": "..."}
DEFAULT_PROMPT_TEMPLATE_PATH = "workplace/MONOPII/pii_prompt_templates.jsonl"

DEFAULT_OUTPUT_DIR = "pii_outputs_600m"
DEFAULT_NUM_SAMPLES_PER_CONFIG = 20_000
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TOP_K = 40
DEFAULT_BATCH_SIZE = 500

# Whether to write a preview file (small samples for sanity check)
DEFAULT_SAVE_PREVIEW = False
DEFAULT_PREVIEW_INTERVAL = 500

# Flush buffer to disk every N records to reduce IO overhead
DEFAULT_WRITE_INTERVAL = 500


def load_prompts_from_jsonl(template_path: str, pii_type: str) -> Dict[str, str]:
    """
    Load prompts from a jsonl template file.
    Each line is a JSON object like:
        {
            "language": "en",
            "email": "...",
            "phone": "..."
        }

    Returns:
        dict: { language_code: prompt_for_given_pii_type }
    """
    prompts: Dict[str, str] = {}

    with open(template_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            lang = obj.get("language")
            if not lang:
                continue

            if pii_type not in obj:
                # Skip if the requested PII type is missing in this record
                continue

            prompt = obj[pii_type]
            if not isinstance(prompt, str) or not prompt.strip():
                continue

            prompts[lang] = prompt.strip()

    if not prompts:
        raise ValueError(
            f"No prompts loaded for PII type '{pii_type}' from template file: {template_path}"
        )

    print(f"Loaded {len(prompts)} prompts for PII type='{pii_type}' from {template_path}")
    print("Languages in template:", ", ".join(sorted(prompts.keys())))
    return prompts


def load_model_and_tokenizer(model_id: str = MODEL_ID):
    """
    Load model + tokenizer. Uses MT5Tokenizer and GPT2LMHeadModel as in your original code.
    """
    print(f"Loading model and tokenizer: {model_id}")

    tokenizer = MT5Tokenizer.from_pretrained(model_id)

    # Some GPT-like models may not have pad_token; fallback to eos_token when possible
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def generate_for_config(
    model,
    tokenizer,
    language: str,
    pii_type: str,
    prompt: str,
    num_samples: int,
    output_dir: str,
    batch_size: int,
    max_new_tokens: int,
    top_k: int,
    save_preview: bool,
    preview_interval: int,
    write_interval: int,
):
    """
    Generate completions for one (language, pii_type) configuration and save to jsonl.
    """
    file_name = f"mgpt600M_{language}_{pii_type}.jsonl"
    output_path = Path(output_dir) / file_name

    print(f"\nGenerating for language={language}, pii_type={pii_type}")
    print(f"Prompt: {prompt}")
    print(f"Saving to {output_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Remove existing output to avoid mixing runs
    if output_path.exists():
        output_path.unlink()

    # Optional preview output file
    if save_preview:
        preview_path = Path(output_dir) / f"preview_{language}_{pii_type}.jsonl"
        if preview_path.exists():
            preview_path.unlink()
        f_prev = open(preview_path, "w", encoding="utf-8")
    else:
        preview_path = None
        f_prev = None

    f_out = open(output_path, "w", encoding="utf-8")

    num_generated = 0
    pbar = tqdm(total=num_samples, desc=f"{language}-{pii_type}")

    # Accumulate lines in memory and flush periodically to reduce IO overhead
    buffer: List[str] = []

    while num_generated < num_samples:
        current_batch_size = min(batch_size, num_samples - num_generated)
        prompt_batch = [prompt] * current_batch_size

        inputs = tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                top_k=top_k,
                min_new_tokens=max_new_tokens,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for full_text in generated_texts:
            # Try to strip the prompt prefix to keep only the completion
            if full_text.startswith(prompt):
                completion = full_text[len(prompt):].lstrip()
            else:
                completion = full_text

            record = {
                "language": language,
                "pii_type": pii_type,
                "completion": completion,
            }

            buffer.append(json.dumps(record, ensure_ascii=False) + "\n")

            # Flush periodically
            if len(buffer) >= write_interval:
                f_out.writelines(buffer)
                f_out.flush()
                buffer.clear()

            # Print / write preview periodically (if enabled)
            if save_preview and (num_generated % preview_interval == 0):
                print(f"\n=== SAMPLE PREVIEW (Every {preview_interval} samples) ===")
                print(f"[{language}-{pii_type}]")
                print(completion[:300])
                print("=================================\n")

                if f_prev is not None:
                    f_prev.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_prev.flush()

            num_generated += 1
            pbar.update(1)

            if num_generated >= num_samples:
                break

    # Flush remaining lines
    if buffer:
        f_out.writelines(buffer)
        f_out.flush()

    f_out.close()
    if f_prev is not None:
        f_prev.close()
    pbar.close()

    print(f"Done: {language}-{pii_type}")
    print(f"Saved {num_samples} samples to {output_path}")
    if save_preview and preview_path is not None:
        print(f"Preview written to {preview_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pii-type",
        type=str,
        default=DEFAULT_PII_TYPE,
        choices=["email", "phone"],
        help="Which PII type to generate: 'email' or 'phone'.",
    )

    parser.add_argument(
        "--template-path",
        type=str,
        default=DEFAULT_PROMPT_TEMPLATE_PATH,
        help="Path to the jsonl prompt template file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save generated jsonl files.",
    )

    parser.add_argument(
        "--languages",
        type=str,
        default=None,
        help=(
            "Comma-separated language codes to run, e.g. 'en,zh,fr'. "
            "If not set, use all languages from the template file."
        ),
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES_PER_CONFIG,
        help="Number of samples to generate for each language configuration.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for generation.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Number of tokens to generate per sample.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-k sampling parameter.",
    )

    parser.add_argument(
        "--save-preview",
        action="store_true",
        default=DEFAULT_SAVE_PREVIEW,
        help="Enable writing preview samples to a separate file.",
    )

    parser.add_argument(
        "--preview-interval",
        type=int,
        default=DEFAULT_PREVIEW_INTERVAL,
        help="Write/print one preview every N samples (if --save-preview enabled).",
    )

    parser.add_argument(
        "--write-interval",
        type=int,
        default=DEFAULT_WRITE_INTERVAL,
        help="Flush buffered records to disk every N samples.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    pii_type = args.pii_type
    template_path = args.template_path
    output_dir = args.output_dir

    # 1) Load prompts for the requested PII type
    prompts_by_lang = load_prompts_from_jsonl(
        template_path=template_path,
        pii_type=pii_type,
    )

    # 2) Filter languages from CLI if provided
    target_langs: Optional[List[str]] = None
    if args.languages is not None and args.languages.strip():
        target_langs = [x.strip() for x in args.languages.split(",") if x.strip()]
        # Deduplicate while preserving order
        target_langs = list(dict.fromkeys(target_langs))
        print("Target languages from CLI:", ", ".join(target_langs))

    # 3) Load model
    model, tokenizer = load_model_and_tokenizer(MODEL_ID)

    # 4) Generate for each language
    for language, prompt in prompts_by_lang.items():
        if target_langs is not None and language not in target_langs:
            continue

        generate_for_config(
            model=model,
            tokenizer=tokenizer,
            language=language,
            pii_type=pii_type,
            prompt=prompt,
            num_samples=args.num_samples,
            output_dir=output_dir,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            save_preview=args.save_preview,
            preview_interval=args.preview_interval,
            write_interval=args.write_interval,
        )

    print("All generations finished.")


if __name__ == "__main__":
    main()
