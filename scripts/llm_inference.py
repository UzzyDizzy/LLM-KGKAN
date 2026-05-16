"""
scripts/llm_inference.py — API-based LLM inference with budget control and caching.
"""
import os, json, time, hashlib
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import (
    API_BUDGET, API_MODELS, BIO_PROMPT_TEMPLATE, LABELS, LABEL2ID, ID2LABEL,
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, RESULTS_DIR,
)

CACHE_DIR = os.path.join(RESULTS_DIR, "api_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Track spending
_total_spent = 0.0
# Approximate costs per 1M tokens (input/output)
COST_PER_1M = {
    "gpt-4o": (2.5, 10.0),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "gemini-2.5-pro": (1.25, 10.0),
}

def _cache_key(model, text, context=""):
    h = hashlib.md5(f"{model}:{context}:{text}".encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{model}_{h}.json")

def _load_cache(model, text, context=""):
    p = _cache_key(model, text, context)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None

def _save_cache(model, text, result, context=""):
    p = _cache_key(model, text, context)
    with open(p, "w") as f:
        json.dump(result, f)

def check_api_key(model_key):
    """Check if API key is available for a model. Returns (available, key)."""
    info = API_MODELS.get(model_key, {})
    key_env = info.get("key_env", "")
    key = os.environ.get(key_env, "")
    return bool(key), key

def estimate_cost(num_samples, avg_tokens=80, model="gpt-4o"):
    """Estimate cost for inference."""
    input_tok = num_samples * (avg_tokens + 150)  # prompt overhead
    output_tok = num_samples * avg_tokens
    costs = COST_PER_1M.get(model, (5.0, 15.0))
    return (input_tok * costs[0] + output_tok * costs[1]) / 1_000_000

def _build_prompt(sample, few_shot_str=""):
    tokens = sample.get("tokens") or str(sample.get("text", "")).split()
    sentence = " ".join(tokens)
    prompt = BIO_PROMPT_TEMPLATE.format(sentence=sentence)
    if few_shot_str:
        prompt = (
            "Examples:\n"
            f"{few_shot_str}\n"
            "Now tag the following sentence.\n\n"
            f"{prompt}"
        )
    prompt += f"\n\nTokenized input: {json.dumps(tokens)}\nNumber of tokens: {len(tokens)}"
    return prompt


def infer_openai(samples, model_name="gpt-4o", api_key=None, few_shot_str="", cache_context=""):
    """Run OpenAI inference on a batch of ABSA samples."""
    global _total_spent
    try:
        from openai import OpenAI
    except ImportError:
        print("[WARN] openai package not installed, skipping")
        return [None] * len(samples)

    if not api_key:
        api_key = OPENAI_API_KEY
    if not api_key:
        print(f"[SKIP] No API key for {model_name}")
        return [None] * len(samples)

    client = OpenAI(api_key=api_key)
    results = []

    import logging
    logger = logging.getLogger()
    logger.info(f"  [OpenAI] Starting inference on {len(samples)} samples")

    for sample in samples:
        text = sample.get("text") or " ".join(sample.get("tokens", []))
        cached = _load_cache(model_name, text, cache_context)
        if cached is not None:
            results.append(cached)
            continue

        # Check budget
        if _total_spent >= API_BUDGET.max_budget_usd:
            print(f"[BUDGET] Exceeded ${API_BUDGET.max_budget_usd}, skipping remaining")
            results.append(None)
            continue

        prompt = _build_prompt(sample, few_shot_str)

        for attempt in range(API_BUDGET.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0,
                )
                output = resp.choices[0].message.content.strip()
                # Estimate cost
                in_tok = resp.usage.prompt_tokens if resp.usage else 100
                out_tok = resp.usage.completion_tokens if resp.usage else 50
                costs = COST_PER_1M.get(model_name, (5.0, 15.0))
                cost = (in_tok * costs[0] + out_tok * costs[1]) / 1_000_000
                _total_spent += cost

                result = {"text": text, "output": output, "cost": cost}
                _save_cache(model_name, text, result, cache_context)
                results.append(result)
                break
            except Exception as e:
                if attempt < API_BUDGET.max_retries - 1:
                    logger.info(f"    [OpenAI] Rate limiting... (sleeping {60.0/API_BUDGET.requests_per_minute:.1f}s)")
                    time.sleep(60.0 / API_BUDGET.requests_per_minute)
                else:
                    print(f"[ERROR] {model_name} failed on: {text[:50]}... -> {e}")
                    results.append(None)

        # Rate limiting
        time.sleep(60.0 / API_BUDGET.requests_per_minute)

    return results

import json
def parse_bio_output(output_str, num_tokens):
    """Parse LLM output into BIO tag IDs."""
    import logging
    logger = logging.getLogger()

    if output_str is None:
        return [LABEL2ID["O"]] * num_tokens
    logger.info(f"    Raw LLM output: {output_str[:100]}")

    try:
        if "```" in output_str:
            output_str = output_str.replace("```json", "```").split("```")[1]
        parsed = json.loads(output_str.strip())
        if isinstance(parsed, dict):
            parsed = parsed.get("tags", parsed.get("labels", []))
        tags_str = parsed if isinstance(parsed, list) else str(parsed).split()
    except Exception:
        tags_str = output_str.replace(",", " ").replace("[", " ").replace("]", " ").split()

    tag_ids = []
    for t in tags_str:
        t = str(t).strip().strip('"').strip("'").upper()
        if t in LABEL2ID:
            tag_ids.append(LABEL2ID[t])
        else:
            tag_ids.append(LABEL2ID["O"])

    # Pad or truncate
    while len(tag_ids) < num_tokens:
        tag_ids.append(LABEL2ID["O"])
    tag_ids = tag_ids[:num_tokens]
    return tag_ids

def run_api_inference(model_key, samples, train_samples=None, k_shot=0, batch_size=None):
    """
    Run API inference on samples. Returns list of tag_id lists.
    samples: list of dicts with 'text' and 'tokens' keys.
    """
    import logging
    logger = logging.getLogger()
    logger.info(f"[API] Starting inference for {len(samples)} samples with {model_key}")
    # 1. Build the few_shot_examples string from `train_samples`
    few_shot_str = ""
    if k_shot > 0 and train_samples:
        import random
        shots = random.sample(train_samples, min(k_shot, len(train_samples)))
        for shot in shots:
            shot_tags = [ID2LABEL[tid] for tid in shot['tag_ids'][:len(shot['tokens'])]]
            few_shot_str += f"Tokens: {json.dumps(shot['tokens'])}\nTags: {json.dumps(shot_tags)}\n\n"

    if batch_size is None:
        batch_size = API_BUDGET.batch_size

    available, api_key = check_api_key(model_key)
    if not available:
        print(f"[SKIP] No API key for {model_key}")
        return [None] * len(samples)

    info = API_MODELS[model_key]
    provider = info["provider"]
    model_name = info["model"]

    all_results = []

    cache_context = f"k{k_shot}:{hashlib.md5(few_shot_str.encode()).hexdigest()}"
    for i in range(0, len(samples), batch_size):
        logger.info(f"  [API] Processing batch {i//batch_size + 1}/{(len(samples)-1)//batch_size + 1}")
        batch = samples[i:i+batch_size]
        if provider == "openai":
            batch_results = infer_openai(
                batch,
                model_name=model_name,
                api_key=api_key,
                few_shot_str=few_shot_str,
                cache_context=cache_context,
            )
        else:
            # Placeholder for anthropic/google — skip if no implementation
            print(f"[SKIP] Provider '{provider}' not implemented yet")
            batch_results = [None] * len(batch)
        all_results.extend(batch_results)

    # Parse outputs to tag IDs
    tag_ids_list = []
    for j, r in enumerate(all_results):
        n_tok = len(samples[j]["tokens"])
        if r is not None and "output" in r:
            tag_ids_list.append(parse_bio_output(r["output"], n_tok))
        else:
            tag_ids_list.append(None)

    return tag_ids_list

def get_total_spent():
    return _total_spent
