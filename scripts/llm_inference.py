"""
scripts/llm_inference.py — API-based LLM inference with budget control and caching.
"""
import os, json, time, hashlib
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import (
    API_BUDGET, API_MODELS, BIO_PROMPT_TEMPLATE, LABELS, LABEL2ID,
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

def _cache_key(model, text):
    h = hashlib.md5(f"{model}:{text}".encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{model}_{h}.json")

def _load_cache(model, text):
    p = _cache_key(model, text)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None

def _save_cache(model, text, result):
    p = _cache_key(model, text)
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

def infer_openai(texts, model_name="gpt-4o", api_key=None):
    """Run OpenAI inference on a batch of texts."""
    global _total_spent
    try:
        from openai import OpenAI
    except ImportError:
        print("[WARN] openai package not installed, skipping")
        return [None] * len(texts)

    if not api_key:
        api_key = OPENAI_API_KEY
    if not api_key:
        print(f"[SKIP] No API key for {model_name}")
        return [None] * len(texts)

    client = OpenAI(api_key=api_key)
    results = []

    for text in texts:
        # Check cache
        cached = _load_cache(model_name, text)
        if cached is not None:
            results.append(cached)
            continue

        # Check budget
        if _total_spent >= API_BUDGET.max_budget_usd:
            print(f"[BUDGET] Exceeded ${API_BUDGET.max_budget_usd}, skipping remaining")
            results.append(None)
            continue

        prompt = BIO_PROMPT_TEMPLATE.format(sentence=text)
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
                _save_cache(model_name, text, result)
                results.append(result)
                break
            except Exception as e:
                if attempt < API_BUDGET.max_retries - 1:
                    time.sleep(API_BUDGET.retry_delay * (attempt + 1))
                else:
                    print(f"[ERROR] {model_name} failed on: {text[:50]}... -> {e}")
                    results.append(None)

        # Rate limiting
        time.sleep(60.0 / API_BUDGET.requests_per_minute)

    return results

def parse_bio_output(output_str, num_tokens):
    """Parse LLM output into BIO tag IDs."""
    if output_str is None:
        return [LABEL2ID["O"]] * num_tokens

    tags_str = output_str.strip().split()
    tag_ids = []
    for t in tags_str:
        t = t.strip().upper()
        if t in LABEL2ID:
            tag_ids.append(LABEL2ID[t])
        else:
            tag_ids.append(LABEL2ID["O"])

    # Pad or truncate
    while len(tag_ids) < num_tokens:
        tag_ids.append(LABEL2ID["O"])
    tag_ids = tag_ids[:num_tokens]
    return tag_ids

def run_api_inference(model_key, samples, batch_size=None):
    """
    Run API inference on samples. Returns list of tag_id lists.
    samples: list of dicts with 'text' and 'tokens' keys.
    """
    if batch_size is None:
        batch_size = API_BUDGET.batch_size

    available, api_key = check_api_key(model_key)
    if not available:
        print(f"[SKIP] No API key for {model_key}")
        return [None] * len(samples)

    info = API_MODELS[model_key]
    provider = info["provider"]
    model_name = info["model"]

    texts = [s["text"] for s in samples]
    all_results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        if provider == "openai":
            batch_results = infer_openai(batch, model_name, api_key)
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
