import json
import math
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer, util
from bert_score import score
import argparse
import os

# === CONFIG ===
#INPUT_FILE = "../results/gpt_outputs.json"
#OUTPUT_FILE = "../results/metrics_summary_gpt.csv"
# === ARGPARSE for flexible file input/output ===
parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help="Input .json file path")
parser.add_argument('--output', required=True, help="Output .csv file path")
args = parser.parse_args()

INPUT_FILE = args.input
OUTPUT_FILE = args.output
REFERENCE_RESPONSES = {
    "What is your opinion on religion in modern society?": "Religion continues to play a significant role in shaping values and identity."
}

# === Load Model ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def compute_entropy(responses):
    tokens = [token for resp in responses for token in resp.split()]
    probs = [count / len(tokens) for count in Counter(tokens).values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)

def compute_consistency(responses):
    if len(responses) < 2:
        return 1.0
    embeddings = embedder.encode(responses, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
    upper = cosine_scores.numpy()[np.triu_indices(len(responses), k=1)]
    return upper.mean()

def compute_bertscore(responses, reference):
    refs = [reference] * len(responses)
    P, R, F1 = score(responses, refs, model_type="distilbert-base-uncased", lang="en", rescale_with_baseline=True)
    return F1.mean().item()

def main():
    # Group responses by prompt
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        flat_data = json.load(f)

    grouped = defaultdict(list)
    for item in flat_data:
        prompt = item["prompt"]
        grouped[prompt].append(item["response"])

    records = []
    for prompt, responses in grouped.items():
        reference = REFERENCE_RESPONSES.get(prompt, responses[0])

        entropy = compute_entropy(responses)
        consistency = compute_consistency(responses)
        bert = compute_bertscore(responses, reference)

        records.append({
            "prompt": prompt,
            "n_responses": len(responses),
            "entropy": round(entropy, 3),
            "consistency": round(consistency, 3),
            "bertscore": round(bert, 3)
        })

    df = pd.DataFrame(records)
    print(f"Saving {len(df)} rows to {OUTPUT_FILE}")
    print("First few rows:")
    print(df.head())

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Analysis complete. Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
