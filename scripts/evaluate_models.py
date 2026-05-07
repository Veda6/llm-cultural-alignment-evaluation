import json
import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
from bert_score import score as bertscore
from collections import defaultdict
from scipy.stats import entropy
from tqdm import tqdm

# === Settings ===
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
prompt_sets = ["prompts", "cultural_region", "global"]
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# === Load GPT outputs ===
print("Looking for:", os.path.abspath("results/gpt_outputs.json"))
with open("D:/Users/vedu0/unstable_foundations_capstone/results/gpt_outputs.json", "r", encoding="utf-8") as f:

#with open("results/gpt_outputs.json", "r", encoding="utf-8") as f:
    gpt_raw = json.load(f)

# Group GPT responses by prompt set and prompt
gpt_data = defaultdict(lambda: defaultdict(list))
for entry in gpt_raw:
    prompt_set = entry.get("prompt_set", "").replace("prompts_", "").replace("prompts", "prompts")  # normalize
    gpt_data[prompt_set][entry["prompt"]].append(entry["response"])


# === Helper functions ===
def calc_entropy(text):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return 0.0
    token_counts = np.bincount(tokens)
    token_probs = token_counts / token_counts.sum()
    return entropy(token_probs, base=2)

def calc_consistency(responses):
    if len(set(responses)) == 1:
        return 1.0
    P, R, F1 = bertscore(responses, [responses[0]]*len(responses), lang="en", rescale_with_baseline=True)
    return float(F1.mean())

# === Process each prompt set ===
for ps in prompt_sets:
    #with open(f"results/gemma_outputs_{ps}.json", "r", encoding="utf-8") as f:
    with open(f"../results/gemma_outputs_{ps}.json", "r", encoding="utf-8") as f:
        gemma_raw = json.load(f)

    # Group Gemma responses by prompt
    gemma_by_prompt = defaultdict(list)
    for entry in gemma_raw:
        gemma_by_prompt[entry["prompt"]].append(entry["response"])

    rows = []
    for prompt in tqdm(gpt_data[ps], desc=f"Processing {ps}"):
        gemma_responses = gemma_by_prompt.get(prompt, [])
        gpt_responses = gpt_data[ps].get(prompt, [])

        if len(gemma_responses) == 0 or len(gpt_responses) == 0:
            continue  # Skip incomplete pairs

        # === GEMMA metrics ===
        gemma_ent = np.mean([calc_entropy(r) for r in gemma_responses])
        gemma_con = calc_consistency(gemma_responses)

        # === GPT metrics ===
        gpt_ent = np.mean([calc_entropy(r) for r in gpt_responses])
        gpt_con = calc_consistency(gpt_responses)

        # === BERTScore: Gemma vs GPT (1-to-1, shortest match)
        min_len = min(len(gpt_responses), len(gemma_responses))
        bert_scores = []
        for i in range(min_len):
            _, _, f1 = bertscore([gemma_responses[i]], [gpt_responses[i]], lang="en", rescale_with_baseline=True)
            bert_scores.append(float(f1[0]))
        avg_bertscore = np.mean(bert_scores)

        # === Store rows ===
        rows.append({"prompt_set": ps, "model": "gemma", "entropy": gemma_ent, "consistency": gemma_con, "bertscore": avg_bertscore})
        rows.append({"prompt_set": ps, "model": "gpt", "entropy": gpt_ent, "consistency": gpt_con, "bertscore": np.nan})

    df = pd.DataFrame(rows)
    df.to_csv(f"{results_dir}/{ps}_metrics.csv", index=False)
    print(f"✅ Saved: {ps}_metrics.csv")

print("🎉 All done! Now ready for plotting.")
