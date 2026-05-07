from openai import OpenAI
import openai
import pandas as pd
import json
import os
import random
from datetime import datetime
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# Define prompt files
prompt_files = {
    "general": "data/prompts.csv",
    "cultural": "data/cultural_region_prompts.csv",
    "global": "data/prompts_global.csv",
    "african": "data/african_prompts.csv"
}

# Load all prompts into a single dataframe
all_prompts = []

for category, path in prompt_files.items():
    df = pd.read_csv(path)
    df['category'] = category
    all_prompts.append(df)

df_all = pd.concat(all_prompts, ignore_index=True)
models = ["gpt-3.5-turbo"]  # You can later add "gpt-4"
temperatures = [0.7, 1.0]
seeds = [0, 1, 2, 3, 4]  # Simulate randomness via loop
results = []

for model in models:
    for temperature in temperatures:
        for seed in seeds:
            for _, row in df_all.iterrows():
                prompt = row["prompt"]
                topic = row["topic"]
                category = row["category"]
                region = row.get("region", None)

                print(f"Model: {model} | Temp: {temperature} | Seed: {seed} | Prompt: {prompt[:60]}...")

                try:
                    # Simulate randomization
                    random.seed(seed)
                    client = OpenAI()
                    openai_response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=300
                    )

                    response_text = openai_response.choices[0].message.content

                    results.append({
                        "model": model,
                        "temperature": temperature,
                        "seed": seed,
                        "topic": topic,
                        "category": category,
                        "region": region,
                        "prompt": prompt,
                        "response": response_text,
                        "timestamp": datetime.now().isoformat()
                    })

                except Exception as e:
                    print("⚠️ Error:", e)
# Save results
os.makedirs("results", exist_ok=True)
with open("results/gpt_outputs.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("✅ Sampling complete. Results saved to results/gpt_outputs.json")
