import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Setup ===
sns.set(style="whitegrid")
results_dir = "../results"
prompt_sets = ["prompts", "cultural_region", "global"]
metrics = ["entropy", "consistency", "bertscore"]
colors = {"gpt": "#4C72B0", "gemma": "#55A868"}

# === Load & Combine Data ===
all_data = []

for ps in prompt_sets:
    path = os.path.join(results_dir, f"{ps}_metrics.csv")
    df = pd.read_csv(path)
    df["prompt_set"] = ps
    all_data.append(df)

df_all = pd.concat(all_data)

# === Plotting ===
for metric in metrics:
    plt.figure(figsize=(8, 5))
    plot_df = df_all.dropna(subset=[metric])
    
    sns.barplot(
        data=plot_df,
        x="prompt_set",
        y=metric,
        hue="model",
        palette=colors,
        errorbar="sd"
    )

    plt.title(f"Average {metric.capitalize()} by Model and Prompt Set")
    plt.ylabel(metric.capitalize())
    plt.xlabel("Prompt Set")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"plot_{metric}.png"))
    plt.show()
