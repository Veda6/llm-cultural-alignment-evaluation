# Unstable Foundations: Cultural Alignment Evaluation in LLMs

This project evaluates how well large language models (GPT, LLaMA, Gemma) align with culturally diverse prompts using quantitative metrics such as entropy, consistency, and BERTScore.

## Problem Statement
LLMs are often evaluated using general benchmarks, but cultural alignment remains underexplored. 
This project investigates whether models respond consistently and appropriately across culturally diverse prompts.

## Objectives
- Evaluate cultural alignment across multiple LLMs
- Compare outputs using entropy, consistency, and BERTScore
- Analyze variation across cultural regions
- Identify reliability gaps in evaluation metrics

## Methodology

### Models Evaluated
- GPT-3.5 (OpenAI API)
- LLaMA 2 7B Chat (Hugging Face)
- Gemma 2B Instruct (Hugging Face)

### Prompt Sets
- General prompts
- Cultural region prompts
- Global prompts
- African prompts

### Pipeline
1. Prompt collection and curation
2. Response generation (multi-model sampling)
3. Metric computation:
   - Entropy (diversity)
   - Consistency (response stability)
   - BERTScore (semantic similarity)
4. Statistical analysis (correlation, ANOVA)

## Evaluation Metrics

- **Entropy**: Measures variability in model responses
- **Consistency**: Measures how stable responses are across runs
- **BERTScore**: Measures semantic similarity using contextual embeddings

## Results

- Entropy increases for culturally nuanced prompts, indicating higher response variability
- Consistency drops across models when handling region-specific queries, highlighting instability in model behavior
- BERTScore remains high even when cultural nuance is misaligned, suggesting semantic similarity alone is insufficient

Visualizations (see `/results/plots`):
- Correlation heatmaps
- Entropy vs Consistency scatter plots
- Distribution plots across prompt sets

## Key Insights

- High semantic similarity does not imply cultural correctness
- Models show instability in culturally specific prompts
- Evaluation metrics may not fully capture cultural alignment

## Reproducibility

### 1. Clone the repository

```bash
git clone https://github.com/your-username/llm-cultural-alignment-evaluation.git
cd llm-cultural-alignment-evaluation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up API keys

This project uses:

* OpenAI API (for GPT)
* Hugging Face API (for Gemma and LLaMA)

#### OpenAI

```bash
export OPENAI_API_KEY=your_openai_api_key
```

#### Hugging Face

```bash
export HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

> Access to LLaMA 2 7B Chat requires approval on Hugging Face.  
> Ensure your account has been granted access before running the pipeline.

---

### 4. Generate model responses

```bash
python scripts/run_sampling.py
```

---

### 5. Compute evaluation metrics

```bash
python scripts/evaluate_models.py
```

---

### 6. Run analysis

Open:

```bash
notebooks/analysis.ipynb
```

---

## Challenges and Troubleshooting

- Managed variability in LLM outputs by using multi-sampling with different temperatures and seeds
- Resolved inconsistencies in response formatting across GPT, LLaMA, and Gemma outputs
- Handled gated model access and environment setup issues for LLaMA 2 via Hugging Face
- Reduced evaluation noise by aggregating responses and comparing metrics across prompt sets
- Observed that high semantic similarity (BERTScore) did not always correspond to culturally aligned responses, motivating cross-metric analysis
### Notes

* API usage may incur cost (OpenAI) and rate limits (Hugging Face)
* Expected runtime: ~X minutes depending on prompt size


## References

1. Zhang et al. (2019). *BERTScore: Evaluating Text Generation with BERT.*
2. Anthropic Helpful-Harmless (HH) Dataset
3. Hugging Face Transformers Documentation
4. OpenAI API Documentation
5. Wang et al. (2021). *RobOT: Robustness-Oriented Testing for Deep Learning Systems.*
6. Stanford Human-Centered AI resources on cultural evaluation and alignment
