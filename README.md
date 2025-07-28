# Identifying Fine-grained Forms of Populism in Political Discourse: A Case Study on Donald Trump’s Presidential Campaigns
This repository contains the code used for the study "Identifying Fine-grained Forms of Populism in Political Discourse: A Case Study on Donald Trump’s Presidential Campaigns" ([Chalkidis et al., 2025](https://arxiv.org/abs/2507.19303)).

## Summary
Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of instruction-following tasks, yet their grasp of nuanced social science concepts remains underexplored. This paper examines whether LLMs can identify and classify fine-grained forms of populism, a complex and contested concept in both academic and media debates. To this end, we curate and release novel datasets specifically designed to capture populist discourse. We evaluate a range of pre-trained (large) language models, both open-weight and proprietary, across multiple prompting paradigms. Our analysis reveals notable variation in performance, highlighting the limitations of LLMs in detecting populist discourse. We find that a fine-tuned RoBERTa classifier vastly outperforms all new-era instruction-tuned LLMs, unless fine-tuned. Additionally, we apply our best-performing model to analyze campaign speeches by Donald Trump, extracting valuable insights into his strategic use of populist rhetoric. Finally, we assess the generalizability of these models by benchmarking them on campaign speeches by European politicians, offering a lens into cross-context transferability in political discourse analysis. In this setting, we find that instruction-tuned LLMs exhibit greater robustness on out-of-domain data.

## Datasets

| Dataset Name | Alias | Link |
| ------------ | ---- | ---- |
| TRUMP-2016 | `coastalcph/populism-trump-2016` | [Link](https://huggingface.co/datasets/coastalcph/populism-trump-2016) |
 | TRUMP-CHRONOS | `coastalcph/populism-trump-chronos` | [Link](https://huggingface.co/datasets/coastalcph/populism-trump-chronos)
| EU-OOD | `coastalcph/populism-eu-speeches` | [Link](https://huggingface.co/datasets/coastalcph/populism-eu-speeches)

# Load Dataset

```python 
from datasets import load_dataset
train_dataset = load_dataset('coastalcph/populism-eu-speeches', split='train')
```

## Main Experiments

- Audit open-weight LLMs using the [audit_llms.py](audit_llms/audit_llms.py) script.
- Fine-tune open-weight LLMs, e.g., Qwen 3 models, using the  [finetune_plms.py](finetune_models%2Ffinetune_plms.py) script.
- Fine-tune PLMs, e.g., RoBERTa, using the [finetune_plms.py](finetune_models%2Ffinetune_plms.py) script.

## Citation

```
@misc{chalkidis-et-al-2025-populism,
    title = "Identifying Fine-grained Forms of Populism in Political Discourse: A Case Study on Donald Trump’s Presidential Campaigns",
    author = "Chalkidis, Ilias and Brandl, Stephanie and Aslanidis, Paris",
    year = "2025",
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2507.19303}
}
```
