# Personalized LLM Decoding via <u>Co</u>ntrasting <u>Pe</u>rsonal Preference

<div align="center">
  <img src="assets/emnlp_2025_logo_v1.png" alt="EMNLP 2025" style="height:60px;margin-bottom:16px;">
</div>

![CoPe teaser](assets/cope_teaser2.png)

<div align="center">
<p style="display:flex;justify-content:center;gap:24px;flex-wrap:wrap;margin:16px 0;width:100%;">
  <a href="https://naughtymaltiz16.github.io/cope_project_page/" target="_blank" style="display:inline-flex;align-items:center;gap:8px;padding:10px 20px;background:#2f2f2f;color:#ffffff;border-radius:9999px;text-decoration:none;font-weight:bold;letter-spacing:0.01em;">
    <span style="font-size:1.1rem;">💻</span>
    <span style="font-weight:bold;"> Project Page </span>
  </a>
  <span style="width:1px;height:24px;background:#666666;margin:0 8px;"></span>
  <a href="https://aclanthology.org/2025.emnlp-main.1723/" target="_blank" style="display:inline-flex;align-items:center;gap:8px;padding:10px 20px;background:#2f2f2f;color:#ffffff;border-radius:9999px;text-decoration:none;font-weight:bold;letter-spacing:0.01em;">
    <span style="font-size:1.1rem;">📄</span>
    <span style="font-weight:bold;"> Paper </span>
  </a>
</p>
</div>

CoPe is a decoding-time personalization framework for large language models (LLMs).
It maximizes implicit user reward by contrasting a personalized model (PEFT/LoRA tuned per user) with the base task-adapted model at token level — enabling personalization without external reward models or extra reward labeling.

This repository provides end-to-end scripts for:
- 🔧 Task-Adaptive Model (TAM) training on non-target users
- 👤 Per-user SFT adapters (OPPU)
- 🔄 Synthetic(or Pseudo) negative generation and selection for DPO
- 🎯 DPO fine-tuning per user
- 🚀 Inference with contrastive decoding (personal vs. base)


**📋 Tasks**
- Supported `--task_name`: `news_headline`, `scholarly_title`, `abstract_generation`, `review_writing`, `topic_writing`


## 📖 Introduction
We present CoPe, a decoding framework for LLM personalization by Contrasting Personal Preference (CoPe). The key idea is to incorporate implicit reward signals of user preference to guide both training (via DPO on selected negative pairs) and inference (via contrastive decoding that down-weights tokens preferred by the base model but disfavored by the personalized model).

![CoPe overview](assets/cope_overview.png)


## 📊 Dataset
- We utilize publicly available data from the LaMP and LongLaMP benchmarks and follow the OPPU setting.
- Download processed data and place under the repository root `./data`:
  - Google Drive: https://drive.google.com/file/d/147_uP-3A3XbEB8jwtaFkZXTXpLuybg8b/view?usp=sharing
- After extracting, you should have paths like `./data/<task_name>/user_top_100_history.json`.


## 💻 Install Requirements
Create a virtual environment (example with conda):

```bash
conda create -n cope python=3.9 -y
conda activate cope
```

Install dependencies from the repo root:

```bash
pip install -r requirements.txt
```

Change into the CoPe project directory before running the scripts:

```bash
cd CoPe
```

💡 **Note**
- A CUDA GPU is recommended. Some steps (e.g., vLLM sampling for synthetic(pseudo) negatives) may require multiple GPUs for speed.
- If you use private models on Hugging Face, set `--access_token` accordingly.


## 🔄 Workflow Overview
1) 🔧 Train TAM on non-target users to adapt the base model to the task domain.
2) 👤 Train per-user SFT adapters (OPPU) using each user's own history.
3) 🔄 Generate pseudo negatives per user and select best negatives by contrasting OPPU vs. TAM likelihoods.
4) 🎯 Run DPO with the selected negatives to refine per-user adapters.
5) 🚀 Inference with contrastive decoding: contrast OPPU (expert) vs. TAM (amateur) at token level.


## 1) 🔧 TAM (Task-Adaptive Model)
Task-Adaptive Model is trained on data excluding the target user. Outputs are saved as LoRA adapters.

▶️ Train TAM (example: news_headline):

```bash
python scripts/TAM.py \
  --task_name news_headline \
  --model_name mistralai/Mistral-7B-Instruct-v0.3 \
  --batch_size 8 \
  --max_epoch 3 \
  --is_train
```

▶️ Evaluate TAM (greedy) on held-out queries:

```bash
python scripts/TAM.py \
  --task_name news_headline \
  --model_name mistralai/Mistral-7B-Instruct-v0.3 \
  --repetition_penalty 1.0 \
  --is_test
```

📝 **Notes**
- TAM saves to `./ckpt/TAM/<task_name>/TAM-<model_name_short>_ckpt/`.


## 2) 👤 OPPU: Per-user SFT
Train one PEFT per user on their own history, initializing from TAM.

▶️ SFT training (per-user adapters):

```bash
python scripts/OPPU_sft.py \
  --task_name news_headline \
  --model_name mistralai/Mistral-7B-Instruct-v0.3 \
  --batch_size 4 \
  --max_epoch 2 \
  --is_train
```

▶️ SFT inference (greedy):

```bash
python scripts/OPPU_sft.py \
  --task_name news_headline \
  --model_name mistralai/Mistral-7B-Instruct-v0.3
```

▶️ SFT inference with contrastive decoding (expert=OPPU, amateur=TAM):

```bash
python scripts/OPPU_sft.py \
  --task_name news_headline \
  --model_name mistralai/Mistral-7B-Instruct-v0.3 \
  --is_cd \
  --contrastive_alpha 0.1 \
  --repetition_penalty 1.0
```

📤 **Outputs**
- Predictions are saved under `./output/<task_name>/OPPU-SFT-<model>-*.json`.


## 3) 🔄 Make DPO Negative Pairs
▶️ **Step 3.1**: Generate pseudo negatives with vLLM sampling per user.

```bash
python scripts/generate_pseudo_negatives.py \
  --task_name news_headline \
  --model_name mistralai/Mistral-7B-Instruct-v0.3 \
  --response_num 3 \
  --temperature 1.0 \
  --data_path ./data
```

✅ This creates `./data/<task_name>/user_top_100_history_with_pseudo_negatives.json` with multiple candidate responses per user.

▶️ **Step 3.2**: Score candidates and select best negatives by contrasting OPPU vs. TAM likelihoods.

```bash
python scripts/compute_scores.py \
  --task_name news_headline \
  --model_name mistralai/Mistral-7B-Instruct-v0.3 \
  --std max
```

✅ This writes `./data/<task_name>/user_top_100_history_with_pseudo_negatives_max.json` (or `_min.json`).


## 4) 🎯 DPO Training (per user)
Run DPO using the selected negatives to refine each user’s adapter.

```bash
python scripts/OPPU_sft+dpo.py \
  --task_name news_headline \
  --model_name mistralai/Mistral-7B-Instruct-v0.3 \
  --batch_size 8 \
  --dpo_beta 0.01 \
  --negative_sampling_method pseudo \
  --mode max \
  --is_train
```

📝 **Notes**
- Ensure TAM adapter is discoverable by this script. By default this code expects TAM at `./ckpt/TAM/<task_name>/TAM-<model>_ckpt/`.
- DPO outputs (adapters) are saved under `./ckpt/OPPU_SFT+DPO/<task_name>/user_<idx>/`.


## 5) 🚀 Inference with Contrastive Decoding
Use the DPO-refined per-user adapter as expert and TAM as amateur, decoding with per-token contrastive scores.

```bash
python scripts/OPPU_sft+dpo.py \
  --task_name news_headline \
  --model_name mistralai/Mistral-7B-Instruct-v0.3 \
  --is_cd \
  --contrastive_alpha 0.1 \
  --repetition_penalty 1.0
```

📤 **Outputs**
- Files like `OPPU-SFT+DPO-<model>-rp<...>-ca<...>-CD-run_*.json` under `./output/<task_name>/`.


## 📈 Evaluation
Evaluate predictions with the provided metrics script:

```bash
python eval/eval_task.py \
  --task <task_name> \
  --golds_json ./data/<task_name>/user_top_100_history_label.json \
  --preds_json ./output/<task_name>/<PRED_FILE>.json \
  --task_name <LaMP_ID> \
  --output_file ./output/<task_name>/<PRED_FILE>.json
```

Examples for `<LaMP_ID>`: `LaMP_4`, `LaMP_5`, `LongLaMP_2`, `LongLaMP_3`, `LongLaMP_4`.


## 💡 Tips and Troubleshooting
- Memory: prefer bfloat16 and gradient checkpointing; adjust `--batch_size` if OOM.
- Access tokens: if using gated models, pass `--access_token <HF_TOKEN>` for loading.
- Reproducibility: set seeds (already set to 42). For sampling, use `--is_sampling` and document runs.


## 📚 Citation
If you find this work useful, please cite:

```bibtex
@inproceedings{bu-etal-2025-personalized,
    title = "Personalized {LLM} Decoding via Contrasting Personal Preference",
    author = "Bu, Hyungjune  and
      Jung, ChanJoo  and
      Kang, Minjae  and
      Kim, Jaehyung",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1723/",
    pages = "33946--33966",
    ISBN = "979-8-89176-332-6"
}
```
