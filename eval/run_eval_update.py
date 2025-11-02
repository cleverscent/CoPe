# run_eval_update.py
import json, os, glob, numpy as np
import evaluate
rouge_metric = evaluate.load('rouge')

# 각 태스크 폴더 내에서 평가를 진행할 파일 이름 목록 (기존 PRED_FILES 목록)
PRED_FILENAMES = [
    "TAM.json",
    "OPPU_SFT.json",
    "OPPU_SFT+CD.json",
    "OPPU_SFT+DPO.json",
    "OPPU_SFT+DPO+CD.json",
]
REF_FILENAME = "user_top_100_history_label.json"  # 참조 파일 이름

Q10_DIR = "Q10"  # eval 폴더 기준 상대 경로
TASK_DIRS = glob.glob(f"{Q10_DIR}/*/") # Q10 하위의 task 폴더들



def rouge_per_sample(pred, ref):
    try:
        scores = rouge_metric.compute(
            predictions=[pred], references=[ref], use_aggregator=False
        )
        return float(scores['rouge1'][0]), float(scores['rougeL'][0])
    except Exception as e:
        print("ROUGE 계산 오류:", e)
        return 0.0, 0.0


if not TASK_DIRS:
    print(f"No task directories found in {Q10_DIR}")

for task_dir in sorted(TASK_DIRS):
    task_name = os.path.basename(os.path.dirname(task_dir))
    print(f"\nProcessing task: {task_name}...")

    REF_PATH = os.path.join(task_dir, REF_FILENAME)
    if not os.path.exists(REF_PATH):
        print(f"  [SKIP] Reference file not found: {REF_PATH}")
        continue

    # ─── 1) gold 로드 & dict 화 ───
    with open(REF_PATH, 'r', encoding='utf-8') as f:
        gold_task = json.load(f)
    ref_dict = {g["id"]: g["output"] for g in gold_task["golds"]}
    
    # PRED_FILENAMES 목록에 있는 파일들에 대해서만 평가 수행
    for pred_fname in sorted(PRED_FILENAMES):
        fp = os.path.join(task_dir, pred_fname)

        if not os.path.exists(fp):
            # print(f"  - Prediction file not found, skipping: {pred_fname}")
            continue

        with open(fp, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)

        if pred_data["task"] != gold_task["task"]:
            print(f"  [SKIP] Task mismatch in {os.path.basename(fp)}")
            continue

        r1_list, rL_list = [], []
        for samp in pred_data["golds"]:
            sid, pred_txt = samp["id"], samp["output"]
            if sid not in ref_dict:
                continue
            ref_txt = ref_dict[sid]
            r1, rL = rouge_per_sample(pred_txt, ref_txt)
            samp["rouge-1"], samp["rouge-L"] = r1, rL
            r1_list.append(r1)
            rL_list.append(rL)

        # ─── 2) 통계 추가 ───
        if r1_list:
            n = len(r1_list)  # 샘플 수

            r1_mean = float(np.mean(r1_list))
            r1_std  = float(np.std(r1_list, ddof=1))  # sample std
            r1_se   = float(r1_std / (n ** 0.5))      # ★ 표준오차

            rL_mean = float(np.mean(rL_list))
            rL_std  = float(np.std(rL_list, ddof=1))
            rL_se   = float(rL_std / (n ** 0.5))      # ★ 표준오차

            pred_data["stats"] = {
                "rouge-1_mean": r1_mean,
                "rouge-1_std":  r1_std,
                "rouge-1_se":   r1_se,      # ← 추가
                "rouge-L_mean": rL_mean,
                "rouge-L_std":  rL_std,
                "rouge-L_se":   rL_se       # ← 추가
            }

        # ─── 3) 파일 덮어쓰기 ───
        with open(fp, "w", encoding='utf-8') as f:
            json.dump(pred_data, f, ensure_ascii=False, indent=2)

        print(f"  →  {os.path.basename(fp)} done")

print("\nAll tasks completed.")
