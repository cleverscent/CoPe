from evaluation import LaMPEvaluation
import argparse
import json
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument("--task", default="news_headline", help="")
parser.add_argument("--golds_json", default='/home/sensiblescent428/OPPU/data/news_headline/user_top_100_history_label.json', help="Address to all gold labels for the task as a json file")
parser.add_argument("--preds_json",  default='/home/sensiblescent428/OPPU/output/news_headline/OPPU-SFT+DPO-Mistral-7B-Instruct-v0.3-rp1.0-beta0.01-max_epoch1-ca0.3-CD-run_3.json', help="Address to all predictions for the task as a json file")
parser.add_argument("--task_name", default='LaMP_4', help="[LaMP_4, LaMP_5, LongLaMP_2, LongLaMP_3, LongLaMP_4]")
parser.add_argument("--output_file", default='/home/sensiblescent428/OPPU/output/news_headline/OPPU-SFT+DPO-Mistral-7B-Instruct-v0.3-rp1.0-beta0.01-max_epoch1-ca0.3-CD-run_3.json', help="Address to the results file")

# if __name__ == "__main__":

#     opts = parser.parse_args()

#     evaluator = LaMPEvaluation(single_gold_json_file_addr=opts.golds_json)
#     results = evaluator.evaluate_task(opts.preds_json, opts.task_name)

#     with open(opts.output_file, 'r') as file:
#         data = json.load(file)

#     data["results"] = results

#     with open(opts.output_file, "w") as file:
#         json.dump(data, file, indent=2)

from collections import defaultdict
import numpy as np

if __name__ == "__main__":
    opts = parser.parse_args()
    evaluator = LaMPEvaluation(single_gold_json_file_addr=opts.golds_json)

    all_results = defaultdict(list)

    # Run evaluation 10 times
    for _ in range(10):
        result = evaluator.evaluate_task(opts.preds_json, opts.task_name)
        for k, v in result.items():
            all_results[k].append(v)

    # Compute average for each metric
    averaged_results = {k: float(np.mean(v)) for k, v in all_results.items()}

    # Load original data
    with open(opts.output_file, 'r') as file:
        data = json.load(file)

    data["results"] = averaged_results

    with open(opts.output_file, "w") as file:
        json.dump(data, file, indent=2)


