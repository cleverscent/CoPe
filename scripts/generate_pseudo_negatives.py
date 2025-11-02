import multiprocessing
multiprocessing.set_start_method('spawn', force=True) # Set multiprocessing start method to 'spawn'

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"    # Limit CUDA asynchronous calls
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

import json
import argparse
from tqdm import tqdm

import torch
from datasets import Dataset
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rank_bm25 import BM25Okapi
from utils import get_first_k_tokens
from vllm import LLM, SamplingParams

set_seed(42)

# Argument Parser ??
parser = argparse.ArgumentParser(description="Generate Pseudo Negatives")
parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.3')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--k', type=int, default=0)
parser.add_argument('--max_step', type=int, default=5000)
parser.add_argument('--cut_off', type=int, default=2048)
parser.add_argument('--max_epoch', type=int, default=2)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--task_name', type=str, default='news_headline')
parser.add_argument('--add_profile', action='store_true')
parser.add_argument('--access_token', type=str, default=None)
parser.add_argument('--data_path', type=str, default='/home/sensiblescent428/OPPU/data')
parser.add_argument('--r', type=int, default=8)
parser.add_argument('--alpha', type=int, default=8)
parser.add_argument('--repetition_penalty', type=float, default=1.0)
parser.add_argument('--response_num', type=int, default=3)
args = parser.parse_args()

# Mistral Chat Template
MISTRAL_CHAT_TEMPLATE = """
{% if messages[0]['role'] == 'system' %}
{% set loop_messages = messages[1:] %}
{% set system_message = messages[0]['content'].strip() + ' ' %}
{% else %}
{% set loop_messages = messages %}
{% set system_message = '' %}
{% endif %}
{% for message in loop_messages %}
    {% if loop.index0 == 0 %}
        {% set content = system_message + message['content'] %}
    {% else %}
        {% set content = message['content'] %}
    {% endif %}
    {% if message['role'] == 'user' %}
        {{ '[INST] ' + content.strip() + ' [/INST]' }}
    {% elif message['role'] == 'assistant' %}
        {{ ' ' + content.strip() + ' ' + eos_token }}
    {% endif %}
{% endfor %}
"""

def construct_prompt(qdict: dict, prompt_template: dict, user_history: list, args, user_profile: str = None) -> str:
    base_prompt = prompt_template[args.task_name]['OPPU_input'].format(**qdict)
    if args.k > 0 and user_history:
        processed_history = [{k: get_first_k_tokens(v, 768) if isinstance(v, str) else v for k, v in p.items()} for p in user_history]
        history_list = [prompt_template[args.task_name]['retrieval_history'].format(**p) for p in processed_history]
        tokenized_corpus = [doc.split(" ") for doc in history_list]
        bm25 = BM25Okapi(tokenized_corpus)
        query_str = prompt_template[args.task_name]["retrieval_query"].format(**qdict)
        tokenized_query = query_str.split(" ")
        retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=args.k)
        base_prompt = "\n".join(retrieved_history) + "\n" + base_prompt
    if args.add_profile and user_profile:
        base_prompt = user_profile + "\n" + base_prompt
    return base_prompt

def apply_chat_template(example, tokenizer, model_name):
    if 'mistral' in model_name.lower():
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
    else:
        raise ValueError("Invalid model specified.")
    prompt_messages = example["prompt"]
    chosen_messages = example["chosen"]
    example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
    return example

def generate_and_tokenize_prompt(data_point, tokenizer):
    tokenized_prompt = tokenizer(
        data_point["prompt"], truncation=True, max_length=args.cut_off, return_tensors=None, add_special_tokens=False
    )
    tokenized_chosen = tokenizer(
        data_point["chosen"], truncation=True, max_length=args.cut_off, return_tensors=None, add_special_tokens=False
    )
    input_ids = tokenized_prompt["input_ids"] + tokenized_chosen["input_ids"]
    attention_mask = tokenized_prompt["attention_mask"] + tokenized_chosen["attention_mask"]
    labels = [-100] * len(tokenized_prompt["input_ids"]) + tokenized_chosen["input_ids"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def create_dataset(user_history_data, user_profile, tokenizer, prompt_template, args, system_prompt):
    data = []
    for idx, qdict in enumerate(user_history_data):
        qdict = {k: get_first_k_tokens(v, 768) if isinstance(v, str) else v for k, v in qdict.items()}
        history = user_history_data[:idx] if idx > 0 else []
        prompt_str = construct_prompt(qdict, prompt_template, history, args, user_profile)
        if args.task_name in ("news_headline", "scholarly_title"):
            chosen = qdict['title']
        elif args.task_name == "abstract_generation":
            chosen = qdict['abstract']
        elif args.task_name == "review_writing":
            chosen = qdict['reviewText']
        elif args.task_name == "topic_writing":
            chosen = qdict['content']
        example = {
            "prompt": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_str}],
            "chosen": [{"role": "assistant", "content": chosen}],
        }
        formatted_example = apply_chat_template(example, tokenizer, args.model_name)
        data.append({"prompt": formatted_example["text_prompt"], "chosen": formatted_example["text_chosen"]})
    return Dataset.from_list(data)

def generate_pseudo(args):
    data_path = os.path.join(args.data_path, args.task_name, "user_top_100_history.json")
    with open(data_path, 'r') as f:
        test_data = json.load(f)
    profile_path = os.path.join(args.data_path, args.task_name, "profile_user_100.json") if args.add_profile else None
    test_profile = json.load(open(profile_path, 'r')) if profile_path else None
    with open('./prompt/prompt.json', 'r') as f:
        prompt_template = json.load(f)

    model_name_short = args.model_name.split('/')[-1]
    tam_lora_adapter_path = f"./ckpt/TAM/{args.task_name}/TAM-{model_name_short}_ckpt"
    merged_model_path = f"./ckpt/merged_model/{args.task_name}"
    if not os.path.exists(merged_model_path):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, device_map="auto", torch_dtype=torch.bfloat16, token=args.access_token
        )
        base_model = PeftModel.from_pretrained(model=model, model_id=tam_lora_adapter_path, is_trainable=False)
        tokenizer = AutoTokenizer.from_pretrained(tam_lora_adapter_path, token=args.access_token)
        model = base_model.merge_and_unload()
        os.makedirs(merged_model_path, exist_ok=True)
        model.save_pretrained(merged_model_path)
        tokenizer.save_pretrained(merged_model_path)
    
    # Load TAM model for vLLM inference
    llm = LLM(
        model=merged_model_path, 
        dtype="bfloat16",
        tokenizer_mode="auto", 
        trust_remote_code=True,
        tensor_parallel_size=2,             # Set tensor parallel size
        pipeline_parallel_size=1,           # Set pipeline parallel size
        max_num_batched_tokens=4096,        # Set max batched tokens
        distributed_executor_backend='ray'  # Set distributed executor backend
    )
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path, token=args.access_token)

    pred_all = []

    for user_idx in tqdm(range(len(test_data)), desc="Inference"):
        user_profile = test_profile[user_idx]['output'] if args.add_profile and test_profile else None
        user_data = test_data[user_idx]
        user_history_data = user_data["profile"]
        SYSTEM_PROMPT = prompt_template[args.task_name]["system"]

        test_question_list, question_id_list = [], []
        if args.k > 0:
            visible_history_list = [prompt_template[args.task_name]['retrieval_history'].format(**{k: get_first_k_tokens(v, 368) if isinstance(v, str) else v for k, v in p.items()}) for p in user_history_data]
            tokenized_corpus = [doc.split(" ") for doc in visible_history_list]
            bm25 = BM25Okapi(tokenized_corpus)

        for q in user_history_data:
            if args.task_name == "news_headline":
                test_article = q["text"]
            elif args.task_name == "scholarly_title":
                test_article = q['abstract']
            elif args.task_name == "abstract_generation":
                test_article = q['title']
            elif args.task_name == "topic_writing":
                test_article = q['summary']           
            elif args.task_name == "review_writing":
                overall = q['overall']
                description = q['description']
                summary = q['summary']
                test_article = (overall, description, summary)

            if args.task_name == 'review_writing':
                test_prompt = prompt_template[args.task_name]['prompt'].format(*test_article)
            else:
                test_prompt = prompt_template[args.task_name]['prompt'].format(test_article)

            if args.k > 0:
                tokenized_query = prompt_template[args.task_name]['retrieval_query_wokey'].format(test_article).split(" ")
                retrieved_history = bm25.get_top_n(tokenized_query, visible_history_list, n=args.k)
                test_prompt = "".join(retrieved_history) + "\n" + test_prompt
            if args.add_profile and user_profile: 
                test_prompt = user_profile + "\n" + test_prompt
            chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": test_prompt}]
            formatted_prompt = tokenizer.apply_chat_template(chat_messages, tokenize=False)

            test_question_list.append(formatted_prompt)
            question_id_list.append(q['id'])

        max_length = 600 if args.task_name in ("review_writing", "abstract_generation", "topic_writing") else 100
        repetition_penalty = args.repetition_penalty if args.task_name in ("review_writing", "abstract_generation", "topic_writing") else 1

        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=max_length,
            repetition_penalty=repetition_penalty,
            top_p=0.95,
            n=args.response_num
        )

        batch_size = 16
        for i in range(0, len(test_question_list), batch_size):
            batch = test_question_list[i:i+batch_size]
            outputs = llm.generate(batch, sampling_params)
            for j, output in enumerate(outputs):
                responses = [o.text.strip() for o in output.outputs]
                pred_all.append({
                    "id": question_id_list[i + j], 
                    "output": responses             
                })
                print(f"ID: {question_id_list[i + j]}, Outputs: {responses}")

    from copy import deepcopy
    # Build processed_users with pseudo_negatives
    processed_users = []
    
    # Build a mapping from id to output
    id2output = {item["id"]: item["output"] for item in pred_all}
    
    for user_data in test_data:
        user_copy = deepcopy(user_data)
        for profile_item in user_copy["profile"]:
            item_id = profile_item["id"]
            if item_id in id2output:
                profile_item["pseudo_negative"] = id2output[item_id]
        processed_users.append(user_copy)
    
    # Save to file
    output_path = f"./data/{args.task_name}/user_top_100_history_with_pseudo_negatives.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(processed_users, f, indent=4)
    
    print(f"Results saved at {output_path}!!!")

def main():
    args = parser.parse_args()
    generate_pseudo(args)

if __name__ == "__main__":
    main()