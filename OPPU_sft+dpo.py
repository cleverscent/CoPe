import torch
import torch.nn.functional as F
import argparse
import json
import os
from functools import partial
from tqdm import tqdm
from datasets import Dataset
from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import DPOTrainer, DPOConfig
from rank_bm25 import BM25Okapi
from utils import (
    split_batch,
    get_first_k_tokens,
    print_trainable_parameters,
    name2taskid,
    extract_news_headline,
    extract_scholarly_title,
    extract_topic_writing,
    extract_review_writing,
    extract_abstract_generation,
    get_output
)

set_seed(42)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["WANDB_DISABLED"] = "true"

parser = argparse.ArgumentParser(description="OPPU_SFT + DPO")
parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.3', help='choose from google/gemma-3-1b-it, meta-llama/Llama-3.1-8B-Instruct, google/gemma-3-12b-it') 
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--k', type=int, default=0)
parser.add_argument('--max_step', type=int, default=5000)
parser.add_argument('--cut_off', type=int, default=2048)
parser.add_argument('--max_epoch', type=int, default=1)
parser.add_argument('--temperature', type=float, default=0.3)
parser.add_argument('--sft_learning_rate', type=float, default=1e-4)
parser.add_argument('--dpo_learning_rate', type=float, default=5e-6)
parser.add_argument('--dpo_beta', type=float, default=0.01)
parser.add_argument('--task_name', type=str, default='topic_writing', help="Choose from 'news_headline', 'scholarly_title', 'abstract_generation', 'review_writing', 'topic_writing'")
parser.add_argument('--add_profile', action='store_true')
parser.add_argument('--access_token', type=str, default=None)
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--negative_sampling_method', type=str, default='pseudo', help="Choose from 'random', 'sbert', 'pseudo'")
parser.add_argument('--mode', type=str, default='max', help="Choose from 'max', 'min'")
parser.add_argument('--r', type=int, default=8)
parser.add_argument('--alpha', type=int, default=8)
parser.add_argument('--contrastive_alpha', type=float, default=0.7)
parser.add_argument('--repetition_penalty', type=float, default=1.2)
parser.add_argument('--is_cd', action='store_true', help="Apply contrastive decoding during inference")
parser.add_argument('--is_train', action='store_true', help="Run in training mode (default: inference mode)")
parser.add_argument('--amateur_is_initial_model', action='store_true')
parser.add_argument('--amateur_is_OPPU', action='store_true')
args = parser.parse_args()

with open("chat_templates.json", "r") as f:
    CHAT_TEMPLATES = json.load(f)

def construct_prompt(qdict: dict, prompt_template: dict, user_history: list, args, user_profile: str = None) -> str:
    base_prompt = prompt_template[args.task_name]['OPPU_input'].format(**qdict)
    if args.k > 0 and user_history:
        processed_history = []
        for p in user_history:
            p_copy = p.copy()
            for key, val in p_copy.items():
                if isinstance(val, str):
                    p_copy[key] = get_first_k_tokens(val, 768)
            processed_history.append(p_copy)
        history_list = [prompt_template[args.task_name]['retrieval_history'].format(**p) for p in processed_history]
        tokenized_corpus = [doc.split(" ") for doc in history_list]
        bm25 = BM25Okapi(tokenized_corpus)
        query_str = prompt_template[args.task_name]["retrieval_query"].format(**qdict)
        tokenized_query = query_str.split(" ")
        retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=args.k)
        history_string = "\n".join(retrieved_history)
        base_prompt = history_string + "\n" + base_prompt
    if args.add_profile and user_profile:
        base_prompt = user_profile + "\n" + base_prompt
    return base_prompt

def get_negative_sample(all_data, user_id, qdict, args):
    mode = args.mode
    if args.negative_sampling_method == 'pseudo':
        negative_response = qdict["negative_response"]
        return negative_response

def apply_chat_template(example, tokenizer, model_name):
    model_name = model_name.lower()
    if 'mistral' in model_name:
        tokenizer.chat_template = CHAT_TEMPLATES["mistral"]
    elif 'llama' in model_name:
        tokenizer.chat_template = CHAT_TEMPLATES["llama"]
    elif "gemma" in model_name:
        tokenizer.chat_template = CHAT_TEMPLATES["gemma"]
    elif "qwen" in model_name:
        tokenizer.chat_template = CHAT_TEMPLATES["qwen2.5"]
    else:
        raise ValueError("Invalid model specified.")
    prompt_messages = example["prompt"]
    chosen_messages = example["chosen"]
    rejected_messages = example.get("rejected", [])
    example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
    if rejected_messages:
        example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
    return example

def generate_and_tokenize_prompt(data_point, tokenizer):
    tokenized_prompt = tokenizer(data_point["prompt"], truncation=True, max_length=args.cut_off, return_tensors=None, add_special_tokens=False)
    def process_response(response, is_prompt=False):
        tokenized = tokenizer(response, truncation=True, max_length=args.cut_off, return_tensors=None, add_special_tokens=not is_prompt)
        if is_prompt:
            tokenized["labels"] = [-100] * len(tokenized["input_ids"])
        return tokenized
    chosen = process_response(data_point["chosen"])
    rejected = process_response(data_point["rejected"])
    return {
        "chosen_input_ids": tokenized_prompt["input_ids"] + chosen["input_ids"],
        "chosen_attention_mask": tokenized_prompt["attention_mask"] + chosen["attention_mask"],
        "rejected_input_ids": tokenized_prompt["input_ids"] + rejected["input_ids"],
        "rejected_attention_mask": tokenized_prompt["attention_mask"] + rejected["attention_mask"],
    }


def contrastive_decoding(
    prompt, expert_model, amateur_model, expert_tokenizer,
    plausability_alpha=0.1, contrastive_alpha=0.7, repetition_penalty=1.0, max_length=200, 
):

    device = expert_model.device

    initial_input_ids = expert_tokenizer.encode(prompt, return_tensors="pt").to(device)
    all_input_ids = initial_input_ids.clone()

    past_key_values_expert = None
    past_key_values_amateur = None

    next_token_input_expert = initial_input_ids
    next_token_input_amateur = initial_input_ids

    logits_processor = LogitsProcessorList([
        RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
    ]) if repetition_penalty != 1.0 else LogitsProcessorList()

    generated_tokens = []

    for _ in range(max_length):
        with torch.no_grad():
            expert_outputs = expert_model(
                input_ids=next_token_input_expert,
                past_key_values=past_key_values_expert,
                use_cache=True,
                return_dict=True
            )
            expert_logits = expert_outputs.logits[:, -1, :]
            past_key_values_expert = expert_outputs.past_key_values

            amateur_outputs = amateur_model(
                input_ids=next_token_input_amateur,
                past_key_values=past_key_values_amateur,
                use_cache=True,
                return_dict=True
            )
            amateur_logits = amateur_outputs.logits[:, -1, :]
            past_key_values_amateur = amateur_outputs.past_key_values

            expert_probs = F.softmax(expert_logits, dim=-1)
            amateur_probs = F.softmax(amateur_logits, dim=-1)

            max_prob = torch.max(expert_probs, dim=-1).values

            vhead_mask = expert_probs >= (plausability_alpha * max_prob)
            truncated_expert_probs = expert_probs * vhead_mask
            truncated_expert_probs = truncated_expert_probs / (truncated_expert_probs.sum(dim=-1, keepdim=True) + 1e-8)

            contrastive_logits = torch.log(truncated_expert_probs + 1e-8) - contrastive_alpha * torch.log(amateur_probs + 1e-8)

            processed_logits = logits_processor(all_input_ids, contrastive_logits)

            contrastive_probs = F.softmax(processed_logits, dim=-1)
            next_token = torch.argmax(contrastive_probs, dim=-1) 

            next_token_id = next_token.item()
            generated_tokens.append(next_token_id)

            all_input_ids = torch.cat([all_input_ids, next_token.unsqueeze(-1)], dim=-1)

            next_token_input_expert = next_token.unsqueeze(-1)
            next_token_input_amateur = next_token.unsqueeze(-1)

            if next_token_id == expert_tokenizer.eos_token_id:
                break

    generated_text = expert_tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text

def create_pair_dataset(user_history_data: list, user_profile: str, tokenizer, prompt_template: dict, args, system_prompt: str, all_data: list, user_id: str):
    data = []
    for idx, qdict in enumerate(user_history_data):
        for key, val in qdict.items():
            if isinstance(val, str):
                qdict[key] = get_first_k_tokens(val, 768)
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
        rejected = get_negative_sample(all_data, user_id, qdict, args)
        example = {
            "prompt": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_str}],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}]
        }
        formatted_example = apply_chat_template(example, tokenizer, args.model_name)
        data.append({
            "prompt": formatted_example["text_prompt"],
            "chosen": formatted_example["text_chosen"],
            "rejected": formatted_example["text_rejected"]
        })
    return data

def load_model_and_tokenizer(model_name, adapter_path=None, access_token=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16, token=access_token
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = False
    # model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    if adapter_path:
        model = PeftModel.from_pretrained(model=model, model_id=adapter_path, is_trainable=False)
        model = model.merge_and_unload()
        print("[LoRA adapter merge and unloaded]")
    return model, tokenizer

def train(args):
    if args.negative_sampling_method == 'pseudo':
        if args.mode == 'max':
            data_path = os.path.join(args.data_path, args.task_name, "user_top_100_history_with_pseudo_negatives_max.json")
        elif args.mode == 'min':
            data_path = os.path.join(args.data_path, args.task_name, "user_top_100_history_with_pseudo_negatives_min.json")
    else:
        data_path = os.path.join(args.data_path, args.task_name, "user_top_100_history_with_negatives.json")
    with open(data_path, 'r') as f:
        test_data = json.load(f)
    profile_path = os.path.join(args.data_path, args.task_name, "profile_user_100.json") if args.add_profile else None
    test_profile = json.load(open(profile_path, 'r')) if profile_path else None
    with open('./prompt/prompt.json', 'r') as f:
        prompt_template = json.load(f)

    model_name_short = args.model_name.split('/')[-1]
    tam_lora_adapter_path = f"./ckpt/{args.task_name}/TAM-{model_name_short}-k{args.k}_ckpt"

    base_model, tokenizer = load_model_and_tokenizer(model_name=args.model_name, adapter_path=tam_lora_adapter_path)
    print_trainable_parameters(base_model)

    for user_idx in tqdm(range(len(test_data)), desc="Training"):
        print("" + "=" * 50)
        print(f"{user_idx}-th LLM Personalization Begins")
        file_suffix = f"./OPPU_SFT+DPO/{args.task_name}/user_{user_idx}"
        save_path = file_suffix

        user_lora_adapter_path = f'./ckpt/OPPU_SFT/{args.task_name}/user_{user_idx}'   
        user_model = PeftModel.from_pretrained(model=base_model, model_id=user_lora_adapter_path, is_trainable=True)
        user_model.train()
        print_trainable_parameters(user_model)

        user_profile = test_profile[user_idx]['output'] if args.add_profile and test_profile else None
        user_data = test_data[user_idx]
        user_history_data = user_data["profile"]
        user_id = user_data["user_name"] if args.task_name in ("review_writing", "abstract_generation", "topic_writing") else user_data["user_id"]
        SYSTEM_PROMPT = prompt_template[args.task_name]["system"]

        dpo_train_data = create_pair_dataset(user_history_data, user_profile, tokenizer, prompt_template, args, 
                                              SYSTEM_PROMPT, test_data, user_id)
        user_train_dataset = Dataset.from_list(dpo_train_data)

        tokenize_fn = partial(generate_and_tokenize_prompt, tokenizer=tokenizer)
        user_train_dataset = user_train_dataset.map(tokenize_fn).shuffle()

        training_args = DPOConfig(
            output_dir="./outputs",
            beta=args.dpo_beta,  
            max_length=args.cut_off,  
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            optim='adamw_torch',
            num_train_epochs=args.max_epoch,
            save_steps=1e10,  
            logging_steps=10,  
            learning_rate=args.dpo_learning_rate,
            weight_decay=1e-2,
            bf16=True,  
            max_grad_norm=0.3,
            warmup_ratio=0.1,  
            lr_scheduler_type='linear',  
            report_to=[], 
        )

        trainer = DPOTrainer(
            model=user_model,  
            args=training_args,  
            train_dataset=user_train_dataset,  
            tokenizer=tokenizer, 
        )

        for name, module in user_model.named_modules():
            if "norm" in name:
                module.to(user_model.dtype)

        trainer.train()
        user_model.save_pretrained(save_path)

def inference(args):
    data_path = os.path.join(args.data_path, args.task_name, "user_top_100_history.json")
    with open(data_path, 'r') as f:
        test_data = json.load(f)
    profile_path = os.path.join(args.data_path, args.task_name, "profile_user_100.json") if args.add_profile else None
    test_profile = json.load(open(profile_path, 'r')) if profile_path else None
    with open('./prompt/prompt.json', 'r') as f:
        prompt_template = json.load(f)

    model_name_short = args.model_name.split('/')[-1]
    tam_lora_adapter_path = f"./ckpt/TAM/{args.task_name}/TAM-{model_name_short}_ckpt"

    base_model, tokenizer = load_model_and_tokenizer(model_name=args.model_name, adapter_path=tam_lora_adapter_path)
    print_trainable_parameters(base_model)

    if args.is_cd:
        if args.amateur_is_initial_model:
            amateur_model = AutoModelForCausalLM.from_pretrained(
                args.model_name, device_map="auto", torch_dtype=torch.bfloat16, token=args.access_token
            )
        elif args.amateur_is_OPPU:
            pass
        else:
            amateur_model, _ = load_model_and_tokenizer(model_name=args.model_name, adapter_path=tam_lora_adapter_path)
        amateur_model.eval()

    pred_all = []

    if args.task_name == "news_headline":
        extract_article = extract_news_headline
    elif args.task_name == "scholarly_title":
        extract_article = extract_scholarly_title
    elif args.task_name == "abstract_generation":
        extract_article = extract_abstract_generation
    elif args.task_name == "review_writing":
        extract_article = extract_review_writing
    elif args.task_name == "topic_writing":
        extract_article = extract_topic_writing

    for user_idx in tqdm(range(len(test_data)), desc="Inference"):
        file_suffix = f"./ckpt/OPPU_SFT+DPO/{args.task_name}/user_{user_idx}"
        adapter_path = file_suffix
        
        user_model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
        expert_model = user_model
        expert_model.eval()

        if args.amateur_is_OPPU:
            oppu_lora_adapter_path = f"./ckpt/OPPU_SFT/{args.task_name}/user_{user_idx}"
            amateur_model, _ = load_model_and_tokenizer(model_name=args.model_name, adapter_path=oppu_lora_adapter_path)
            amateur_model.eval()

        user_profile = test_profile[user_idx]['output'] if args.add_profile and test_profile else None
        user_data = test_data[user_idx]
        user_history_data = user_data["profile"]
        SYSTEM_PROMPT = prompt_template[args.task_name]["system"]

        test_question_list, question_id_list = [], []
        if args.k > 0:
            visible_history_list = [prompt_template[args.task_name]['retrieval_history'].format(**{k: get_first_k_tokens(v, 368) if isinstance(v, str) else v for k, v in p.items()}) for p in user_history_data]
            tokenized_corpus = [doc.split(" ") for doc in visible_history_list]
            bm25 = BM25Okapi(tokenized_corpus)

        for q in user_data['query']:
            test_question = q['input']
            test_article = extract_article(test_question)

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

        out_list = []
        max_length = 600 if args.task_name in ("review_writing", "abstract_generation", "topic_writing") else 100
        repetition_penalty = args.repetition_penalty if args.task_name in ("review_writing", "abstract_generation", "topic_writing") else 1.0
        if args.is_cd:
            for prompt_text in test_question_list:
                generated_text = contrastive_decoding(
                    prompt_text, expert_model, amateur_model, tokenizer, 
                    contrastive_alpha=args.contrastive_alpha, repetition_penalty=repetition_penalty, max_length=max_length
                )
                out_list.append(generated_text.strip())
        else:
            test_batch_size = 16
            test_batch_list = split_batch(test_question_list, test_batch_size)
            with torch.no_grad():
                for batch in test_batch_list:
                    inputs = tokenizer(batch, return_tensors="pt", padding=True, return_token_type_ids=False).to(expert_model.device)
                    outputs = expert_model.generate(
                        **inputs, do_sample=False, top_k=50, temperature=args.temperature,
                        top_p=0.95, pad_token_id=tokenizer.eos_token_id, max_new_tokens=max_length, repetition_penalty=repetition_penalty
                    )
                    out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    out_list.extend([get_output(sentence, args) for sentence in out_sentence])

        for i, output in enumerate(out_list):
            pred_all.append({"id": question_id_list[i], "output": output})
            print(f"ID:{question_id_list[i]}, Output: {output}")
            print("")

    model_name_short = args.model_name.split('/')[-1]
    file_suffix = f"beta{args.dpo_beta}"
    file_suffix += f"-max_epoch{args.max_epoch}" if args.max_epoch == 1 else ""
    file_suffix += f"-ca{args.contrastive_alpha}-CD" if args.is_cd else ""
    file_suffix += "-profile" if args.add_profile else ""
    file_suffix += "-amateur_is_initial_model" if args.amateur_is_initial_model else ""
    file_suffix += "-amateur_is_OPPU" if args.amateur_is_OPPU else ""
    file_path = f"./output/{args.task_name}/OPPU-SFT+DPO-{model_name_short}-k{args.k}-{args.mode}-sft_lr{args.sft_learning_rate}-dpo_lr{args.dpo_learning_rate}-rp{repetition_penalty}-{file_suffix}.json"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump({'task': name2taskid[args.task_name], 'golds': pred_all, 'model': args.model_name}, f, indent=2)
    print("Inference completed!")

def main():
    if args.is_train:
        train(args)
    else:
        inference(args)

if __name__=="__main__":
    main()


