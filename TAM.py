import torch
import argparse
import json
import os
import logging
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from rank_bm25 import BM25Okapi
from utils import (
    split_batch,
    get_first_k_tokens,
    print_trainable_parameters,
    name2taskid,
    extract_news_headline,
    extract_scholarly_title,
    extract_abstract_generation,
    extract_review_writing,
    extract_topic_writing,
    get_output
)
set_seed(42)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

parser = argparse.ArgumentParser(description="Parser for TAM")
parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.3', help='choose from mistralai/Mistral-7B-Instruct-v0.3, google/gemma-3-4b-it, meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen2.5-1.5B-Instruct') 
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--k', type=int, default=0)
parser.add_argument('--max_step', type=int, default=5000)
parser.add_argument('--cut_off', type=int, default=2048)
parser.add_argument('--max_seq_length', type=int, default=2048)
parser.add_argument('--max_epoch', type=int, default=3)
parser.add_argument('--temperature', type=float, default=0.3)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--task_name', type=str, default='abstract_generation')
parser.add_argument('--add_profile', action='store_true')
parser.add_argument('--access_token', type=str, default=None)
parser.add_argument('--r', type=int, default=16)
parser.add_argument('--alpha', type=int, default=16)
parser.add_argument('--repetition_penalty', type=float, default=1.3)
parser.add_argument('--is_initial_model', action='store_true')
parser.add_argument('--is_train', action='store_true')
parser.add_argument('--is_test', action='store_true')

args = parser.parse_args()
task_name = args.task_name
batch_size = args.batch_size
k = args.k
cutoff_len = args.cut_off
max_epoch = args.max_epoch
model_name_short = args.model_name.split('/')[-1]

with open("chat_templates.json", "r") as f:
    CHAT_TEMPLATES = json.load(f)
with open(f"./data/{args.task_name}/user_others.json", 'r') as f:
    train = json.load(f)
with open(f"./data/{args.task_name}/user_top_100_history.json", 'r') as f:
    test_data = json.load(f)
with open('./prompt/prompt.json', 'r') as f:
    prompt_template = json.load(f)
if args.add_profile:
    with open(f'./data/{args.task_name}/profile_user_100.json', 'r') as f:
        test_profile = json.load(f)

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

print('#' * 70)
logging.basicConfig(level=logging.INFO)
logging.info(f"Loaded model name: {args.model_name}")

SYSTEM_PROMPT = prompt_template[args.task_name]["system"]

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
    
    example["text_prompt"] = tokenizer.apply_chat_template(
        example["prompt"], 
        tokenize=False, 
        add_generation_prompt=True
    )
    example["text_chosen"] = tokenizer.apply_chat_template(
        example["chosen"], 
        tokenize=False
    )
    return example

def create_sft_dataset(train, tokenizer, model_name):
    sft_data = []
    for i, user in enumerate(train):
        for q in user['query']:
            prompt = q['input']
            if args.k > 0:
                if 'profile' in user:
                    visible_history_list = user['profile']
                    for p in visible_history_list:
                        for key, value in p.items():
                            p[key] = get_first_k_tokens(p[key], 368)
                    history_list = [
                        prompt_template[args.task_name]['retrieval_history'].format(**p)
                        for p in visible_history_list
                    ]
                    tokenized_corpus = [doc.split(" ") for doc in history_list]
                    bm25 = BM25Okapi(tokenized_corpus)
                    tokenized_query = prompt_template[args.task_name]['retrieval_query_wokey'].format(prompt).split(" ")
                    retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=args.k)
                    history_string = "".join(retrieved_history)
                    prompt = history_string + "\n" + prompt
            chosen = q['gold']
            example = {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "chosen": [{"role": "assistant", "content": chosen}]
            }
            formatted_example = apply_chat_template(example, tokenizer, model_name)
            sft_data.append({
                "prompt": formatted_example["text_prompt"],
                "chosen": formatted_example["text_chosen"]
            })
    return sft_data

def generate_and_tokenize_prompt(data_point):
    tokenized_prompt = tokenizer(
        data_point["prompt"],
        truncation=True,
        max_length=cutoff_len,
        return_tensors=None,
        add_special_tokens=False
    )
    tokenized_chosen = tokenizer(
        data_point["chosen"],
        truncation=True,
        max_length=cutoff_len,
        return_tensors=None,
        add_special_tokens=False
    )
    input_ids = tokenized_prompt["input_ids"] + tokenized_chosen["input_ids"]
    attention_mask = tokenized_prompt["attention_mask"] + tokenized_chosen["attention_mask"]
    labels = [-100] * len(tokenized_prompt["input_ids"]) + tokenized_chosen["input_ids"]

    return {
        "input_ids": input_ids,       
        "attention_mask": attention_mask,
        "labels": labels
    }

########## Training (SFT) ##########
if args.is_train:

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=args.access_token,
    )

    if "qwen" in args.model_name.lower():
        model.config.attn_implementation = "eager"  # or "flash_attention_2"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side = "left", token=args.access_token)

    # print("EOS token:", tokenizer.eos_token)          # EOS token: </s>
    # print("EOS token ID:", tokenizer.eos_token_id)    # EOS token ID: 2

    # print("PAD token:", tokenizer.pad_token)          # PAD token: None
    # print("PAD token ID:", tokenizer.pad_token_id)    # PAD token ID: None
    
    # print("BOS token:", tokenizer.bos_token)          # BOS token: <s>
    # print("BOS token ID:", tokenizer.bos_token_id)    # BOS token ID: 1

    tokenizer.pad_token = tokenizer.eos_token           # Set PAD token to EOS token
    tokenizer.pad_token_id = tokenizer.eos_token_id     # Set PAD token ID to EOS token ID

    model.config.use_cache = False          # Disable cache
    model.gradient_checkpointing_enable()   # Enable gradient checkpointing
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    train_data = create_sft_dataset(train, tokenizer, args.model_name)
    train_dataset = Dataset.from_list(train_data)

    train_dataset = train_dataset.shuffle()

    print(train_dataset[:3])
    train_dataset = train_dataset.map(generate_and_tokenize_prompt, remove_columns=["prompt", "chosen"])
    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    training_args = TrainingArguments(
        output_dir='outputs/',
        # max_prompt_length=512,
        # max_length = args.cut_off,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        optim='adamw_torch',
        num_train_epochs=max_epoch,
        save_steps=1e9,
        logging_steps=50,
        learning_rate=args.learning_rate,
        weight_decay=1e-2,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        group_by_length=False,
        lr_scheduler_type='linear',
        report_to=[],
        # run_name=f"{task_name}_sft",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    # Convert LayerNorm to model dtype
    for name, module in model.named_modules():
        if "norm" in name:
            module.to(model.dtype)

    trainer.train()

    adapter_path = f"./ckpt/TAM/{args.task_name}/TAM-{model_name_short}_ckpt"

    os.makedirs(os.path.dirname(adapter_path), exist_ok=True)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

########## Test ##########
if args.is_test:
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=args.access_token,
    )
    base_model.config.use_cache = False

    if args.is_initial_model:
        model = base_model
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side = "left", token=args.access_token)
        tokenizer.pad_token = tokenizer.eos_token           # Set PAD token to EOS token
        tokenizer.pad_token_id = tokenizer.eos_token_id     # Set PAD token ID to EOS token ID
        # print("EOS token:", tokenizer.eos_token)          # EOS token: </s>
        # print("EOS token ID:", tokenizer.eos_token_id)    # EOS token ID: 2

        # print("PAD token:", tokenizer.pad_token)          # PAD token: None
        # print("PAD token ID:", tokenizer.pad_token_id)    # PAD token ID: None
        
        # print("BOS token:", tokenizer.bos_token)          # BOS token: <s>
        # print("BOS token ID:", tokenizer.bos_token_id)    # BOS token ID: 1
    else:
        adapter_path = f"./ckpt/TAM/{args.task_name}/TAM-{model_name_short}_ckpt"
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, padding_side = "left", token=args.access_token)
        model = PeftModel.from_pretrained(model=base_model, model_id=adapter_path, is_trainable=False)
        model = model.merge_and_unload()

    pred_all = []

    for i in tqdm(range(len(test_data)), desc="Testing Users"):
        expert_model = model
        expert_model.eval()

        if args.add_profile:
            profile = test_profile[i]['output']
        else:
            profile = None
        if k > 0:
            visible_history_list = test_data[i]['profile']
            for p in visible_history_list:
                for key, value in p.items():
                    # p[key] = get_first_k_tokens(p[key], 368)
                    p[key] = get_first_k_tokens(str(p[key]), 368)
            history_list = [prompt_template[args.task_name]['retrieval_history'].format(**p) for p in visible_history_list]
            tokenized_corpus = [doc.split(" ") for doc in history_list]
            bm25 = BM25Okapi(tokenized_corpus)

        test_question_list = []
        question_id_list = []

        for q in test_data[i]['query']:
            if args.task_name == "review_writing":
                test_question = q['input']
                test_article = extract_article(test_question)
                test_prompt = prompt_template[args.task_name]['prompt'].format(*test_article)
            else:
                test_question = q['input']
                test_article = extract_article(test_question)
                test_prompt = prompt_template[args.task_name]['prompt'].format(test_article)

            if k > 0:
                if args.task_name == 'review_writing':
                    tokenized_query = prompt_template[args.task_name]['retrieval_query_wokey'].format(*test_article).split(" ")
                else:
                    tokenized_query = prompt_template[args.task_name]['retrieval_query_wokey'].format(test_question).split(" ")
                retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=args.k)
                history_string = "".join(retrieved_history)
                test_prompt = history_string + "\n" + test_prompt
            if args.add_profile:
                test_prompt = profile + "\n" + test_prompt

            chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": test_prompt}]
            formatted_prompt = tokenizer.apply_chat_template(chat_messages, tokenize=False)
            test_question_list.append(formatted_prompt)
            question_id_list.append(q['id'])

        # print(f"Test question list: {test_question_list}")

        out_list = []

        max_length = 600 if args.task_name in ("review_writing", "abstract_generation", "topic_writing") else 100
        repetition_penalty = args.repetition_penalty if args.task_name in ("review_writing", "abstract_generation", "topic_writing") else 1

        test_batch_size = 16
        test_batch_list = split_batch(test_question_list, test_batch_size)
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(test_batch_list), total=len(test_batch_list)):
                inputs = tokenizer(batch, return_tensors="pt", padding=True, padding_side="left", return_token_type_ids=False)
                inputs = inputs.to(expert_model.device)
                with torch.autocast(device_type="cuda"):
                    outputs = expert_model.generate(
                        **inputs,
                        do_sample=False,
                        top_k=50,
                        temperature=args.temperature,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                        max_new_tokens=max_length,
                        repetition_penalty=repetition_penalty,
                    )
                out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                # print("out_sentence:", out_sentence)
                out_list.extend([get_output(sentence, args) for sentence in out_sentence])
                # processed_sentences = [sentence for sentence in out_sentence]

        for i, output in enumerate(out_list):
            pred_all.append({"id": question_id_list[i], "output": output})
            print(f"ID: {question_id_list[i]}, Output: {output}")

    output_file = {
        'task': name2taskid[args.task_name],
        'golds': pred_all,
        'model': args.model_name,
    }

    file_suffix = "-profile" if args.add_profile else ""
    file_suffix += "-initial" if args.is_initial_model else ""
    file_suffix += "-RAG" if args.k > 0 else ""
    file_suffix += "-PAG" if args.add_profile else ""
    file_path = f"./output/{args.task_name}/TAM-{model_name_short}-k{args.k}-rp{repetition_penalty}{file_suffix}.json"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(output_file, f, indent=2)

    print("Results saved!!!")