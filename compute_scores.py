import os
import json
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

set_seed(42)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
import copy

parser = argparse.ArgumentParser(description="Compare fOPPU and fBase")
parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.3')
parser.add_argument('--task_name', type=str, default='news_headline')
parser.add_argument('--std', type=str, default='max') 
parser.add_argument('--st', type=int, default=0)
parser.add_argument('--end', type=int, default=100)

args = parser.parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Load the JSON file that contains all the responses
test_data_path = f"./data/{args.task_name}/user_top_100_history_with_pseudo_negatives.json"

with open(test_data_path, 'r') as f:
    test_data = json.load(f)

print(f"Loaded test data with {len(test_data)} users.")

# Load base model and tokenizer (shared across fBase and fOPPU)
def load_model_and_tokenizer(model_name, adapter_path=None, access_token=None):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16, token=access_token
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if adapter_path:
        model = PeftModel.from_pretrained(model=model, model_id=adapter_path, is_trainable=False)
        model = model.merge_and_unload()
        print("[Base model lora adapter merge and unloaded]")
    return model, tokenizer
    
def compute_likelihood(model, prompt, responses):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True) 
    input_ids = inputs.input_ids.to(model.device) 
    attention_mask = inputs.attention_mask.to(model.device) 

    response_scores = []
    for response in responses:
        response_ids = tokenizer(response, return_tensors="pt", padding=True, truncation=True).input_ids.to(model.device) 
        full_input = torch.cat([input_ids, response_ids], dim=1) 

        with torch.no_grad():
            outputs = model(full_input, attention_mask=torch.ones_like(full_input))
            logits = outputs.logits[:, :-1, :]
            target_ids = full_input[:, 1:]

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction='mean'
            )
            response_scores.append(-loss.item())

    return response_scores

# Load Base model
print("\nLoading fBase model...")
model_name_short = args.model_name.split('/')[-1]
tam_lora_adapter_path = f"./ckpt/TAM/{args.task_name}/TAM-{model_name_short}_ckpt/"
for_oppu_base_model, tokenizer = load_model_and_tokenizer(model_name=args.model_name, adapter_path=tam_lora_adapter_path)

base_model, tokenizer = load_model_and_tokenizer(model_name=args.model_name, adapter_path=tam_lora_adapter_path)

# We'll create a DEEP COPY of the test_data so we can separately store fOPPU-only results
fOPPU_only_data = copy.deepcopy(test_data)

# Define the range for user processing
processed_indices = range(args.st, min(args.end,len(test_data)))

# Process each user for comparison
for user_idx in tqdm(processed_indices, desc="Processing Users"):
    user = test_data[user_idx]
    adapter_path = f'./ckpt/OPPU_SFT/{args.task_name}/user_{user_idx}'
    print(f"\nLoading fOPPU adapter for user {user['user_id']} from {adapter_path}")
    user_model = PeftModel.from_pretrained(for_oppu_base_model, adapter_path, is_trainable=False)
    user_model.eval()
    base_model.eval()
    
    # Evaluate each profile's responses using both models
    for profile_idx, profile in tqdm(enumerate(user["profile"]), total=len(user["profile"]), desc=f"Processing profiles for user {user['user_id']}"):
    
        if args.task_name in ["news_headline", "product_rating", "news_categorize"]:
            prompt_text = profile["text"]
        elif args.task_name == "scholarly_title":
            prompt_text = profile["abstract"]
        elif args.task_name == "movie_tagging":
            prompt_text = profile["description"]
        elif args.task_name == "abstract_generation":
            prompt_text = profile['title']
        elif args.task_name == "topic_writing":
            prompt_text = profile['summary']           
        elif args.task_name == "review_writing":
            overall = profile['overall']
            description = profile['description']
            summary = profile['summary']
            prompt_text = f"{overall} {description} {summary}"
        else:
            raise ValueError(f"Check whether title, abstract, or description is correct.")

        responses = [resp for resp in profile.get('pseudo_negative', [])]
    
        if not responses:
            continue
    
        if profile_idx < 3:
            print(f"\nEvaluating profile ID {profile['id']} with {len(responses)} responses.")
    
        # Compute scores for fOPPU and fBase
        OPPU_scores = compute_likelihood(user_model, prompt_text, responses)
        base_scores = compute_likelihood(base_model, prompt_text, responses)
    
        
        if profile_idx < 3:
            print(f"\nProfile ID {profile['id']} - Detailed Losses:")
            for idx2, (response, OPPU_loss, base_loss) in enumerate(zip(responses, OPPU_scores, base_scores)):
                diff = base_loss - OPPU_loss
                print(f"Response {idx2 + 1}:")
                print(f"  - Base Loss = {base_loss:.6f}")
                print(f"  - OPPU Loss = {OPPU_loss:.6f}")
                print(f"  - Difference (Base - OPPU) = {diff:.6f}\n")
    
        # Calculate the difference between fOPPU and fBase scores
        score_differences = [base - OPPU for OPPU, base in zip(OPPU_scores, base_scores)]
    
        # Select the best response based on the min or max of the differences
        if args.std == "min":
            best_idx = score_differences.index(min(score_differences))
        elif args.std == "max":
            best_idx = score_differences.index(max(score_differences))
        else:
            raise ValueError(f"Unexpected value for --std: {args.std}. Use 'min' or 'max'.")
    
        # Retrieve the best response and related info
        best_response = responses[best_idx]
        best_difference = score_differences[best_idx]
        OPPU_score = OPPU_scores[best_idx]
        base_score = base_scores[best_idx]
    
        # Print final selection details
        if profile_idx < 3:
            print(f"Best response selected: {best_response}")
            print(f"Score Difference (base - OPPU): {best_difference:.6f}")
            print(f"OPPU Score: {OPPU_score:.6f}, Base Score: {base_score:.6f}")
    
        # Update profile with selected results (base vs OPPU)
        profile['pseudo_negative'] = responses
        profile["OPPU_scores"] = OPPU_scores
        profile["base_scores"] = base_scores
        profile["score_differences"] = score_differences
        profile["final_score_difference"] = best_difference
        profile["negative_response"] = best_response
    
    print(f"Finished processing user {user['user_id']}.")

# Save the combined output for fOPPU vs fBase
output_file_path = f"./data/{args.task_name}/user_top_100_history_with_pseudo_negatives_{args.std}.json"
with open(output_file_path, 'w') as f:
    json.dump(test_data, f, indent=4)

print(f"\nFinal output saved to {output_file_path}")
