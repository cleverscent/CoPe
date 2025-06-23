import re
import torch
import torch.nn.functional as F

def extract_option(s, num):
    # Look for string after [1]: and between "
    match = re.search(r'\[' + str(num) + '\]: "([^"]*)"', s)
    return match.group(1) if match else None

def extract_citation_title(text):
    pattern = r'written the paper with the title "([^"]*)"'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return None


def extract_movie(text):
    marker = "] description: "
    # Find the position of the marker in the text
    marker_pos = text.find(marker)
    
    # Check if the marker is found
    if marker_pos == -1:
        raise ValueError()

    # Extract the string after the marker
    extracted_string = text[marker_pos + len(marker):]

    return extracted_string

def extract_news_cat(text):
    marker = "] article: "
    # Find the position of the marker in the text
    marker_pos = text.find(marker)
    
    # Check if the marker is found
    if marker_pos == -1:
        raise ValueError()

    # Extract the string after the marker
    extracted_string = text[marker_pos + len(marker):]

    return extracted_string

def extract_news_headline(text):
    marker = "Generate a headline for the following article: "
    # Find the position of the marker in the text
    marker_pos = text.find(marker)
    
    # Check if the marker is found
    if marker_pos == -1:
        raise ValueError()

    # Extract the string after the marker
    extracted_string = text[marker_pos + len(marker):]

    return extracted_string

def extract_product_review(text):
    marker = "without further explanation. review: "
    # Find the position of the marker in the text
    marker_pos = text.find(marker)
    
    # Check if the marker is found
    if marker_pos == -1:
        raise ValueError()

    # Extract the string after the marker
    extracted_string = text[marker_pos + len(marker):]

    return extracted_string


def extract_scholarly_title(text):
    marker = "Generate a title for the following abstract of a paper: "
    # Find the position of the marker in the text
    marker_pos = text.find(marker)
    
    # Check if the marker is found
    if marker_pos == -1:
        raise ValueError()

    # Extract the string after the marker
    extracted_string = text[marker_pos + len(marker):]

    return extracted_string


def extract_tweet_paraphrasing(text):
    marker = "Paraphrase the following tweet without any explanation before or after it: "
    # Find the position of the marker in the text
    marker_pos = text.find(marker)
    
    # Check if the marker is found
    if marker_pos == -1:
        raise ValueError()

    # Extract the string after the marker
    extracted_string = text[marker_pos + len(marker):]

    return extracted_string

def extract_abstract_generation(text):
    marker = "Generate an abstract for the title "
    
    marker_pos = text.find(marker)
    
    if marker_pos == -1:
        raise ValueError("Marker for abstract generation not found.")
    
    # Extract everything after the marker
    extracted_string = text[marker_pos + len(marker):]
    return extracted_string
    

def extract_review_writing(text):
    # Extract rating
    rating_match = re.search(r'overall rating of "([^"]+)"', text)
    overall = rating_match.group(1) if rating_match else "N/A"

    # Extract product description
    description_match = re.search(r'product with description "([^"]+)"', text)
    description = description_match.group(1) if description_match else "N/A"

    # Extract review summary
    summary_match = re.search(r'summary of the review text is "([^"]+)"', text)
    summary = summary_match.group(1) if summary_match else "N/A"

    return overall, description, summary
    
    
def extract_topic_writing(text):
    topic_marker = "Generate the content for a reddit post "
    
    topic_start = text.find(topic_marker)
    if topic_start == -1:
        raise ValueError("Topic marker not found.")
    
    topic = text[topic_start + len(topic_marker):].strip()
    
    return topic


def get_first_k_tokens(text, k):
    """
    Extracts the first k tokens from a text string.

    :param text: The input text string.
    :param k: The number of tokens to extract.
    :return: The first k tokens of the text string.
    """
    # Split the text into tokens based on whitespace
    tokens = text.split()
    output = " ".join(tokens[:k])

    # Return the first k tokens
    return output

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list

def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# def get_output(output, args):
#     if args.task_name == "news_headline":
#         headline_start = output.rfind("headline:")
#         if headline_start != -1:  
#             result = output[headline_start + len("headline:"):].strip() 
#         else:
#             result = "" 
#     elif args.task_name == "scholarly_title":
#         title_start = output.rfind("title:")
#         if title_start != -1: 
#             result = output[title_start+ len("title:"):].strip()
#         else:
#             result = ""
#     elif args.task_name == "abstract_generation":
#         abstract_start = output.rfind("Abstract:")
#         if abstract_start != -1:
#             result = output[abstract_start + len("Abstract:"):].strip()
#         else:
#             result = ""
#     elif args.task_name == "review_writing":
#         review_start = output.rfind("Review:")
#         if review_start != -1:
#             result = output[review_start + len("Review:"):].strip()
#         else:
#             result = ""
#     elif args.task_name == "topic_writing":
#         post_start = output.rfind("content:")
#         if post_start != -1:
#             result = output[post_start + len("content:"):].strip()
#         else:
#             result = ""
#     return result

def get_output(output, args):
    model_name = args.model_name.lower()

    def strip_quotes(text):
        return text.strip().strip('"').strip("'")

    if "mistral" in model_name:
        if args.task_name == "news_headline":
            headline_start = output.rfind("headline:")
            result = output[headline_start + len("headline:"):].strip() if headline_start != -1 else ""
        elif args.task_name == "scholarly_title":
            title_start = output.rfind("title:")
            result = output[title_start + len("title:"):].strip() if title_start != -1 else ""
        elif args.task_name == "abstract_generation":
            abstract_start = output.rfind("Abstract:")
            result = output[abstract_start + len("Abstract:"):].strip() if abstract_start != -1 else ""
        elif args.task_name == "review_writing":
            review_start = output.rfind("Review:")
            result = output[review_start + len("Review:"):].strip() if review_start != -1 else ""
        elif args.task_name == "topic_writing":
            post_start = output.rfind("content:")
            result = output[post_start + len("content:"):].strip() if post_start != -1 else ""
        else:
            result = output.strip()

    elif "llama" or "gemma" in model_name:
        lines = output.strip().splitlines()
        candidate_lines = [line.strip() for line in lines if line.strip().startswith('"') or line.strip().endswith('"')]

        if args.task_name == "news_headline":
            headline_start = output.rfind("headline:")
            result = output[headline_start + len("headline:"):].strip() if headline_start != -1 else ""
        if args.task_name == "scholarly_title":
            # 번호로 시작하는 제목 리스트 중 가장 첫 번째만 추출
            title_lines = [line for line in lines if line.strip().startswith("1.")]
            result = title_lines[0].split("1.", 1)[-1].strip() if title_lines else lines[-1].strip()
        elif args.task_name == "abstract_generation":
            abstract_start = output.rfind("Abstract:")
            result = output[abstract_start + len("Abstract:"):].strip() if abstract_start != -1 else ""
        else:
            result = candidate_lines[-1] if candidate_lines else lines[-1].strip()

    return strip_quotes(result)


name2taskid = {
    "citation": "LaMP_1",
    "movie_tagging": "LaMP_2M",
    "news_categorize": "LaMP_2N",
    "news_headline": "LaMP_4",
    "product_rating": "LaMP_3",
    "scholarly_title": "LaMP_5",
    "tweet_paraphrase": "LaMP_7",
    "abstract_generation": "LongLaMP_2",
    "review_writing": "LongLaMP_3",
    "topic_writing": "LongLaMP_4",
    "gsm":"gsm"
}
