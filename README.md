# CoPe: Personalized LLM Decoding via Contrasting Personal Preference

CoPe is a decoding-time personalization framework for large language models (LLMs).  
It maximizes *implicit user reward* by contrasting personalized (PEFT-tuned) and base models at the token level — enabling personalization without external reward models or extra training.


## Introduction ## 

We present our new decoding framework for LLM personalization by Contrasting Personal preference (COPE). Our key idea is incorporating implicit reward signals for user preference to guide both training and inference.

## Dataset ## 
We utilize publicly available data from the [LaMP](https://arxiv.org/abs/2304.11406) and [LongLaMP](https://arxiv.org/abs/2407.11016) benchmark. And our work is fundamentally based on the [OPPU](https://arxiv.org/abs/2402.04401) setting.
You can download the our processed data [here](https://drive.google.com/file/d/147_uP-3A3XbEB8jwtaFkZXTXpLuybg8b/view?usp=sharing), extract the contents, and place them in the ```./data``` folder

## Install Requirements ##
First, create your own Python virtual environment using e.g., Conda:
```bash
conda create -n cope python=3.9 && conda activate cope
```

Next, use your conda environment to install all required packages by running:
```bash
pip install -r requirements.txt
```

## CoPe ## 
The task_name parameter supports any of the following options:
```[news_headline, scholarly_title, abstract_generation, review_writing, topic_writing]```
In this example, we demonstrate using ```news_headline```.

## TAM ##

Task-Adaptive Model (TAM) is trained on data from users excluding the target user, adapting the base model to the overall task domain.
```bash

```


## OPPU ##


## Make DPO negative pairs ##


## Select best negative pair ##


## DPO training ##


## Inference with contrastive decoding ## 




## Citation
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
