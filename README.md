# Personalized LLM Decoding via Contrasting Personal Preference

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

## Training ##


