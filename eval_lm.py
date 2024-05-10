import lm_eval
import json
import numpy as np
import time
import torch
import wandb
import yaml
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, field
from pathlib import Path
from lm_eval import evaluator
from lm_eval.tasks import TaskManager
import argparse
import os
import torch
import argparse
import numpy as np
import json
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, Dataset, DatasetDict
from unsloth import FastLanguageModel
from trl import SFTTrainer
from collections import Counter
import re


parser = argparse.ArgumentParser(prog="training")
parser.add_argument("-m", "--model")
parser.add_argument("-b", "--base")
parser.add_argument("-z", "--zero", action="store_true")
# parser.add_argument("-d", "--data")
# parser.add_argument("-o", "--output")
args = parser.parse_args()

model_save_path, base, zero_shot = args.model, args.base, args.zero

# model_save_path, data_path, model_name, unsloth, evaluation_mode = (
#     args.output,
#     args.data,
#     args.model,
#     args.unsloth,
#     args.ev,
# )

models = {
    "13bM": "meta-llama/Llama-2-13b-chat-hf",
    "7bM": "meta-llama/Llama-2-7b-chat-hf",
    "7bUc": "unsloth/llama-2-7b-chat-bnb-4bit",
    "7bU": "unsloth/llama-2-7b-bnb-4bit",
    "13bU": "unsloth/llama-2-13b-bnb-4bit",
    "7bCodeU": "unsloth/codellama-7b-bnb-4bit",
    "Mistral7bU": "unsloth/mistral-7b-bnb-4bit",
    "Mistral7bUInstr": "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
}

if base:
    model_save_path = models[base]


wandb.init(
    project="codemath-final",
    config={
        "dataset": "gsm8k",
    },
    name=f"{model_save_path}_zero_{zero_shot}",
)


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_save_path,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token_id = tokenizer.eos_token_id
# tokenizer.padding_side = "left"

task_manager = TaskManager(include_path=None)  # no include path needed

# model = model.to("cuda")
model = model.eval()

lm = lm_eval.api.registry.get_model(model_name="hf").create_from_arg_string(
    "",
    {
        "pretrained": model,
        "tokenizer": tokenizer,
        "trust_remote_code": True,
        "batch_size": 8,
    },
)

results = evaluator.simple_evaluate(
    model=lm,
    tasks=[
        "gsm8k",
        "bbh_cot_fewshot_date_understanding",
        "bbh_cot_fewshot_movie_recommendation",
        "bbh_cot_fewshot_reasoning_about_colored_objects",
    ],
    num_fewshot=3,
    task_manager=task_manager,
)


# TASKS_WE_USE = [
#     {'name': 'hellaswag', 'num_shots': 10, 'is_gen': False, 'in_openllm': True, 'metric': 'acc_norm'},
#     {'name': 'arc_challenge', 'num_shots': 25, 'is_gen': False, 'in_openllm': True, 'metric': 'acc_norm'},
#     {'name': 'truthfulqa_mc2', 'num_shots': 0, 'is_gen': False, 'in_openllm': True, 'metric': 'acc'},
#     {'name': 'winogrande', 'num_shots': 5, 'is_gen': False, 'in_openllm': True, 'metric': 'acc'},
#     {'name': 'gsm8k', 'num_shots': 5, 'is_gen': True, 'in_openllm': True, 'metric': 'exact_match,strict-match'},
#     {'name': 'mmlu', 'num_shots': 5, 'is_gen': False, 'in_openllm': True, 'metric': 'acc'},
#     {'name': 'bbh_cot_fewshot_date_understanding', 'num_shots': None, 'is_gen': True, 'in_openllm': False, 'metric': 'exact_match,get-answer'},
#     {'name': 'bbh_cot_fewshot_movie_recommendation', 'num_shots': None, 'is_gen': True, 'in_openllm': False, 'metric': 'exact_match,get-answer'},
#     {'name': 'bbh_cot_fewshot_reasoning_about_colored_objects', 'num_shots': None, 'is_gen': True, 'in_openllm': False, 'metric': 'exact_match,get-answer'}
# ]
