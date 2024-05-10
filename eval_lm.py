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

TASKS_WE_USE = [
    {
        "name": "gsm8k",
        "num_shots": 5,
        "is_gen": True,
        "in_openllm": True,
        "metric": "exact_match,strict-match",
    },
    {
        "name": "bbh_cot_fewshot_date_understanding",
        "num_shots": None,
        "is_gen": True,
        "in_openllm": False,
        "metric": "exact_match,get-answer",
    },
    {
        "name": "bbh_cot_fewshot_movie_recommendation",
        "num_shots": None,
        "is_gen": True,
        "in_openllm": False,
        "metric": "exact_match,get-answer",
    },
    {
        "name": "bbh_cot_fewshot_reasoning_about_colored_objects",
        "num_shots": None,
        "is_gen": True,
        "in_openllm": False,
        "metric": "exact_match,get-answer",
    },
]

TASK_TO_METRIC = {v["name"]: v["metric"] for v in TASKS_WE_USE}
TASK_TO_NUM_SHOT = {v["name"]: v["num_shots"] for v in TASKS_WE_USE}
ALL_TASKS = [v["name"] for v in TASKS_WE_USE]
GEN_TASKS = set([v["name"] for v in TASKS_WE_USE if v["is_gen"]])
OPENLLM_TASKS = set([v["name"] for v in TASKS_WE_USE if v["in_openllm"]])

"""
Doc for the args
-m is the model name. Use this to pass the path to a finetuned model 
-b pass this to tell it what base model to use (non finetuned)
-o path to save test results to
"""
parser = argparse.ArgumentParser(prog="training")
parser.add_argument("-m", "--model", required=False)
parser.add_argument("-b", "--base", required=False)
parser.add_argument("-o", "--output")
args = parser.parse_args()

model_save_path, base, output = args.model, args.base, args.output

# python eval_lm.py -m t -b 7bUc -o llama_chat_base

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
    project="codemath-eval",
    config={},
    name=f"{model_save_path}",
)


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_save_path,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)


tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

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


class LMEvalArguments:
    output_path: str


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def get_performance(all_results, all_tasks):
    metrics = {}
    all_averages = []
    openllm_averages = []
    classification_average = []
    generation_average = []
    for task, task_result in all_results["results"].items():
        if task in all_tasks:
            # clean the "acc,none" to "acc"
            task_result_cleaned = {}
            for k, v in task_result.items():
                if k == "alias":
                    continue
                k = k.replace(",none", "")
                task_result_cleaned[k] = v

                # get the average
                if k != TASK_TO_METRIC[task]:
                    continue
                ### now v is the metric of interest
                all_averages.append(v)
                # openllm
                if task in OPENLLM_TASKS:
                    openllm_averages.append(v)
                # gen or classification
                if task in GEN_TASKS:
                    generation_average.append(v)
                else:
                    classification_average.append(v)
            metrics[task] = task_result_cleaned
    metrics["openllm_average"] = np.mean(openllm_averages).item()
    metrics["classification_average"] = np.mean(classification_average).item()
    metrics["generation_average"] = np.mean(generation_average).item()
    metrics["all_average"] = np.mean(all_averages).item()

    ### save this thing
    # path = Path(args.output_path)
    # output_path_file = path.joinpath("performance.json")
    try:
        output_path_file = Path(f"{output}_performance.json")
        dumped = json.dumps(
            metrics, indent=2, default=_handle_non_serializable, ensure_ascii=False
        )
        with output_path_file.open("w", encoding="utf-8") as f:
            f.write(dumped)
    except:
        print("error saving")
    return metrics


for task in ALL_TASKS:
    print(f"Running task {task} with {TASK_TO_NUM_SHOT[task]} shots")
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=[task],
        # num_fewshot=3,
        # num_examples=3,
        task_manager=task_manager,
    )

    # print the results in prev_results
    all_tasks = ALL_TASKS

    performance = get_performance(results, all_tasks)
    print(json.dumps(performance, indent=2, ensure_ascii=False))

    ## upload results
    # if args.wandb_id != "":
    # wandb.init(project=args.wandb_project, id=args.wandb_id, resume=True)
    wandb.init(
        project="codemath-eval",
        config={},
        name=f"{model_save_path}_{task}",
    )
    wandb_perf = {f"lm_eval/{k}": v for k, v in performance.items()}
    wandb.log(wandb_perf)
    wandb.finish()
