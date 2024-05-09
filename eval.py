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

# python train.py -o models/llama7b -m 7bU -u -d datasets/Training\ Trace\ Dataset.json -ev

# custom imports
from prompts import (
    TRACE_PROMPT,
    EVAL_TEMPLATE,
    PREAMBLE,
    GSM8K_FEW_PROMPT,
    PST,
    MISTRAL_EVAL,
)

os.environ["WANDB_MODE"] = "offline"

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

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

if base:
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Configuration for PEFT, adjust as needed
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )


def format_gsm8k(examples):
    inputs = examples["question"]
    outputs = examples["answer"]
    texts = []
    if zero_shot:
        for input, output in zip(inputs, outputs):
            text = EVAL_TEMPLATE.format(PREAMBLE, input, output)
            texts.append(text)
    else:
        for input, output in zip(inputs, outputs):
            text = EVAL_TEMPLATE.format(PREAMBLE, GSM8K_FEW_PROMPT, input, output)
            texts.append(text)
    return {"text": texts}


def correct_solution_gsm8k(prediction_str, reference_str):
    """
    Compare the final numerical output of the model with the reference tokens.

    Args:
    - prediction_tokens: List of token IDs representing the model's prediction.
    - reference_tokens: List of token IDs representing the reference output.

    Returns:
    - 1 if the final numerical output of the model matches the reference tokens exactly, else 0.
    """
    # prediction_str = tokenizer.decode(prediction_tokens, skip_special_tokens=True)
    # reference_str = tokenizer.decode(reference_tokens, skip_special_tokens=True)

    # print(prediction_str)
    # print("##################")
    # print(reference_str)
    # exit(1)

    gt = re.findall(r"\d+", reference_str.strip().split("\n")[-1].strip())

    if any(num in prediction_str.split("[/INST]")[1] for num in gt):
        return 1
    else:
        return 0


# Helper function to encode inputs
def prepare_input(input):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    # print(question)
    if zero_shot:
        prompt = EVAL_TEMPLATE.format(PREAMBLE, "", input, "")
    else:
        prompt = EVAL_TEMPLATE.format(PREAMBLE, GSM8K_FEW_PROMPT, input, "")
    return tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        # padding_side="left",
    ).input_ids


# Function to decode model output
def decode_output(output_ids):
    return tokenizer.decode(output_ids, skip_special_tokens=True)


gsm8k_test = load_dataset("gsm8k", "main", split="test")

# Assuming model and tokenizer are already initialized
model.eval()  # Set the model to evaluation mode


# Manual testing loop
all_correct = 0
all_responses = {}
idx = 0
total = len(gsm8k_test)
total = 400

for task_id, problem in enumerate(gsm8k_test, start=1):
    if idx == total:
        break

    print(f"task_id {task_id}")

    input_ids = prepare_input(problem["question"])
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=120,
        )

    response = decode_output(output_ids[0])
    all_responses[task_id] = response
    all_correct += correct_solution_gsm8k(response, problem["answer"])

    # print(f"Model answer: {model_number}")
    # print(f"Ground truth answer: {ground_truth_number}")
    print(f"Correct: {all_correct} out of {total}")
    print("=" * 40)
    wandb.log(
        {
            "final_eval_gsm8k_total_correct": all_correct,
            "final_eval_gsm8k_current_pct_correct": all_correct / task_id,
        }
    )
    idx += 1


# Final accuracy
accuracy = all_correct / len(gsm8k_test)
print(f"Final Accuracy: {accuracy:.2f}")
