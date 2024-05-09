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
from prompts import TRACE_PROMPT, EVAL_TEMPLATE, PREAMBLE, GSM8K_FEW_PROMPT, PST_PROMPT

os.environ["WANDB_MODE"] = "offline"

parser = argparse.ArgumentParser(prog="training")
parser.add_argument("-m", "--model")
parser.add_argument("-d", "--data")
parser.add_argument("-o", "--output")
parser.add_argument("-u", "--unsloth", action="store_true")
parser.add_argument("-ev", "--ev", action="store_true")
parser.add_argument("-traced", "--traced", action="store_true")
parser.add_argument("-pst", "--pst", action="store_true")
args = parser.parse_args()

model_save_path, data_path, model_name, unsloth, evaluation_mode, traced, pst = (
    args.output,
    args.data,
    args.model,
    args.unsloth,
    args.ev,
    args.traced,
    args.pst,
)

models = {
    "13bM": "meta-llama/Llama-2-13b-chat-hf",
    "7bM": "meta-llama/Llama-2-7b-chat-hf",
    "7bUc": "unsloth/llama-2-7b-chat-bnb-4bit",
    "7bU": "unsloth/llama-2-7b-bnb-4bit",
    "13bU": "unsloth/llama-2-13b-bnb-4bit",
    "7bCodeU": "unsloth/codellama-7b-bnb-4bit",
    "Mistral7bU": "unsloth/mistral-7b-bnb-4bit",
    "Mistral7bUInstruct": "unsloth/mistral-7b-bnb-4bit",
}

wandb.init(
    project="codemath-FINAL_EVAL",
    config={
        "dataset": f"{data_path}",
    },
    name=f"{models[model_name]}_{'pst' if pst else 'traced'}",
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=models[model_name],
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
tokenizer.pad_token = tokenizer.eos_token

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


# both functions have the eos token appended in the formatte3d prompt
def format_trace_data(examples, prompt=TRACE_PROMPT):
    entry_vars = examples["entry_variables"]
    src_seq = examples["src_seq"]
    value_seq = examples["value_type_seq"]
    abstract_seq = examples["abstract_value_seq"]
    texts = []
    for entry_vars, src_seq, value_seq, abstract_seq in zip(
        entry_vars, src_seq, value_seq, abstract_seq
    ):
        text = prompt.format(src_seq, entry_vars, abstract_seq, value_seq)
        texts.append(text)
    return {"text": texts}


def format_pst_data(examples):
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for input, output in zip(inputs, outputs):
        text = PST.format(input, output)
        texts.append(text)
    return {"text": texts}


def format_gsm8k(examples):
    inputs = examples["question"]
    outputs = examples["answer"]
    texts = []
    for input, output in zip(inputs, outputs):
        text = EVAL_TEMPLATE.format(PREAMBLE, GSM8K_FEW_PROMPT, input, output)
        texts.append(text)
    return {"text": texts}


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=512
    )


def extract_number_from_text(text, prefix="answer is"):
    """
    Extracts the last number from a text string that follows a given prefix.
    Args:
        text (str): The text from which to extract the number.
        prefix (str): The prefix to search for before extracting the number.
    Returns:
        float or None: The extracted number, or None if no valid number is found.
    """
    # Find the part of the text that starts with the prefix
    # print(text)
    match = re.search(re.escape(prefix) + r".*", text)
    if match:
        # Extract all numbers from the matched text
        numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+", match.group(0))
        if numbers:
            # Return the last number found as a float
            last_number = numbers[-1]
            try:
                return float(last_number)
            except ValueError:
                print(f"Could not convert '{last_number}' to float.")
                return None
    return None


def calculate_token_level_f1(prediction_tokens, reference_tokens):
    """
    Calculate precision, recall, and F1 score based on token overlap.
    """
    common_token_count = Counter(prediction_tokens) & Counter(reference_tokens)
    num_same = sum(common_token_count.values())

    if num_same == 0:
        return 0, 0, 0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(reference_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


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

    print(prediction_str)

    gt = re.findall(r"\d+", reference_str.strip().split("\n")[-1].strip())

    if any(num in prediction_str for num in gt):
        return 1
    else:
        return 0


def custom_metrics_gsm8k(preds):
    # TODO Changed this function group to work with gsm8k
    logits = torch.tensor(preds.predictions)
    labels = torch.tensor(preds.label_ids)

    batch_size, seq_length, vocab_size = logits.shape

    # steal from inside llama
    # shift logits by 1 index cuz of causal lm
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    # loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(batch_size, -1, vocab_size)
    shift_labels = shift_labels.view(batch_size, -1)

    probs = torch.nn.functional.softmax(shift_logits.view(-1, vocab_size), dim=-1)
    p_true_tokens = probs.view(-1, vocab_size)[
        torch.arange(batch_size * (seq_length - 1)), shift_labels.view(-1)
    ].view(batch_size, (seq_length - 1))

    nll = -torch.log(p_true_tokens)
    mean_nll = nll.mean()
    ppl = torch.exp(mean_nll)

    # compute percentage of correct tokens
    correct_tokens = (
        (shift_logits.view(-1, vocab_size).argmax(-1) == shift_labels.view(-1))
        .float()
        .mean()
    )

    pred_max_labels = shift_logits.argmax(-1).view(batch_size, -1)

    f1_scores = []
    precision_scores = []
    recall_scores = []
    solution_scores = []

    for i in range(batch_size):
        unmasked_label_tokens = shift_labels[i][shift_labels[i] != -100][
            :-1
        ]  # drop eos_token
        # find the index where the instruction token ends and the answer begins
        inst_token_seq = tokenizer.encode("[/INST]", return_tensors="pt")[0][1:]
        first_output_idx = None
        for j in range(unmasked_label_tokens.shape[0] - len(inst_token_seq)):
            if torch.equal(
                unmasked_label_tokens[j : j + len(inst_token_seq)], inst_token_seq
            ):
                first_output_idx = j + len(inst_token_seq)
                break
        assert (
            first_output_idx is not None
        ), "Could not find the end of the instruction token"

        # get ground truth output tokens
        gt_output_tokens = unmasked_label_tokens[first_output_idx:]
        # get predicted output tokens (including padding)
        pred_output_tokens_masked = pred_max_labels[i][first_output_idx:]
        # drop the pad tokens
        pred_output_tokens_unmasked = pred_output_tokens_masked[
            pred_output_tokens_masked != -100
        ]

        eos_token_indices = torch.where(
            pred_output_tokens_unmasked == tokenizer.eos_token_id
        )[0]

        if eos_token_indices.size(0) > 0:
            first_pred_output_stop_idx = eos_token_indices[0].item()
        else:
            first_pred_output_stop_idx = len(pred_output_tokens_unmasked) - 1

        pred_output_tokens = pred_output_tokens_unmasked[:first_pred_output_stop_idx]

        gt_output_str = tokenizer.decode(gt_output_tokens)
        pred_output_str = tokenizer.decode(pred_output_tokens)

        precision, recall, f1 = calculate_token_level_f1(pred_output_str, gt_output_str)

        correct = correct_solution_gsm8k(pred_output_str, gt_output_str)
        solution_scores.append(correct)

        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

    mean_f1 = np.mean(f1_scores) if f1_scores else 0
    mean_precision = np.mean(precision_scores) if precision_scores else 0
    mean_recall = np.mean(recall_scores) if recall_scores else 0
    solve_rate = np.mean(solution_scores) if solution_scores else 0

    wandb.log(
        {
            "gsm8k_train_solve_rate": solve_rate,
            "perplexity": ppl.item(),
            "correct_tokens": correct_tokens.item(),
            "f1": mean_f1,
            "mean_precision": mean_precision,
            "mean_recall": mean_recall,
        }
    )
    return {
        "perplexity": ppl,
        "correct_tokens": correct_tokens.item(),
        "f1": mean_f1,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "solve_rate": solve_rate,
    }


### TRAINING ###


## Load Dataset and train ##
if traced:
    dataset_traced = load_dataset("json", data_files=data_path, split="train")
    dataset_traced = dataset_traced.map(
        format_trace_data, batched=True
    ).train_test_split(train_size=0.05)
elif pst:
    dataset_traced = load_dataset("json", data_files=data_path, split="train")
    dataset_traced = dataset_traced.map(format_pst_data, batched=True).train_test_split(
        train_size=0.005
    )
# dataset_traced = load_dataset("gsm8k", "main", split="train")
# dataset_traced = dataset_traced.map(format_gsm8k, batched=True).train_test_split(
#     train_size=0.05
# )

dataset_gsm8k = load_dataset("gsm8k", "main", split="test")
eval_dataset = dataset_gsm8k.map(format_gsm8k, batched=True).train_test_split(
    test_size=0.1
)

eval_args = TrainingArguments(
    report_to="wandb",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=1,
    learning_rate=0,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="constant",
    seed=3407,
    output_dir=f"outputs/eval/{model_name}",
    evaluation_strategy="steps",
    eval_steps=1,
    do_eval=True,
    eval_accumulation_steps=50,
)

train_args = TrainingArguments(
    report_to="wandb",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    num_train_epochs=1,
    # max_steps=10,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="constant",
    seed=3407,
    output_dir=f"outputs/train/{model_name}",
    evaluation_strategy="steps",
    eval_steps=100,
    do_eval=True,
    eval_accumulation_steps=50,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset_traced["train"],
    eval_dataset=eval_dataset["test"],
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,  # Packing setting
    args=eval_args if evaluation_mode else train_args,
)

# Train model
trainer.compute_metrics = custom_metrics_gsm8k
trainer.train()

if not evaluation_mode:
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

wandb.finish()