import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from ast import literal_eval
import wandb
import os
from collections import Counter
import numpy as np
from transformers import AdamW
from torch.utils.data import DataLoader
import re

# WANDB
# os.environ["WANDB_PROJECT"] = "codemath"
# os.environ["WANDB_LOG_MODEL"] = "true"
# os.environ["WANDB_WATCH"] = "false"
# os.environ["WANDB_MODE"]="offline"
# FILE CONFIGS CHANGE SETTINGS HERE
# PRETRAINED_MODEL = True
PRETRAINED_MODEL = False
PRETRAIN = False
max_seq_length = 2048
json_file_path = "./python_states_singleline.json"
# trace_prompt_trace = """<s>[INST] Below is an input which contains the state of variables and code that acts upon these variables or not. Given the state and the code give the state after the code executes for each variable. Be very careful. You should clearly outline your intermediate steps and your final answer should be a newline with exactly the variables and their values. Here is the State and Code. {}
# Now generate the final state for each variable. Generate intermediate outputs.[/INST] {}</s>"""
# trace_prompt_gsm8k = """<s>[INST] {} [/INST] {}</s>"""
# trace_prompt = trace_prompt_trace if PRETRAIN else trace_prompt_gsm8k
model_save_path = (
    "model_save_path/mistral_7b_pretrain_trace_finetuned_gsm8k_with_eval"
    if PRETRAINED_MODEL
    else "model_save_path/mistral_7b_finetuned_gsm8k_with_eval"
)

# export CUDA_VISIBLE_DEVICES=1

wandb.init(
    project="codemath",
    config={
        "pretrained": PRETRAINED_MODEL,
        "architecture": "Mistral 7B",
        # "dataset": "trace python and gsm8K" if PRETRAINED_MODEL else "gsm8k only",
        "dataset": "TRACED",
        "epochs": 5,
    },
    name="big_refactor_slp",
)


# LOAD IN MODEL DEPENDING ON FLAG
if PRETRAINED_MODEL:
    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name="model_save_path/mistral_7b_finetuned_trace_python_with_mtoleseval",
    #     max_seq_length=max_seq_length,
    #     dtype=None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    #     load_in_4bit=True,  # Use 4bit quantization to reduce memory usage. Can be Fals
    # )
    pass
else:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit=True,  # Use 4bit quantization to reduce memory usage. Can be False
    )

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

alpaca_prompt = """<s> [INST] Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Write down all of the state changes that take place after the code snippet is executed in the format
variable1 = value1; variable2 = value2; etc...

### Input:
{}
 [/INST] 
### Response:
{}
</s>"""


# SETUP FUNCTIONS
def formatting_prompts_func(examples):
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for input, output in zip(inputs, outputs):
        text = alpaca_prompt.format(input, output)
        texts.append(text)
    return {"text": texts}


# LOAD IN DATASET AND TRAIN EVAL SPLIT
# dataset = load_dataset("gsm8k", "main", split="train")
dataset = load_dataset("json", data_files=json_file_path, split="train").select(
    range(10000)
)
dataset = dataset.map(formatting_prompts_func, batched=True)  # had to unset batched
# split_ratio = 0.1
split_ratio = 0.01
split_datasets = dataset.train_test_split(test_size=split_ratio)
train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]

# train_dataset = train_dataset.select([0, 20])
# eval_dataset = eval_dataset.select([2])


# EVAL LOGIC
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


def correct_solution(prediction_str, reference_str):
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

    prediction_lines = prediction_str.strip().split("\n")
    reference_lines = reference_str.strip().split("\n")

    # print(prediction_lines)
    # print(reference_lines)

    last_prediction_line = prediction_lines[-1].strip()
    last_reference_line = reference_lines[-1].strip()

    # print("predicted ",last_prediction_line)
    # print("reference ",last_reference_line)
    # print(last_prediction_line== last_reference_line)

    last_prediction_num = re.findall(r"\d+", last_prediction_line)
    last_reference_num = re.findall(r"\d+", last_reference_line)

    if last_prediction_num == last_reference_num:
        return 1
    else:
        return 0


def str_to_objs(input_str: str):
    """
    converts a string to a dict of objects
    input_str: str
        Must be a string of the form 'i = 4; p = [0, 1, 1, 2, 5];'
    returns: dict
    """
    items = [x.strip() for x in input_str.split(";")]
    objs = dict()
    for item in items:
        try:
            key, value_str = item.split("=")
            key = key.strip()
        except:
            continue  # no key found
        try:
            value_str = value_str.strip()

            # quick and dirty object conversion to make all literals hashable
            # TODO: create a hashable class for dicts and lists
            literal_value = literal_eval(value_str)  # safe eval
            value = make_hashable(literal_value)
            objs[key] = value
        except:
            objs[key] = "NONLITERAL_STRING"
    return objs


def make_hashable(obj):
    if isinstance(obj, dict):
        # Convert dict to a sorted tuple of key-value pairs, making keys/values hashable recursively
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    if isinstance(obj, set):
        return frozenset(obj)
    if isinstance(obj, tuple):
        return tuple(make_hashable(item) for item in obj)
    # recursion case
    elif isinstance(obj, list):
        # Convert lists to tuples
        return tuple(make_hashable(item) for item in obj)
    else:
        # Assume the object is hashable (e.g., numbers, strings, tuples)
        return obj


def custom_metrics(preds):
    print(preds)
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
    f1s = []
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
        try:
            first_pred_output_stop_idx = torch.where(
                pred_output_tokens_unmasked == tokenizer.eos_token_id
            )[0][0]
        except:
            try:
                first_pred_output_stop_idx = torch.where(
                    pred_output_tokens_unmasked == tokenizer.pad_token_id
                )[0][0]
            except:
                first_pred_output_stop_idx = -1
        pred_output_tokens = pred_output_tokens_unmasked[:first_pred_output_stop_idx]

        gt_output_str = tokenizer.decode(gt_output_tokens)
        pred_output_str = tokenizer.decode(pred_output_tokens)

        # compare gt/preds interpreted in python
        gt_state = str_to_objs(gt_output_str)
        pred_state = str_to_objs(pred_output_str)
        # compute f1 for values in the two states
        gt_vars = set(gt_state.items())
        pred_vars = set(pred_state.items())
        try:
            precision = len(gt_vars.intersection(pred_vars)) / len(pred_vars)
            recall = len(gt_vars.intersection(pred_vars)) / len(gt_vars)
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1 = 0
        f1s.append(f1)
    f1_mean = torch.tensor(f1s).mean().item()
    wandb.log(
        {
            "perplexity": ppl.item(),
            "correct_tokens": correct_tokens.item(),
            "f1": f1_mean,
        }
    )
    return {"perplexity": ppl, "correct_tokens": correct_tokens.item(), "f1": f1_mean}


# TRAINING ONLY RUN TO SAVE MODEL FOR LATER TESTING
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Packing setting
    args=TrainingArguments(
        report_to="wandb",
        per_device_train_batch_size=40,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=5,
        learning_rate=5e-4,
        # learning_rate=0,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="constant",  # don't want to decay this for now
        # lr_scheduler_type="linear", # don't want to decay this for now
        seed=3407,
        output_dir="outputs",
        evaluation_strategy="steps",
        # eval_steps=10,
        eval_steps=1,
        do_eval=True,
        eval_accumulation_steps=50,
    ),
)

trainer.compute_metrics = custom_metrics
trainer.train()
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
wandb.finish()
