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

prompt_slp = """<s>[INST] Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Write down all of the state changes that take place after the code snippet is executed in the format
variable1 = value1; variable2 = value2; etc...

### Input:
{}
[/INST]
### Response:
{}
</s>"""
prompt_gsm8k = """<s>[INST] Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
As an expert problem solver solve step by step the following mathematical questions. Here are some examples:

Input: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Response: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.

Input: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Response: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.

Input: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Response: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.

Input: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Response: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.

Input: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Response: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.

Input: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Response: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.

Input: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Response: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.

Input: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Response: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.

### Input:
{}
[/INST]
### Response:
{}
</s>"""

# SETUP FUNCTIONS
def formatting_prompts_func_slp(examples):
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for input, output in zip(inputs, outputs):
        text = prompt_slp.format(input, output)
        texts.append(text)
    return {"text": texts}
# SETUP FUNCTIONS
def formatting_prompts_func_gsm8k(examples):
    inputs = examples["question"]
    outputs = examples["answer"]
    texts = []
    for input, output in zip(inputs, outputs):
        text = prompt_gsm8k.format(input, output)
        texts.append(text)
    return {"text": texts}

# LOAD IN DATASET AND TRAIN EVAL SPLIT
# dataset = load_dataset("gsm8k", "main", split="train")
dataset_slp = load_dataset("json", data_files=json_file_path, split="train").select(range(10000)) 
dataset_slp = dataset_slp.map(formatting_prompts_func_slp, batched=True) # had to unset batched
# split_ratio = 0.1
# split_datasets_slp = dataset_slp.train_test_split(test_size=split_ratio)
# train_dataset_slp = split_datasets_slp["train"]
# eval_dataset_slp = split_datasets_slp["test"]


# load the gsm8k dataset as the eval dataset
dataset_gsm8k = load_dataset("gsm8k", "main", split="train")
eval_dataset_gsm8k = dataset_gsm8k.map(formatting_prompts_func_gsm8k, batched=True)

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


def correct_solution_slp(prediction_str, reference_str):
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
    converts a string to a list of objects
    input_str: str
        Must be a string of the form 'i = 4; p = [0, 1, 1, 2, 5];'
    returns: list
    """
    items = [x.strip() for x in input_str.split(";")]
    objs = dict()
    for item in items:
        try:
            key, value_str = item.split("=")
            key = key.strip()
        except:
            continue # no key found
        try:
            value_str = value_str.strip()

            # quick and dirty object conversion to make all literals hashable
            # TODO: create a hashable class for dicts and lists
            literal_value = literal_eval(value_str) # safe eval
            value = make_hashable(literal_value)
            objs[key] = value
        except ValueError:
            objs[key] = "NONLITERAL_STRING"
    return objs

def make_hashable(obj):
    if isinstance(obj, dict):
        # Convert dict to a sorted tuple of key-value pairs, making keys/values hashable recursively
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    if isinstance(obj, set):
        return frozenset(obj)
    if isinstance(obj, list):
        return tuple(obj)
    # recursion case
    elif isinstance(obj, list):
        # Convert lists to tuples
        return tuple(make_hashable(item) for item in obj)
    else:
        # Assume the object is hashable (e.g., numbers, strings, tuples)
        return obj
    
def custom_metrics_slp(preds):
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
        torch.arange(batch_size * (seq_length-1)), shift_labels.view(-1)
    ].view(batch_size, (seq_length-1))

    nll = -torch.log(p_true_tokens)
    mean_nll = nll.mean()
    ppl = torch.exp(mean_nll)

    # compute percentage of correct tokens
    correct_tokens = (shift_logits.view(-1, vocab_size).argmax(-1) == shift_labels.view(-1)).float().mean()

    pred_max_labels = shift_logits.argmax(-1).view(batch_size, -1)
    f1s = []
    for i in range(batch_size):
        unmasked_label_tokens = shift_labels[i][shift_labels[i] != -100][:-1] # drop eos_token
        # find the index where the instruction token ends and the answer begins
        inst_token_seq = tokenizer.encode("[/INST]", return_tensors="pt")[0][1:]
        first_output_idx = None
        for j in range(unmasked_label_tokens.shape[0] - len(inst_token_seq)):
            if torch.equal(unmasked_label_tokens[j:j+len(inst_token_seq)], inst_token_seq):
                first_output_idx = j + len(inst_token_seq) 
                break
        assert first_output_idx is not None, "Could not find the end of the instruction token"

        # get ground truth output tokens
        gt_output_tokens = unmasked_label_tokens[first_output_idx:]
        # get predicted output tokens (including padding)
        pred_output_tokens_masked = pred_max_labels[i][first_output_idx:]
        # drop the pad tokens 
        pred_output_tokens_unmasked = pred_output_tokens_masked[pred_output_tokens_masked != -100]
        first_pred_output_stop_idx = torch.where(pred_output_tokens_unmasked == tokenizer.eos_token_id)[0][0]
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
    wandb.log({"perplexity": ppl.item(), "correct_tokens": correct_tokens.item(), "f1": f1_mean})
    return {"perplexity": ppl, "correct_tokens": correct_tokens.item(), "f1": f1_mean}
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
    ppl = nll.exp().mean()

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

        correct = correct_solution(pred_output_str, gt_output_str)
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
            "perplexity": ppl.item(),
            "correct_tokens": correct_tokens.item(),
            "f1": mean_f1,
            "solve_rate": solve_rate,
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

# TRAINING ONLY RUN TO SAVE MODEL FOR LATER TESTING
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset_slp,
    eval_dataset=eval_dataset_gsm8k,
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
        learning_rate=1e-3,
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
        eval_steps=10,
        # eval_steps=1,
        do_eval=True,
        eval_accumulation_steps=50,
    ),
)

trainer.compute_metrics = custom_metrics_gsm8k
trainer.train()
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
wandb.finish()
