# Import necessary libraries
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

# Mistral 7B Finetune on Python Single Line Dataset
# print(torch.version.cuda)

# print("Available CUDA devices:", torch.cuda.device_count())
# for i in range(torch.cuda.device_count()):
#     print(f"Device {i}: {torch.cuda.get_device_name(i)} with {torch.cuda.mem_get_info(i)[1]} total memory")

# # Setup
# torch.cuda.set_device(1)  # Explicitly setting the device to GPU 0

# SET CUDA DEVICE BY SETTING TMUX ENV VARS
# export CUDA_VISIBLE_DEVICES=0

# INIT WANDB
os.environ["WANDB_PROJECT"] = "codemath"
os.environ["WANDB_LOG_MODEL"] = "true"
os.environ["WANDB_WATCH"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


max_seq_length = 2048
# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b",
    max_seq_length=max_seq_length,
    dtype=None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit=True,  # Use 4bit quantization to reduce memory usage. Can be False
    device_map={"": 0},
)

# print("loading pretrained model and tokenizer")
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name="model_save_path/mistral_7b_finetuned_trace_python_with_mtoleseval",
#     max_seq_length=max_seq_length,
#     dtype=None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
#     load_in_4bit=True,  # Use 4bit quantization to reduce memory usage. Can be Fals
# )

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# model.to(device)
# Configure PEFT for the model
# TODO can comment this out for finetune a pretrin model
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


# Prepare the dataset
json_file_path = "./train.jsonl" # gsm8k dataset
trace_prompt = """<s>[INST] {} [/INST] {}</s>"""


def formatting_prompts_func(examples):
    inputs = examples["question"]
    outputs = examples["answer"]
    texts = []
    for input, output in zip(inputs, outputs):
        text = trace_prompt.format(input, output)
        texts.append(text)
    return {"text": texts}


# dataset = load_dataset("json", data_files=json_file_path, split="train")

dataset = load_dataset("gsm8k", "main", split="train")
# dataset = dataset['train']

split_ratio = 0.05
split_datasets = dataset.train_test_split(test_size=split_ratio)

train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]

# Example of manually selecting two examples for training and one for evaluation
# train_dataset = dataset.select([0, 20])  # Select the first two examples for training
# eval_dataset = dataset.select([2])  # Select the third example for evaluation


# subset_size = int(0.05 * len(dataset))  # Calculate 5% of the dataset size
# dataset = dataset.select(range(subset_size))  # Select the first 5% of the dataset

# dataset = dataset.train_test_split(test_size=1)["test"]  # Use test split for demonstration
train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

# Setup Trainer
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
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=5,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="constant",  # don't want to decay this for now
        # lr_scheduler_type="linear", # don't want to decay this for now
        seed=3407,
        output_dir="outputs",
        evaluation_strategy="steps",
        eval_steps=5,
        do_eval=True,
        # eval_accumulation_steps=50,
    ),
)


# TODO can cut this out, we need to compare token for token


# TODO add a function which compares the final output should be denoted by ####
# newlines represent one logical step, so they are important as well
def calculate_token_level_f1(prediction_tokens, reference_tokens):
    """
    Calculate precision, recall, and F1 score based on token overlap.
    """

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
    ppl = torch.exp(mean_nll)  # perplexity

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

        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

    mean_f1 = np.mean(f1_scores) if f1_scores else 0
    mean_precision = np.mean(precision_scores) if precision_scores else 0
    mean_recall = np.mean(recall_scores) if recall_scores else 0

    wandb.log(
        {
            "perplexity": ppl.item(),
            "correct_tokens": correct_tokens.item(),
            "f1": mean_f1,
        }
    )
    return {
        "perplexity": ppl,
        "correct_tokens": correct_tokens.item(),
        "f1": mean_f1,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
    }


# Start training
trainer.compute_metrics = custom_metrics_gsm8k
trainer_stats = trainer.train()
wandb.finish()

# Save the model and tokenizer after training
model_save_path = "model_save_path/mistral_7b_finetuned_gsm8k_with_eval"
# model_save_path = "model_save_path/mistral_7b_finetuned_gsm8k_pretrain"
# model_save_path = "model_save_path/mistral_7b_finetuned_trace_python_with_mtoleseval"
# tokenizer_save_path = "model_save_path/mistral_7b_finetuned_gsm8k_pretrain"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)


## Finetune on gsm8k now
