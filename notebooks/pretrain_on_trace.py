# Import necessary libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from ast import literal_eval
import wandb
import os


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
# os.environ["WANDB_PROJECT"]="codemath"
# os.environ["WANDB_LOG_MODEL"]="true"
# os.environ["WANDB_WATCH"]="false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


max_seq_length = 2048
# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b",
    max_seq_length=max_seq_length,
    dtype=None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit=True,  # Use 4bit quantization to reduce memory usage. Can be False
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
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
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
json_file_path = "./train.jsonl"
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

dataset = load_dataset("gsm8k", 'main', split='train')
# dataset = dataset['train']

split_ratio = 0.1 
split_datasets = dataset.train_test_split(test_size=split_ratio)

train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]

# Example of manually selecting two examples for training and one for evaluation
train_dataset = dataset.select([0, 20])  # Select the first two examples for training
eval_dataset = dataset.select([2])      # Select the third example for evaluation


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
        # report_to='wandb',
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=5,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        evaluation_strategy="steps",
        do_eval=True,
        eval_accumulation_steps=1,
    ),
)


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



# Start training
trainer.compute_metrics = custom_metrics
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
