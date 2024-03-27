# Import necessary libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer

# Mistral 7B Finetune on Python Single Line Dataset
# print(torch.version.cuda)

# print("Available CUDA devices:", torch.cuda.device_count())
# for i in range(torch.cuda.device_count()):
#     print(f"Device {i}: {torch.cuda.get_device_name(i)} with {torch.cuda.mem_get_info(i)[1]} total memory")

# # Setup
# torch.cuda.set_device(1)  # Explicitly setting the device to GPU 0

max_seq_length = 2048
# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b",
    max_seq_length=max_seq_length,
    dtype=None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit=True,  # Use 4bit quantization to reduce memory usage. Can be False
)

# Configure PEFT for the model
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
json_file_path = "./python_states_singleline.json"
trace_prompt = """<s>[INST] Below is an input which contains the state of variables and code that acts upon these variables or not. Given the state and the code give the state after the code executes for each variable. Be very careful. You should clearly outline your intermediate steps and your final answer should be a newline with exactly the variables and their values. Here is the State and Code. {}
Now generate the final state for each variable. Generate intermediate outputs.[/INST] {}</s>"""

def formatting_prompts_func(examples):
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for input, output in zip(inputs, outputs):
        text = trace_prompt.format(input, output)
        texts.append(text)
    return {"text": texts}

dataset = load_dataset("json", data_files=json_file_path, split="train")

# Subsetting to only 10% of the dataset
dataset = dataset.shuffle(seed=42)  # Ensure a random subset is selected
subset_size = int(0.05 * len(dataset))  # Calculate 5% of the dataset size
dataset = dataset.select(range(subset_size))  # Select the first 5% of the dataset

dataset = dataset.train_test_split(test_size=0.1)["test"]  # Use test split for demonstration
dataset_single_line = dataset.map(formatting_prompts_func, batched=True)

# Setup Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset_single_line,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Packing setting
    args=TrainingArguments(
        per_device_train_batch_size=45,
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
    ),
)

# Start training
trainer_stats = trainer.train()

# Save the model and tokenizer after training
model_save_path = "model_save_path/mistral_7b_finetuned_trace_python"
tokenizer_save_path = "tokenizer_save_path/mistral_7b_finetuned"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)

## Finetune on gsm8k now
