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

# custom imports
from prompts import TRACE_PROMPT


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


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=512
    )


def main(args):
    output_path, data_path, model, unsloth = (
        args.output,
        args.data,
        args.model,
        args.unsloth,
    )

    models = {
        "13bM": "meta-llama/Llama-2-13b-chat-hf",
        "7bM": "meta-llama/Llama-2-7b-chat-hf",
        "7bU": "unsloth/llama-2-7b-chat-bnb-4bit",
    }

    wandb.init(
        project="codemath",
        config={
            "dataset": f"{data_path}",
        },
        name=models[model],
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=models[model],
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
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

    ## Load Dataset and train ##
    dataset_traced = load_dataset("json", data_files=data_path, split="train")
    dataset_traced = dataset_traced.map(
        format_trace_data, batched=True
    ).train_test_split(train_size=0.1)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_traced["train"],
        # eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,  # Packing setting
        args=TrainingArguments(
            report_to="wandb",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=1,
            learning_rate=2e-4,
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

    # Train model
    trainer.train()


if __name__ == "__main__":
    # python train.py -o models -m 7bU -u -d datasets/Training\ Trace\ Dataset.json
    parser = argparse.ArgumentParser(prog="training")
    parser.add_argument("-m", "--model")
    parser.add_argument("-d", "--data")
    parser.add_argument("-o", "--output")
    parser.add_argument("-u", "--unsloth", action="store_true")
    args = parser.parse_args()

    main(args)

    # peft_config = LoraConfig(
    #     task_type=TaskType.CAUSAL_LM,
    #     inference_mode=False,
    #     r=8,
    #     lora_alpha=32,
    #     lora_dropout=0.1,
    # )

    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf")
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()
