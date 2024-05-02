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
        args.model.args.unsloth,
    )

    models = {
        "13bM": "meta-llama/Llama-2-13b-chat-hf",
        "7bM": "meta-llama/Llama-2-7b-chat-hf",
        "7bU": "meta-llama/Llama-2-7b-chat-hf",
    }
    
    if unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
            max_seq_length=2048,
            dtype=None,  
            load_in_4bit=False,  
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
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        tokenizer = AutoTokenizer.from_pretrained(models[model])
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(models[model])
        model = model.to(dtype=torch.bfloat16)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    ## Load Dataset and train ##
    dataset_traced = load_dataset("json", data_files=data_path, split="train").select(
        range(1000)
    )
    dataset_traced = dataset_traced.map(format_trace_data, batched=True)
    dataset_traced = dataset_traced.map(
        tokenize_function,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        remove_columns=dataset_traced.column_names,
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        # bf16=torch.cuda.is_bf16_supported(),
        seed=3407,
        remove_unused_columns=False,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_traced,  # ["train"],
        args=training_args,
    )

    # Train model
    trainer.train()


if __name__ == "__main__":
    # python train.py -o models -m 7bU -d datasets/Training\ Trace\ Dataset.json
    parser = argparse.ArgumentParser(prog="training")
    parser.add_argument("-m", "--model")
    parser.add_argument("-d", "--data")
    parser.add_argument("-o", "--output")
    parser.add_argument("-u", "--unsloth" action="store_true")
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
