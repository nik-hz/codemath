import os
import argparse

import wandb
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from datasets import load_dataset

from gsm8k_eval import (
    prepare_input,
    decode_output,
    extract_number_from_text,
    extract_response_after_question,
    formatting_prompts_func,
    PROMPT,
    PREAMBLE,
    TRAIN_TEMPLATE,
    EVAL_TEMPLATE,
)

parser = argparse.ArgumentParser(
    prog="codemath comparator",
    description="compare all of our codemath models",
)

parser.add_argument("-n", "--name")  # model name
parser.add_argument("-t", "--train", action="store_true")  # train yes or no

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.


# os.environ["WANDB_MODE"] = "offline"


def load_model(model_name):
    to_load = ""
    if model_name == "codellama":
        to_load = "unsloth/codellama-7b-bnb-4bit"
    elif model_name == "llama2":
        to_load = "unsloth/mistral-7b-bnb-4bit"
    elif model_name == "mistral":
        to_load = "unsloth/mistral-7b-bnb-4bit"
    return to_load


if __name__ == "__main__":

    args = parser.parse_args()

    model_name, pretrain = args.name, args.train

    model_save_path = f"second_round_evals/{model_name}"

    params = {
        "model_name": model_name,
        "with training": True,
        "dataset": "single_line",
        "extra notes": "training single line + gsm8k train set",
    }

    wandb.init(
        project="Codemath",
        config={
            "Dataset": "gsm8k",
        },
        name=f"{model_name}_{pretrain}",
    )

    model_hf = load_model(model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_hf,  # Choose ANY! eg mistralai/Mistral-7B-Instruct-v0.2
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
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
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    gsm8k_test = load_dataset("gsm8k", "main", split="test")

    model.eval()

    all_correct = 0
    all_responses = {}
    idx = 0
    total = len(gsm8k_test)
    # total = 100

    if pretrain:
        dataset = load_dataset("gsm8k", "main", split="train")
        dataset = dataset.map(formatting_prompts_func, batched=True)
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
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
                output_dir=f"outputs/{model_name}",
            ),
        )
        trainer.train()

    #### ALWAYS RUN EVAL ####

    for task_id, problem in enumerate(gsm8k_test, start=1):
        if idx == total:
            break

        # Prepare the input for the model
        input_ids = prepare_input(problem["question"], tokenizer)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, max_new_tokens=120
            )  # Adjust max_length as needed

        response = decode_output(output_ids[0], tokenizer)
        all_responses[task_id] = response

        answer_line = extract_response_after_question(response, problem["question"])

        # Compare model output to the ground truth
        model_number = extract_number_from_text(answer_line, "The answer is")
        ground_truth_number = extract_number_from_text(problem["answer"], "####")

        # print(model_number)
        # print(ground_truth_number)

        if model_number == ground_truth_number:
            all_correct += 1

        print(f"Model answer: {model_number}")
        print(f"Ground truth answer: {ground_truth_number}")
        print(f"Correct: {all_correct} out of {total}")
        print("=" * 40)

        wandb.log(
            {
                "total_correct": all_correct,
                "current_pct_correct": all_correct / task_id,
            }
        )
        idx += 1

    accuracy = all_correct / total * 100
    print(f"Final Accuracy Mistral: {accuracy:.2f}%")

    model.save_pretrained(model_save_path)
