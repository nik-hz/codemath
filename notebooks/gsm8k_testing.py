import os

import wandb
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset

from gsm8k_eval import (
    prepare_input,
    decode_output,
    extract_number_from_text,
    extract_response_after_question,
    PROMPT,
    PREAMBLE,
    TRAIN_TEMPLATE,
    EVAL_TEMPLATE,
)

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.


os.environ["WANDB_MODE"] = "offline"


def traced():
    test = {
        "entry_variables": "3 3\n9 3 8\n4\n6\n5",
        "src_seq": '#include <stdio.h> \n int main ( void ) { \n int n , q , c [ 300 ] , qn ; \n int i , j = 1 , max [ 300 ] ; \n scanf ( "%d%d" , & n , & q ) ; \n for ( i = 0 ; i < n ; i ++ ) scanf ( "%d" , & c [ i ] ) ; \n for ( i = 0 ; i < q ; i ++ ) { \n scanf ( "%d" , & qn ) ; \n max [ i ] = c [ 0 ] % qn ; \n for ( j = 1 ; j < n ; j ++ ) { \n if ( c [ j ] % qn > max [ i ] ) \n max [ i ] = c [ j ] % qn ; \n } \n } \n for ( i = 0 ; i < q ; i ++ ) printf ( "%d" , max [ i ] ) ; \n return 0 ; \n }',
        "var_type_seq": "## ## ## ## ## ## ## ## ## ## ## basic_type ## basic_type ## array ## ## ## ## basic_type ## ## ## basic_type ## basic_type ## ## ## array ## ## ## ## ## ## ## ## ## ## basic_type ## ## basic_type ## ## ## ## ## basic_type ## ## ## basic_type ## basic_type ## basic_type ## ## ## ## ## ## ## array ## basic_type ## ## ## ## ## ## basic_type ## ## ## basic_type ## basic_type ## basic_type ## ## ## ## ## ## ## ## ## basic_type ## ## ## array ## basic_type ## ## array ## ## ## ## basic_type ## ## ## ## basic_type ## ## ## basic_type ## basic_type ## basic_type ## ## ## ## ## ## array ## basic_type ## ## basic_type ## array ## basic_type ## ## ## array ## basic_type ## ## array ## basic_type ## ## basic_type ## ## ## ## ## ## ## ## basic_type ## ## ## basic_type ## basic_type ## basic_type ## ## ## ## ## ## array ## basic_type ## ## ## ## ## ## ## ## ##",
        "value_type_seq": "## ## ## ## ## ## ## ## ## ## ## int ## int ## int ## ## ## ## int ## ## ## int ## int ## ## ## int ## ## ## ## ## ## ## ## ## ## int ## ## int ## ## ## ## ## int ## ## ## int ## int ## int ## ## ## ## ## ## ## int ## int ## ## ## ## ## ## int ## ## ## int ## int ## int ## ## ## ## ## ## ## ## ## int ## ## ## int ## int ## ## int ## ## ## ## int ## ## ## ## int ## ## ## int ## int ## int ## ## ## ## ## ## int ## int ## ## int ## int ## int ## ## ## int ## int ## ## int ## int ## ## int ## ## ## ## ## ## ## ## int ## ## ## int ## int ## int ## ## ## ## ## ## int ## int ## ## ## ## ## ## ## ## ##",
        "abstract_value_seq": "## ## ## ## ## ## ## ## ## ## ## NOT_REACHED ## NOT_REACHED ## NOT_REACHED ## ## ## ## NOT_REACHED ## ## ## UNKNOWN ## UNKNOWN ## ## ## UNKNOWN ## ## ## ## ## ## ## ## ## ## POSITIVE-VL ## ## NEGATIVE-VL ## ## ## ## ## ZERO ## ## ## ZERO ## POSITIVE-REG ## ZERO ## ## ## ## ## ## ## KNOWN ## ZERO ## ## ## ## ## ## POSITIVE-REG ## ## ## POSITIVE-REG ## POSITIVE-REG ## POSITIVE-REG ## ## ## ## ## ## ## ## ## ZERO ## ## ## KNOWN ## POSITIVE-REG ## ## KNOWN ## ## ## ## POSITIVE-REG ## ## ## ## POSITIVE-REG ## ## ## POSITIVE-REG ## POSITIVE-REG ## POSITIVE-REG ## ## ## ## ## ## KNOWN ## POSITIVE-REG ## ## POSITIVE-REG ## KNOWN ## POSITIVE-REG ## ## ## KNOWN ## POSITIVE-REG ## ## KNOWN ## POSITIVE-REG ## ## POSITIVE-REG ## ## ## ## ## ## ## ## POSITIVE-REG ## ## ## POSITIVE-REG ## POSITIVE-REG ## POSITIVE-REG ## ## ## ## ## ## KNOWN ## POSITIVE-REG ## ## ## ## ## ## ## ## ##",
    }

    # I = {[CLS],𝑒1,...,𝑒𝑖,[SEP],[SEP],𝑐1,...,𝑐𝑗,[SEP]}

    pass


def make_wandb(params):
    wandb.init(params)  # TODO make cli guide this


def load_model(model_name):
    to_load = ""
    if model_name == "codellama":
        to_load = "unsloth/codellama-7b-bnb-4bit"
    elif model_name == "llama2":
        to_load = "unsloth/mistral-7b-bnb-4bit"
    elif model_name == "mistral":
        to_load = "unsloth/mistral-7b-bnb-4bit"


if __name__ == "__main__"():
    model_name = "mistral"

    model_save_path = f"second_round_evals/{model_name}"

    params = {
        "model_name": model_name,
        "with training": True,
        "dataset": "single_line",
        "extra notes": "training single line + gsm8k train set",
    }
    make_wandb(params)

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
    total = 100

    for task_id, problem in enumerate(gsm8k_test):
        if idx == total:
            break

        print(f"task_id {task_id}")

        # Prepare the input for the model
        input_ids = prepare_input(problem["question"])
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, max_new_tokens=120
            )  # Adjust max_length as needed

        response = decode_output(output_ids[0])
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
            {"total_correct": all_correct, "current_pct_correct": all_correct / task_id}
        )
        idx += 1

    accuracy = all_correct / total * 100
    print(f"Final Accuracy Mistral: {accuracy:.2f}%")

    model.save_pretrained(model_save_path)
