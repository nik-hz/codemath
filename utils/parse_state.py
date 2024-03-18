"""
For single line python dataset
Converts to json for parsing.
Useful idk but might be good for formatting unity across our other datasets as well
"""

import json


def process_file(input_file_path, output_file_path):
    """some formatting issues in kaggle dataset. this fixes it."""
    with open(input_file_path, "r") as file:
        lines = file.readlines()

    processed_lines = []
    for i, line in enumerate(lines):
        if not line.startswith("state:") and i > 0:
            processed_lines[-1] = processed_lines[-1].strip() + " "
        processed_lines.append(line)

    with open(output_file_path, "w") as file:
        file.writelines(processed_lines)


def convert_txt_to_json(txt_file_path, json_file_path):
    with open(txt_file_path, "r") as txt_file:
        lines = txt_file.readlines()
        data = []
        example = 0
        for line in lines:
            # print(line.split("; output:"))
            # return
            state_code, output = line.split("; output:")
            output = output.strip()
            input_text = state_code.strip()
            data.append(
                {"input": input_text, "output": output, "example": example},
            )
            example += 1

    with open(json_file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


# Specify the paths for your TXT and JSON files
# TODO convert this to absolute filepaths if necessary
txt_file_path = "../datasets/new_all_states.txt"
txt_file_path2 = "../datasets/new_all_states_clean.txt"
json_file_path = "../datasets/python_states_singleline.json"

process_file(txt_file_path, txt_file_path2)
convert_txt_to_json(txt_file_path2, json_file_path)
