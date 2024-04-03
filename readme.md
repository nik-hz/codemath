# Codemath project

# Quickstart (For Grading)

Setup to add in missing dirs
```
mkdir notebooks/model_save_path
mkdir notebooks/outputs
```

After downloading data from Kaggle (below), train the model with

```
python3 notebooks/mistral_codemath_4bit.py
```

## First test

-   Set as seq2seq task. Given state and code the model should predict the output. I will run the tests on mistral in ft1 using this approach

## Datasets

-   For now, we are using the python state changes from here: https://www.kaggle.com/datasets/frasergreenlee/python-state-changes

## Setup

Download the datasets and put them in the dataset/ dir in the root of the project. You can then run the code in utils to set up the json version.

## General Notes

-   The unsloth notebook seems to work on colab. You need a kaggle account to download the dataset with the first block of code. Otherwise, the prompt needs some work and more testing needs to be done. No evaluations yet.
