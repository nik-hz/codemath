# #!/bin/bash

# models: llama2, llama2-chat, codellama
# datasets: (base), traced, slp

### Train the models ###

# # traced
python3 train.py -m 7bU -d datasets/Training_Trace_Dataset.json -u -traced -o models/llama_traced 
python3 train.py -m 7bUc -d datasets/Training_Trace_Dataset.json -u -traced -o models/llama_chat_traced 
python3 train.py -m 7bCodeU -d datasets/Training_Trace_Dataset.json -u -traced -o models/llama_code_traced 

# # slp
python3 train.py -m 7bU -d datasets/python_states_singleline.json -u -pst -o models/llama_slp 
python3 train.py -m 7bUc -d datasets/python_states_singleline.json -u -pst -o models/llama_chat_slp
python3 train.py -m 7bCodeU -d datasets/python_states_singleline.json -u -pst -o models/llama_code_slp

### Evaluate the models ###
# Results will be saved to WandB

# # base models
python3 eval_lm.py -b 7bU -o outputs/llama_base -z
python3 eval_lm.py -b 7bU -o outputs/llama_base 
python3 eval_lm.py -b 7bUc -o outputs/llama_chat_base  -z
python3 eval_lm.py -b 7bUc -o outputs/llama_chat_base 
python3 eval_lm.py -b 7bCodeU -o outputs/llama_code_base  -z
python3 eval_lm.py -b 7bCodeU -o outputs/llama_code_base 

# # traced models
python3 eval_lm.py -m models/llama_traced -o outputs/llama_traced -z
python3 eval_lm.py -m models/llama_traced -o outputs/llama_traced
python3 eval_lm.py -m models/llama_chat_traced -o outputs/llama_chat_traced -z
python3 eval_lm.py -m models/llama_chat_traced -o outputs/llama_chat_traced
python3 eval_lm.py -m models/llama_code_traced -o outputs/llama_code_traced -z
python3 eval_lm.py -m models/llama_code_traced -o outputs/llama_code_traced

# # slp models
python3 eval_lm.py -m models/llama_slp -o outputs/llama_slp -z
python3 eval_lm.py -m models/llama_slp -o outputs/llama_slp
python3 eval_lm.py -m models/llama_chat_slp -o outputs/llama_chat_slp -z
python3 eval_lm.py -m models/llama_chat_slp -o outputs/llama_chat_slp
python3 eval_lm.py -m models/llama_code_slp -o outputs/llama_code_slp -z
python3 eval_lm.py -m models/llama_code_slp -o outputs/llama_code_slp