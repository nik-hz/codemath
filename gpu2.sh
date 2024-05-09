#!/bin/bash
echo "Training CodeLlama on both TRACED and PST on gpu1"
# training for Llama2 chat
COMMAND1="python train.py -m 7bCodeU -d datasets/Training_Trace_Dataset.json -u -traced -o models/codellama7b_traced"
COMMAND2="python train.py -m 7bCodeU -d datasets/python_states_singleline.json -u -pst -o models/codellama7b_pst"

# Start the first training script on GPU 0
CUDA_VISIBLE_DEVICES=0 $COMMAND1

wait 
# Start the second training script on GPU 1
CUDA_VISIBLE_DEVICES=0 $COMMAND2

# Wait for both training scripts to finish
wait

echo "Both training processes have completed."

# #!/bin/bash
# echo "Training CodeLlama on both TRACED and PST on gpu1"
# # training for Llama2 chat
# COMMAND1="python train.py -m Mistral7bUInstruct -d datasets/Training_Trace_Dataset.json -u -traced -o models/mistral_instr_traced"
# COMMAND2="python train.py -m Mistral7bUInstruct -d datasets/python_states_singleline.json -u -pst -o models/mistral_instr_pst"

# # Start the first training script on GPU 0
# CUDA_VISIBLE_DEVICES=0 $COMMAND1

# wait 
# # Start the second training script on GPU 1
# CUDA_VISIBLE_DEVICES=0 $COMMAND2

# # Wait for both training scripts to finish
# wait

# echo "Both training processes have completed."
