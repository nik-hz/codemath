#!/bin/bash
echo "Training Llama2 chat on both TRACED and PST on gpu1"
# training for Llama2 chat
COMMAND1="python train.py -m 7bUc -d datasets/Training_Trace_Dataset.json -u -traced -o models/llama7b_chat_traced"
COMMAND2="python train.py -m 7bUc -d datasets/python_states_singleline.json -u -pst -o models/llama7b_chat_pst"

# Start the first training script on GPU 0
CUDA_VISIBLE_DEVICES=1 $COMMAND1

wait 
# Start the second training script on GPU 1
CUDA_VISIBLE_DEVICES=1 $COMMAND2

# Wait for both training scripts to finish
wait

echo "Both training processes have completed."
