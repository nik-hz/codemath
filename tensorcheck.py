import os
import hashlib
import safetensors

# Path to the directory containing all the models
models_dir = "models"


# Function to load safetensors and compute the hash of the tensor data
def compute_safetensor_hash(filepath):
    tensors = safetensors.safe_open(filepath, framework="pt")
    tensor_data_hash = hashlib.sha256()

    for key in tensors.keys():
        tensor = tensors.get_tensor(key)
        tensor_data_hash.update(tensor.numpy().tobytes())

    return tensor_data_hash.hexdigest()


# List to store the hashes and model names
model_hashes = {}

# Traverse through each model directory and calculate the hash of the weights
for model_name in os.listdir(models_dir):
    model_path = os.path.join(models_dir, model_name, "adapter_model.safetensors")

    if os.path.exists(model_path):
        model_hash = compute_safetensor_hash(model_path)
        print(f"Model: {model_name}, Hash: {model_hash}")

        if model_hash in model_hashes:
            print(
                f"Identical weights found between models: {model_name} and {model_hashes[model_hash]}"
            )
        else:
            model_hashes[model_hash] = model_name
    else:
        print(f"Warning: No 'adapter_model.safetensors' found for {model_name}")

# Summary of models with identical weights
if len(model_hashes) == len(os.listdir(models_dir)):
    print("No identical weights found across models.")
else:
    print("Some models have identical weights.")
