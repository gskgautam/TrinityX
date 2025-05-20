import torch
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset

def load_dataset():
    """Load the Alpaca dataset."""
    try:
        alpaca_train = pd.read_json("Path")
        alpaca_test = pd.read_json("Path")
        print("Dataset loaded successfully.")
        return alpaca_train, alpaca_test
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def tokenize_data(tokenizer, dataset, is_train=True, device='cpu'):
    """Tokenize input and add labels as the same input for supervised learning."""
    column1, column2, label_column = ('instruction', 'input', 'output') if is_train else ('instruction', 'output', 'output')

    tokenized = dataset.apply(lambda row: tokenizer(f"{row[column1]} {row[column2]}", truncation=True, padding='max_length', max_length=512), axis=1)

    tokenized_dict = {
        "input_ids": [x['input_ids'] for x in tokenized],
        "attention_mask": [x['attention_mask'] for x in tokenized],
        "labels": [x['input_ids'] for x in tokenized]  # Labels are the same as input_ids for CausalLM
    }

    # Move the data to GPU or CPU depending on the device
    tokenized_dict["input_ids"] = [torch.tensor(x).to(device) for x in tokenized_dict["input_ids"]]
    tokenized_dict["attention_mask"] = [torch.tensor(x).to(device) for x in tokenized_dict["attention_mask"]]
    tokenized_dict["labels"] = [torch.tensor(x).to(device) for x in tokenized_dict["labels"]]

    return Dataset.from_dict(tokenized_dict)

def print_trainable_parameters(model):
    """Print trainable parameters in the model."""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable parameters: {trainable_params} / {all_params} ({100 * trainable_params / all_params:.2f}%)")

def train_lora_adapter(base_model, train_data, eval_data, adapter_name, device):
    """Train a LoRA adapter on the base model with debugging logs."""
    lora_config = LoraConfig(r=64, lora_alpha=32, lora_dropout=0.1)
    model = get_peft_model(base_model, lora_config).to(device)

    print_trainable_parameters(model)

    training_args = TrainingArguments(
        output_dir=f"./results/{adapter_name}",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        learning_rate=5e-4,  # Explicit learning rate
        logging_dir=f"./logs/{adapter_name}",
        save_steps=500,
        evaluation_strategy="epoch",
        logging_steps=10,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
    )

    trainer.train()

    print("Checking if adapter layers have been updated...")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable: {name}, Norm: {torch.norm(param).item()}")

    model.save_pretrained(f"./results/{adapter_name}")
    return model

def compute_task_vector(base_model, adapter_model, device):
    """Compute the task vector for a given adapter model with detailed debugging."""
    task_vector = {}

    base_state_dict = base_model.state_dict()
    adapter_state_dict = adapter_model.state_dict()

    print("Debug: Checking parameter differences between base and adapter models...")

    nonzero_count = 0  # Track number of parameters with actual differences

    for k in adapter_state_dict.keys():
        if k in base_state_dict:
            diff = adapter_state_dict[k] - base_state_dict[k]
            norm_diff = torch.norm(diff).item()

            if norm_diff > 0:
                task_vector[k] = diff
                nonzero_count += 1
                print(f"Layer '{k}' has a nonzero difference (Norm: {norm_diff:.6f})")
            else:
                print(f"Layer '{k}' has no difference (Norm: {norm_diff:.6f}) - Check if training updated parameters!")
        else:
            print(f"Skipping '{k}': Not in base model (Possibly a LoRA-specific layer).")

    if not task_vector:
        print("Task vector is empty! Ensure adapter model was trained correctly.")
        print("Debugging steps:")
        print("1. Verify that training loss decreased during training.")
        print("2. Check that LoRA layers are updating using print_trainable_parameters().")
        print("3. Inspect saved adapter weights to confirm changes.")
        raise ValueError("🚨 Task vector is empty. Ensure adapter model was trained correctly.")

    print(f"Computed task vector with {nonzero_count} nonzero layers.")
    torch.save(task_vector, "task_vector.pt")  # Save task vector to disk
    print("Task vector saved successfully.")
    return task_vector

def compute_gamma_value(task_vector):
    """Compute a single gamma value based on the task vector norm."""
    norm = torch.norm(torch.cat([v.flatten() for v in task_vector.values()]))
    gamma_value = 1.0  # Since only one adapter is used, it fully contributes to the final model.

    with open("gamma_value.json", "w") as f:
        json.dump({"gamma": gamma_value}, f)
    print(f"Gamma value {gamma_value} saved successfully.")

    return gamma_value

def apply_task_vector(base_model, task_vector, gamma_value, device):
    """Apply a weighted sum of the task vector to the base model."""
    updated_model_state = base_model.state_dict()
    for k in task_vector.keys():
        updated_model_state[k] += gamma_value * task_vector[k]

    base_model.load_state_dict(updated_model_state)
    return base_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose GPU if available
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token  # Fix pad token to eos token
    base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(device)

    print("Base model loaded.")

    train_data, test_data = load_dataset()  # Load training and test data
    print("Dataset loaded.")

    tokenized_train_data = tokenize_data(tokenizer, train_data, is_train=True, device=device)  # Tokenize the data
    tokenized_test_data = tokenize_data(tokenizer, test_data, is_train=False, device=device)
    print("Data tokenized.")

    adapter_name = "alpaca_adapter"
    adapter_model = train_lora_adapter(base_model, tokenized_train_data, tokenized_test_data, adapter_name, device)
    print("Adapter model trained.")

    task_vector = compute_task_vector(base_model, adapter_model, device)  # Compute task vector
    gamma_value = compute_gamma_value(task_vector)  # Compute gamma value

    final_model = apply_task_vector(base_model, task_vector, gamma_value, device)  # Apply task vector to base model
    final_model.save_pretrained("./results/final_model")  # Save the final model
    print("Final model saved successfully.")

    return final_model

if __name__ == "__main__":
    final_model = main()
    print("Model training and adaptation completed successfully!")

"""To Check Weights"""

from safetensors.torch import load_file

# Path to the LoRA adapter weights
adapter_weights_path = "Path"

# Load the weights
weights = load_file(adapter_weights_path)

# List all weight tensors
print("Stored Weight Keys:", list(weights.keys()))

# Example: Print the first few values of a weight tensor
for key in weights:
    print(f"Layer: {key}\nWeights:\n{weights[key]}\n")
    break  # Remove this break to print all weights

"""Gamma Weight Check"""

import json

with open("Path", "r") as f:
    config = json.load(f)

print("Gamma:", config.get("lora_alpha", "Not found"))

"""Base Model Weights"""

import torch
from transformers import AutoModelForCausalLM

def save_base_model_weights(base_model, filename="base_model_weights.pth"):
    """
    Save the base model weights to a file.
    Args:
        base_model: The base model from which to save weights.
        filename: The path where to save the weights.
    """
    try:
        torch.save(base_model.state_dict(), filename)
        print(f"Base model weights saved to {filename}.")
    except Exception as e:
        print(f"Error saving base model weights: {e}")
        raise


def load_base_model_weights(base_model, filename="base_model_weights.pth"):
    """
    Load the base model weights from a file.
    Args:
        base_model: The base model to load weights into.
        filename: The path from which to load the weights.
    """
    try:
        base_model.load_state_dict(torch.load(filename))
        print(f"Base model weights loaded from {filename}.")
    except Exception as e:
        print(f"Error loading base model weights: {e}")
        raise


def main():
    # Load your base model
    base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

    # Save the base model weights
    save_base_model_weights(base_model)

    # Optionally, load them back to verify
    load_base_model_weights(base_model)


if __name__ == "__main__":
    main()

"""To Check Base Weights Alpaca"""

import torch

# Load the model's state_dict (weights)
model = torch.load('Path')

# Print the state_dict of the model
for name, param in model.items():
    print(f'{name} : {param.shape}')

