#!/usr/bin/env python
# coding: utf-8

# # W_MoE_Inference_Memory

# In[ ]:


import torch
import pandas as pd
import json
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from safetensors.torch import load_file
import numpy as np
import os

# Define paths for different expert datasets and models
expert_configs = {
    "alpaca": {
        "adapter_weights": "/kaggle/input/worksapce/worksapce/orkspace/LLaMa-2-7B-Alpaca/results/alpaca_adapter/adapter_model.safetensors",
        "gamma": "/kaggle/input/worksapce/worksapce/orkspace/LLaMa-2-7B-Alpaca/results/alpaca_adapter/adapter_config.json",
        "base_weights": "/kaggle/input/worksapce/worksapce/orkspace/LLaMa-2-7B-Alpaca/base_model_weights.pth",
        "train_data": "/kaggle/input/worksapce/workspace/orkspace/Dataset/Alpaca/Alpaca_Train.json",
        "test_data": "/kaggle/input/worksapce/workspace/orkspace/Dataset/Alpaca/Alpaca_Test.json"
    },
    "beavertails": {
        "adapter_weights": "/kaggle/input/worksapce/worksapce/orkspace/LLaMa-2-7B-BeaverTails/results/beavertails_adapter/adapter_model.safetensors",
        "gamma": "/kaggle/input/worksapce/worksapce/orkspace/LLaMa-2-7B-BeaverTails/results/beavertails_adapter/adapter_config.json",
        "base_weights": "/kaggle/input/worksapce/worksapce/orkspace/LLaMa-2-7B-BeaverTails/base_model_weights.pth",
        "train_data": "/kaggle/input/worksapce/workspace/orkspace/Dataset/BeaverTails/BeaverTails_Train.csv",
        "test_data": "/kaggle/input/worksapce/workspace/orkspace/Dataset/BeaverTails/BeaverTails_Test.csv"
    },
    "truthfulqa": {
        "adapter_weights": "/kaggle/input/worksapce/worksapce/orkspace/LLaMa-2-7b-TruthfulQA/results/truthfulqa_adapter/adapter_model.safetensors",
        "gamma": "/kaggle/input/worksapce/worksapce/orkspace/LLaMa-2-7b-TruthfulQA/results/truthfulqa_adapter/adapter_config.json",
        "base_weights": "/kaggle/input/worksapce/worksapce/orkspace/LLaMa-2-7b-TruthfulQA/base_model_weights.pth",
        "train_data": "/kaggle/input/worksapce/workspace/orkspace/Dataset/TruthfulQA/TruthfulQA_Train.csv",
    "test_data":  "/kaggle/input/worksapce/workspace/orkspace/Dataset/TruthfulQA/TruthfulQA_Test.csv"
    }
}

# Define Feed Forward Network (FFN) for each expert
class ExpertFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        return self.fc2(x)

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=500)

def text_to_numeric(text_series):
    """Convert text data into TF-IDF numerical vectors."""
    return vectorizer.fit_transform(text_series).toarray()

# Function to load dataset and ensure text is converted to numerical values
def load_data(expert_name):
    paths = expert_configs[expert_name]
    
    # Load dataset
    if paths["train_data"].endswith(".json"):
        train_data = pd.read_json(paths["train_data"])
        test_data = pd.read_json(paths["test_data"])
    else:
        train_data = pd.read_csv(paths["train_data"])
        test_data = pd.read_csv(paths["test_data"])

    # Identify text columns
    text_columns = train_data.select_dtypes(include=['object']).columns
    if not text_columns.empty:
        # Vectorize text data into numerical format
        train_data = pd.DataFrame(text_to_numeric(train_data[text_columns[0]]))  # Only using first text column for simplicity
        test_data = pd.DataFrame(text_to_numeric(test_data[text_columns[0]]))

    # Select only numeric columns
    train_data = train_data.select_dtypes(include=['number'])
    test_data = test_data.select_dtypes(include=['number'])

    return train_data, test_data

# Load all experts and initialize FFN models
experts = {}
for name in expert_configs.keys():
    train_data, test_data = load_data(name)
    input_dim = train_data.shape[1] if train_data.shape[1] > 0 else 1  # Handle edge cases
    experts[name] = {
        "ffn": ExpertFFN(input_dim=input_dim, hidden_dim=128, output_dim=64),
        "train_data": train_data,
        "test_data": test_data
    }

# Apply temperature scaling for gamma values
def temperature_scaled_softmax(gamma_values, temperature=0.7):
    """Apply temperature scaling to gamma values for smoother expert selection."""
    gamma_tensor = torch.tensor(list(gamma_values.values()), dtype=torch.float32)
    scaled_softmax = F.softmax(gamma_tensor / temperature, dim=0)
    return {k: v.item() for k, v in zip(gamma_values.keys(), scaled_softmax)}

# Function to calculate entropy regularization
def entropy_regularization(probabilities):
    """Compute the entropy of the expert probability distribution."""
    return -torch.sum(probabilities * torch.log(probabilities + 1e-8))

# Function to calculate KL divergence penalty
def kl_divergence(p, q, epsilon=1e-8):
    """Compute the KL Divergence between two probability distributions."""
    p = torch.clamp(p, min=epsilon)  # Avoid zero values
    q = torch.clamp(q, min=epsilon)  # Avoid zero values
    return torch.sum(p * torch.log(p / q))  # Remove redundant (p + epsilon) and (q + epsilon) inside log

def update_gamma_values(gamma_values, expert_losses, scaling_factor=0.1):
    """Update gamma values dynamically based on expert losses."""
    updated_gamma_values = {}
    total_loss = sum(expert_losses.values())
    
    for expert, loss in expert_losses.items():
        # Inverse of loss for scaling (lower loss means higher gamma value)
        updated_gamma_values[expert] = gamma_values[expert] * (total_loss / (loss + 1e-8)) * scaling_factor
        
    # Normalize to ensure the sum of gamma values is 1
    gamma_sum = sum(updated_gamma_values.values())
    normalized_gamma_values = {k: v / gamma_sum for k, v in updated_gamma_values.items()}
    
    return normalized_gamma_values


# Router class to manage expert selection using MoCaE
class MoCaERouterWithPenalties(nn.Module):
    def __init__(self, expert_ffns, gamma_values, previous_gamma_values=None, temperature=0.7):
        super().__init__()
        self.expert_ffns = expert_ffns
        self.gamma_values = gamma_values
        self.previous_gamma_values = previous_gamma_values or gamma_values  # Initialize previous gamma if not provided
        self.temperature = temperature

    def forward(self, x):
        # Apply temperature-scaled softmax to get the expert probabilities
        gamma_scaled = temperature_scaled_softmax(self.gamma_values, self.temperature)

        # Calculate the weighted sum of expert outputs
        expert_outputs = {
            expert: ffn(x) * gamma_scaled[expert]
            for expert, ffn in self.expert_ffns.items()
        }
        
        weighted_sum = sum(expert_outputs.values())
        
        # Apply entropy regularization to encourage balanced expert selection
        entropy_reg = entropy_regularization(torch.tensor(list(gamma_scaled.values()), dtype=torch.float32))

        # Apply KL Divergence penalty to prevent large shifts in expert probabilities
        kl_penalty = kl_divergence(
            torch.tensor(list(gamma_scaled.values()), dtype=torch.float32),
            torch.tensor(list(self.previous_gamma_values.values()), dtype=torch.float32)
        )

        # Update previous gamma values for the next step
        self.previous_gamma_values = self.gamma_values

        # Combine outputs with penalties
        total_loss = torch.mean(weighted_sum) + 0.1 * entropy_reg + 0.01 * kl_penalty  # Use mean to get scalar

        # Dynamically update gamma values based on expert performance (losses)
        expert_losses = {expert: total_loss.item() for expert in self.expert_ffns.keys()}  # Now total_loss is scalar
        
        self.gamma_values = update_gamma_values(self.gamma_values, expert_losses)

        return total_loss, weighted_sum, entropy_reg, kl_penalty


# Initialize router with penalties
expert_ffns = {name: experts[name]["ffn"] for name in experts.keys()}
gamma_values = {name: 1.0 for name in experts.keys()}  # Placeholder gamma values
router_with_penalties = MoCaERouterWithPenalties(expert_ffns, gamma_values)


# Function to process input embeddings with penalties
def process_input_data_with_penalties():
    """Pass expert input embeddings through MoCaE router with penalties."""
    for expert, values in experts.items():
        input_data = values["train_data"]

        if input_data.empty:
            print(f"Skipping {expert}: No numeric data found!")
            continue

        # Convert to PyTorch tensor
        input_embeddings = torch.tensor(input_data.values, dtype=torch.float32)

        # Forward pass through the router with penalties
        total_loss, weighted_sum, entropy_reg, kl_penalty = router_with_penalties(input_embeddings)
        
        # Ensure total_loss, entropy_reg, and kl_penalty are scalars for printing
        total_loss_scalar = total_loss.sum().item() if total_loss.numel() > 1 else total_loss.item()
        entropy_reg_scalar = entropy_reg.sum().item() if entropy_reg.numel() > 1 else entropy_reg.item()
        kl_penalty_scalar = kl_penalty.sum().item() if kl_penalty.numel() > 1 else kl_penalty.item()

        print(f"Processed {expert} - Total Loss: {total_loss_scalar} - Entropy: {entropy_reg_scalar} - KL Penalty: {kl_penalty_scalar}")

# Run processing with penalties
process_input_data_with_penalties()


# Function to save aggregated output embeddings
def save_aggregated_output_embeddings():
    """Save aggregated output embeddings for evaluation."""
    aggregated_outputs = {}
    for expert, values in experts.items():
        input_data = values["train_data"]
        if input_data.empty:
            print(f"Skipping {expert}: No numeric data found!")
            continue

        input_embeddings = torch.tensor(input_data.values, dtype=torch.float32)

        # Forward pass through the router with penalties
        total_loss, weighted_sum, entropy_reg, kl_penalty = router_with_penalties(input_embeddings)

        # Aggregate the outputs (here, we will store the weighted sum of all experts' outputs)
        aggregated_outputs[expert] = weighted_sum.detach().cpu().numpy()

    # Save aggregated embeddings to disk
    output_dir = '/workspace/Dataset/aggregated_embeddings'
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'aggregated_embeddings.npy')
    np.save(output_file, aggregated_outputs)
    print(f"Aggregated embeddings saved to {output_file}")

# Call the function to save the aggregated embeddings
save_aggregated_output_embeddings()


# In[ ]:


import numpy as np

# Load the aggregated embeddings
def check_aggregated_embeddings_shape(file_path):
    """Load the aggregated embeddings and print their shape."""
    # Load the embeddings from the saved .npy file
    aggregated_embeddings = np.load(file_path, allow_pickle=True).item()
    
    # Print the shape of each expert's aggregated embedding
    for expert, embedding in aggregated_embeddings.items():
        print(f"Shape of {expert}'s aggregated embedding: {embedding.shape}")

# Path to the saved aggregated embeddings file
aggregated_embeddings_file = '/kaggle/input/worksapce/workspace/orkspace/Dataset/aggregated_embeddings/aggregated_embeddings.npy'

# Check the shape of the aggregated embeddings
check_aggregated_embeddings_shape(aggregated_embeddings_file)


# In[ ]:


import os
import time
import psutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file

# --------------------
# Config & Paths
# --------------------
expert_configs = {
    "alpaca": {
        "train_data": "/kaggle/input/worksapce/orkspace/Dataset/Alpaca/Alpaca_Train.json"
    },
    "beavertails": {
        "train_data": "/kaggle/input/worksapce/orkspace/Dataset/BeaverTails/BeaverTails_Train.csv"
    },
    "truthfulqa": {
        "train_data": "/kaggle/input/worksapce/orkspace/Dataset/TruthfulQA/TruthfulQA_Train.csv"
    },
    
}
EMBEDDINGS_FILE = '/workspace/Dataset/aggregated_embeddings/aggregated_embeddings.npy'

# --------------------
# Label Loader with Safe Fallback
# --------------------
def load_labels(expert_name, label_col='label'):
    path = expert_configs[expert_name]['train_data']
    if path is None:
        raise KeyError(f"No train_data path for {expert_name}")
    df = pd.read_json(path) if path.endswith('.json') else pd.read_csv(path)
    if label_col in df.columns:
        return df[label_col].values
    # Try inferring label column: integer dtype with few unique values
    int_cols = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c])]
    for c in int_cols:
        if df[c].nunique() < len(df) / 2:
            print(f"Info: Using inferred label column '{c}' for {expert_name}")
            return df[c].values
    # No suitable label found
    raise KeyError(f"No label column found for {expert_name} in {path}")

# --------------------
# Calibration & Scoring Metrics
# --------------------

def compute_ece(probs, labels, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    acc = (preds == labels).astype(float)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if mask.any():
            ece += abs(confidences[mask].mean() - acc[mask].mean()) * mask.sum() / len(labels)
    return ece


def compute_brier(probs, labels):
    N, C = probs.shape
    true_onehot = np.zeros_like(probs)
    true_onehot[np.arange(N), labels] = 1
    return np.mean(np.sum((probs - true_onehot)**2, axis=1))


def temperature_scale(probs, temperature=1.0):
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)

# --------------------
# Zero-Shot & Few-Shot Evaluation
# --------------------

def eval_zero_shot(embeddings, labels, temp=1.0):
    # Inference timing
    start = time.time()
    logits = torch.tensor(embeddings, dtype=torch.float32)
    probs = torch.softmax(logits, dim=1).numpy()
    infer_time = time.time() - start

    # Metrics
    ece = compute_ece(probs, labels)
    ece_t = compute_ece(temperature_scale(probs, temp), labels)
    brier = compute_brier(probs, labels)

    return {
        'ECE': round(ece,4),
        'ECE-t': round(ece_t,4),
        'Brier': round(brier,4),
        'Inference_Time_s': round(infer_time,4),
        'Train_Time_s': 0.0,
        'Train_Memory_MB': 0.0
    }


def eval_few_shot(embeddings, labels, epochs=5, temp=1.0, device='cpu'):
    X = torch.tensor(embeddings, dtype=torch.float32).to(device)
    y = torch.tensor(labels, dtype=torch.long).to(device)
    num_classes = len(np.unique(labels))
    model = nn.Linear(embeddings.shape[1], num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    # Training timing & memory
    t0 = time.time()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = crit(logits, y)
        loss.backward()
        optimizer.step()
    train_time = time.time() - t0
    train_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2

    # Inference
    t1 = time.time()
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    infer_time = time.time() - t1

    # Metrics
    ece = compute_ece(probs, labels)
    ece_t = compute_ece(temperature_scale(probs, temp), labels)
    brier = compute_brier(probs, labels)

    return {
        'ECE': round(ece,4),
        'ECE-t': round(ece_t,4),
        'Brier': round(brier,4),
        'Inference_Time_s': round(infer_time,4),
        'Train_Time_s': round(train_time,4),
        'Train_Memory_MB': round(train_mem,4)
    }

# --------------------
# Main Loop
# --------------------

if __name__ == '__main__':
    agg = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for expert, emb in agg.items():
        print(f"--- {expert.upper()} ---")
        try:
            labels = load_labels(expert)
        except KeyError as e:
            print(f"Skipping '{expert}': {e}")
            continue

        zero = eval_zero_shot(emb, labels)
        few = eval_few_shot(emb, labels, epochs=5, device=device)

        print("Zero-Shot Metrics:")
        for k,v in zero.items(): print(f"  {k}: {v}")
        print("Few-Shot Metrics:")
        for k,v in few.items(): print(f"  {k}: {v}")
        print()


# In[ ]:




