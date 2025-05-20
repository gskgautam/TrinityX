

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
        "adapter_weights": "Path",
        "gamma": "Path",
        "base_weights": "Path",
        "train_data": "Path",
        "test_data": "Path"
    },
    "beavertails": {
        "adapter_weights": "Path",
        "gamma": "Path",
        "base_weights": "Path",
        "train_data": "Path",
        "test_data": "Path"
    },
    "truthfulqa": {
        "adapter_weights": "Path",
        "gamma": "Path",
        "base_weights": "Path",
        "train_data": "Path",
    "test_data":  "Path"
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
    output_dir = 'Path'
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'aggregated_embeddings.npy')
    np.save(output_file, aggregated_outputs)
    print(f"Aggregated embeddings saved to {output_file}")

# Call the function to save the aggregated embeddings
save_aggregated_output_embeddings()

# Load the aggregated embeddings
def check_aggregated_embeddings_shape(file_path):
    """Load the aggregated embeddings and print their shape."""
    # Load the embeddings from the saved .npy file
    aggregated_embeddings = np.load(file_path, allow_pickle=True).item()
    
    # Print the shape of each expert's aggregated embedding
    for expert, embedding in aggregated_embeddings.items():
        print(f"Shape of {expert}'s aggregated embedding: {embedding.shape}")

# Path to the saved aggregated embeddings file
aggregated_embeddings_file = 'Path'

# Check the shape of the aggregated embeddings
check_aggregated_embeddings_shape(aggregated_embeddings_file)
