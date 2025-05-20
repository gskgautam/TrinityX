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
        super(ExpertFFN, self).__init__()
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
    if paths["train_data"].endswith(".json"):
        train_data = pd.read_json(paths["train_data"])
        test_data = pd.read_json(paths["test_data"])
    else:
        train_data = pd.read_csv(paths["train_data"])
        test_data = pd.read_csv(paths["test_data"])
    text_columns = train_data.select_dtypes(include=['object']).columns
    if not text_columns.empty:
        train_data = pd.DataFrame(text_to_numeric(train_data[text_columns[0]]))
        test_data = pd.DataFrame(text_to_numeric(test_data[text_columns[0]]))
    train_data = train_data.select_dtypes(include=['number'])
    test_data = test_data.select_dtypes(include=['number'])
    return train_data, test_data

# Load all experts and initialize FFN models
experts = {}
for name in expert_configs.keys():
    train_data, test_data = load_data(name)
    input_dim = train_data.shape[1] if train_data.shape[1] > 0 else 1
    experts[name] = {
        "ffn": ExpertFFN(input_dim=input_dim, hidden_dim=128, output_dim=64),
        "train_data": train_data,
        "test_data": test_data
    }

# Temperature-scaled softmax for gating
def temperature_scaled_softmax(gamma_values, temperature=0.7):
    gamma_tensor = torch.tensor(list(gamma_values.values()), dtype=torch.float32)
    scaled_softmax = F.softmax(gamma_tensor / temperature, dim=0)
    return {k: v.item() for k, v in zip(gamma_values.keys(), scaled_softmax)}

# Entropy regularizer
def entropy_regularization(probabilities):
    return -torch.sum(probabilities * torch.log(probabilities + 1e-8))

# KL divergence penalty
def kl_divergence(p, q, epsilon=1e-8):
    p = torch.clamp(p, min=epsilon)
    q = torch.clamp(q, min=epsilon)
    return torch.sum(p * torch.log(p / q))

# Update gamma based on losses
def update_gamma_values(gamma_values, expert_losses, scaling_factor=0.1):
    updated = {}
    total_loss = sum(expert_losses.values())
    for expert, loss in expert_losses.items():
        updated[expert] = gamma_values[expert] * (total_loss / (loss + 1e-8)) * scaling_factor
    s = sum(updated.values())
    return {k: v/s for k, v in updated.items()}

# Router class to manage expert selection using MoCaE + Gating Loss + Regularization Loss (RL)
class MoCaERouterWithPenalties(nn.Module):
    def __init__(self, expert_ffns, gamma_values, previous_gamma_values=None, temperature=0.7):
        super().__init__()
        self.expert_ffns = expert_ffns
        self.gamma_values = gamma_values
        self.previous_gamma_values = previous_gamma_values or gamma_values
        self.temperature = temperature

    def forward(self, x):
        # 1. Temperature-scaled gating
        gamma_scaled = temperature_scaled_softmax(self.gamma_values, self.temperature)

        # 2. Compute weighted expert outputs
        expert_outputs = {
            expert: ffn(x) * gamma_scaled[expert]
            for expert, ffn in self.expert_ffns.items()
        }
        weighted_sum = sum(expert_outputs.values())

        # 3. Entropy regularization
        entropy_reg = entropy_regularization(
            torch.tensor(list(gamma_scaled.values()), dtype=torch.float32)
        )

        # 4. KL divergence penalty to prevent sharp gamma jumps
        kl_penalty = kl_divergence(
            torch.tensor(list(gamma_scaled.values()), dtype=torch.float32),
            torch.tensor(list(self.previous_gamma_values.values()), dtype=torch.float32)
        )

        # 5. Gating loss: KL(gamma_scaled || uniform)
        num_experts = len(gamma_scaled)
        uniform = torch.full((num_experts,), 1.0 / num_experts)
        gating_loss = kl_divergence(
            torch.tensor(list(gamma_scaled.values()), dtype=torch.float32),
            uniform
        )

        # 6. Regularization loss (RL): KL(gamma_scaled || gamma_prior)
        gamma_prior = torch.full((num_experts,), 1.0 / num_experts)  # assuming uniform prior
        rl_loss = kl_divergence(
            torch.tensor(list(gamma_scaled.values()), dtype=torch.float32),
            gamma_prior
        )

        # 7. Update previous gammas
        self.previous_gamma_values = self.gamma_values

        # 8. Total loss
        total_loss = (
            torch.mean(weighted_sum)
            + 0.1 * entropy_reg
            + 0.01 * kl_penalty
            + 0.05 * gating_loss
            + 0.05 * rl_loss  # Added RL loss here
        )

        # 9. Update gamma values based on expert performance
        expert_losses = {expert: total_loss.item() for expert in self.expert_ffns.keys()}
        self.gamma_values = update_gamma_values(self.gamma_values, expert_losses)

        return total_loss, weighted_sum, entropy_reg, kl_penalty

# Process input through router
def process_input_data_with_penalties():
    for expert, vals in experts.items():
        data = vals["train_data"]
        if data.empty:
            print(f"Skipping {expert}: No numeric data found!")
            continue
        tensor = torch.tensor(data.values, dtype=torch.float32)
        total_loss, weighted_sum, entropy_reg, kl_pen = router_with_penalties(tensor)
        print(f"Processed {expert} - Loss:{total_loss.item()} Entropy:{entropy_reg.item()} KL:{kl_pen.item()}")

# Save aggregated embeddings
def save_aggregated_output_embeddings():
    aggregated = {}
    for expert, vals in experts.items():
        data = vals["train_data"]
        if data.empty:
            print(f"Skipping {expert}: No numeric data!")
            continue
        tensor = torch.tensor(data.values, dtype=torch.float32)
        _, weighted_sum, _, _ = router_with_penalties(tensor)
        aggregated[expert] = weighted_sum.detach().cpu().numpy()
    outd = 'Path'
    os.makedirs(outd, exist_ok=True)
    np.save(os.path.join(outd, 'aggregated_embeddingsMoE_Gl_Rl.npy'), aggregated)
    print(f"Aggregated embeddings saved to {outd}/aggregated_embeddings.npy")

# Initialize and run
gamma_values = {n: 1.0 for n in experts.keys()}
router_with_penalties = MoCaERouterWithPenalties(expert_ffns, gamma_values)
process_input_data_with_penalties()
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


