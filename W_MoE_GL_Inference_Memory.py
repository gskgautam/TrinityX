#!/usr/bin/env python
# coding: utf-8

# # W_MoE_GL_Inference_Memory

# In[ ]:


import torch
import pandas as pd
import json
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os

# Define paths
expert_configs = {
    "alpaca": {
        "adapter_weights": "/kaggle/input/worksapce/orkspace/LLaMa-2-7B-Alpaca/results/alpaca_adapter/adapter_model.safetensors",
        "gamma":           "/kaggle/input/worksapce/orkspace/LLaMa-2-7B-Alpaca/results/alpaca_adapter/adapter_config.json",
        "base_weights":    "/kaggle/input/worksapce/orkspace/LLaMa-2-7B-Alpaca/base_model_weights.pth",
        "train_data":      "/kaggle/input/worksapce/orkspace/Dataset/Alpaca/Alpaca_Train.json",
        "test_data":       "/kaggle/input/worksapce/orkspace/Dataset/Alpaca/Alpaca_Test.json"
    },
    "beavertails": {
        "adapter_weights": "/kaggle/input/worksapce/orkspace/LLaMa-2-7B-BeaverTails/results/beavertails_adapter/adapter_model.safetensors",
        "gamma":           "/kaggle/input/worksapce/orkspace/LLaMa-2-7B-BeaverTails/results/beavertails_adapter/adapter_config.json",
        "base_weights":    "/kaggle/input/worksapce/orkspace/LLaMa-2-7B-BeaverTails/base_model_weights.pth",
        "train_data":      "/kaggle/input/worksapce/orkspace/Dataset/BeaverTails/BeaverTails_Train.csv",
        "test_data":       "/kaggle/input/worksapce/orkspace/Dataset/BeaverTails/BeaverTails_Test.csv"
    },
    "truthfulqa": {
        "adapter_weights": "/kaggle/input/worksapce/orkspace/LLaMa-2-7b-TruthfulQA/results/truthfulqa_adapter/adapter_model.safetensors",
        "gamma":           "/kaggle/input/worksapce/orkspace/LLaMa-2-7b-TruthfulQA/results/truthfulqa_adapter/adapter_config.json",
        "base_weights":    "/kaggle/input/worksapce/orkspace/LLaMa-2-7b-TruthfulQA/base_model_weights.pth",
        "train_data":      "/kaggle/input/worksapce/orkspace/Dataset/TruthfulQA/TruthfulQA_Train.csv",
        "test_data":       "/kaggle/input/worksapce/orkspace/Dataset/TruthfulQA/TruthfulQA_Test.csv"
    }
}

# FFN definition
class ExpertFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=500)

def text_to_numeric(series):
    return vectorizer.fit_transform(series).toarray()

# Load data
def load_data(name):
    p = expert_configs[name]
    if p['train_data'].endswith('.json'):
        td = pd.read_json(p['train_data'])
        vd = pd.read_json(p['test_data'])
    else:
        td = pd.read_csv(p['train_data'])
        vd = pd.read_csv(p['test_data'])
    col = td.select_dtypes(include=['object']).columns
    if len(col):
        td = pd.DataFrame(text_to_numeric(td[col[0]]))
        vd = pd.DataFrame(text_to_numeric(vd[col[0]]))
    return td.select_dtypes(include=['number']), vd.select_dtypes(include=['number'])

# Initialize experts
experts = {}
for name in expert_configs:
    td, vd = load_data(name)
    dim = td.shape[1] if td.shape[1]>0 else 1
    experts[name] = {'ffn': ExpertFFN(dim,128,64), 'train_data': td, 'test_data': vd}

# Temperature-scaled softmax
def temperature_scaled_softmax(gv, temp=0.7):
    t = torch.tensor(list(gv.values()), dtype=torch.float32)
    probs = F.softmax(t/temp, dim=0)
    return {k: v.item() for k,v in zip(gv.keys(), probs)}

# Entropy reg & KL
def entropy_regularization(p): return -torch.sum(p*torch.log(p+1e-8))
def kl_divergence(p,q,eps=1e-8): return torch.sum(p*torch.log(p/q))

def update_gamma_values(gv, losses, sf=0.1):
    up = {}; tot = sum(losses.values())
    for e,l in losses.items(): up[e] = gv[e]*(tot/(l+1e-8))*sf
    s = sum(up.values()); return {k:v/s for k,v in up.items()}

# Router with gating loss
class MoCaERouterWithPenalties(nn.Module):
    def __init__(self, ffns, gv, prev=None, temp=0.7):
        super().__init__(); self.ffns=ffns; self.gamma_values=gv; self.previous_gamma_values=prev or gv; self.temperature=temp
    def forward(self,x):
        gs = temperature_scaled_softmax(self.gamma_values, self.temperature)
        outs = {e:fn(x)*gs[e] for e,fn in self.ffns.items()}
        wsum = sum(outs.values())
        ent = entropy_regularization(torch.tensor(list(gs.values())))
        klp = kl_divergence(torch.tensor(list(gs.values())), torch.tensor(list(self.previous_gamma_values.values())))
        num = len(gs); uni = torch.full((num,),1.0/num)
        gl = kl_divergence(torch.tensor(list(gs.values())), uni)
        self.previous_gamma_values = self.gamma_values
        total = torch.mean(wsum) + 0.1*ent + 0.01*klp + 0.05*gl
        losses = {e: total.item() for e in self.ffns}
        self.gamma_values = update_gamma_values(self.gamma_values, losses)
        return total, wsum, ent, klp

# Initialize router
ep_ffns = {n: experts[n]['ffn'] for n in experts}
gamma_values = {n:1.0 for n in experts}
router = MoCaERouterWithPenalties(ep_ffns, gamma_values)

# Process
def process_input_data_with_penalties():
    for e,v in experts.items():
        df=v['train_data']
        if df.empty: print(f"Skipping {e}"); continue
        t=torch.tensor(df.values, dtype=torch.float32)
        L,WS,Ent,KL=router(t)
        print(f"Processed {e} - Loss:{L.item():.4f} Ent:{Ent.item():.4f} KL:{KL.item():.4f}")
process_input_data_with_penalties()

# Save aggregated embeddings both locally and to Kaggle output
def save_aggregated_output_embeddings():
    agg={}
    for e,v in experts.items():
        df=v['train_data']
        if df.empty: continue
        t=torch.tensor(df.values, dtype=torch.float32)
        _,WS,_,_ = router(t)
        agg[e]=WS.detach().cpu().numpy()
    # local path
    local='/workspace/Dataset/aggregated_embeddings'
    os.makedirs(local,exist_ok=True)
    lp=os.path.join(local,'aggregated_embeddingsmoe_gl.npy')
    np.save(lp,agg)
    print(f"Saved to {lp}")
    # Kaggle output
    kop='/kaggle/working/aggregated_embeddingsmoe_gl.npy'
    np.save(kop,agg)
    print(f"Also saved to {kop}")
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

