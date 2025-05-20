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
    local='Path'
    os.makedirs(local,exist_ok=True)
    lp=os.path.join(local,'aggregated_embeddingsmoe_gl.npy')
    np.save(lp,agg)
    print(f"Saved to {lp}")
  
    kop='Path'
    np.save(kop,agg)
    print(f"Also saved to {kop}")
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
