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
