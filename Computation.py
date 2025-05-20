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

