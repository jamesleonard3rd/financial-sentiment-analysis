import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "yiyanghkust/finbert-tone"
LABELS = ["negative", "neutral", "positive"]  # model’s output order

# Keep it simple: load on each run. (You can cache the model object later if needed.)
def score_texts_with_finbert(text_list, batch_size=16):
    """
    text_list: list of strings (headlines)
    returns: DataFrame with probs and label per text
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    out_rows = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        enc = tokenizer(
            batch, truncation=True, padding=True, return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().tolist()

        for t, p in zip(batch, probs):
            label_id = int(max(range(3), key=lambda j: p[j]))
            out_rows.append({
                "text": t,
                "prob_negative": float(p[0]),
                "prob_neutral":  float(p[1]),
                "prob_positive": float(p[2]),
                "label": LABELS[label_id],
            })

    return pd.DataFrame(out_rows)
