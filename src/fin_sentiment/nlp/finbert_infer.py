import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "yiyanghkust/finbert-tone"  # or ProsusAI/finbert

def score_texts_with_finbert(text_list, batch_size=16):
    """
    Returns a DataFrame with columns:
    text, prob_negative, prob_neutral, prob_positive, label
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    # Read the model's own label order, e.g. {0:'neutral',1:'positive',2:'negative'}
    id2label = model.config.id2label
    # Make a list like ['neutral','positive','negative'] in the exact order of logits
    ordered_labels = [id2label[i].lower() for i in range(len(id2label))]

    rows = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().tolist()

        for text, p in zip(batch, probs):
            row = {"text": text}
            # add prob_<label> for whatever order the model uses
            for j, lab in enumerate(ordered_labels):
                row[f"prob_{lab}"] = float(p[j])

            # ensure all three keys exist for consistency
            for lab in ["negative", "neutral", "positive"]:
                row.setdefault(f"prob_{lab}", 0.0)

            # pick the predicted label by max prob (using the correct order)
            pred_idx = int(max(range(len(p)), key=lambda j: p[j]))
            row["label"] = ordered_labels[pred_idx]
            rows.append(row)

    return pd.DataFrame(rows)
