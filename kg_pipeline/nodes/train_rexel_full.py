import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
import json
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "roberta-base"

MAX_SPAN_WIDTH = 5
NUM_TYPES = 10
NUM_RELATIONS = 20


# =========================
# ENCODER
# =========================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)

    def forward(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(DEVICE)

        outputs = self.model(**inputs)
        return outputs.last_hidden_state.squeeze(0), inputs


# =========================
# REXEL MODEL (JOINT)
# =========================
class REXEL(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.span_scorer = nn.Linear(hidden_dim * 2, 1)
        self.type_head = nn.Linear(hidden_dim, NUM_TYPES)
        self.rel_head = nn.Linear(hidden_dim * 2, NUM_RELATIONS)

    def span_score(self, s, e):
        return torch.sigmoid(self.span_scorer(torch.cat([s, e])))

    def span_embed(self, span):
        return torch.mean(span, dim=0)

    def predict_type(self, e):
        return self.type_head(e)

    def predict_rel(self, e1, e2):
        return self.rel_head(torch.cat([e1, e2]))


# =========================
# DATA LOADER
# =========================
def load_docred(path):
    with open(path) as f:
        return json.load(f)


# =========================
# HELPERS
# =========================
def flatten_text(doc):
    return " ".join([" ".join(sent) for sent in doc["sents"]])


def get_gold_spans(doc):
    spans = []
    for entity in doc["vertexSet"]:
        for mention in entity:
            spans.append((mention["pos"][0], mention["pos"][1]))
    return spans


def get_relations(doc):
    rels = []
    for l in doc.get("labels", []):
        rels.append((l["h"], l["t"], l["r"]))
    return rels


# =========================
# TRAIN
# =========================
def train():
    encoder = Encoder().to(DEVICE)
    hidden_dim = encoder.model.config.hidden_size

    model = REXEL(hidden_dim).to(DEVICE)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(model.parameters()),
        lr=2e-5
    )

    span_loss_fn = nn.BCELoss()
    type_loss_fn = nn.CrossEntropyLoss()
    rel_loss_fn = nn.CrossEntropyLoss()

    data = load_docred("data/train_annotated.json")

    for epoch in range(3):
        print(f"\n===== Epoch {epoch} =====")

        total_loss = 0

        for doc in tqdm(data[:200]):  # limit for speed

            text = flatten_text(doc)
            embeddings, inputs = encoder(text)

            gold_spans = get_gold_spans(doc)
            gold_rels = get_relations(doc)

            seq_len = embeddings.shape[0]

            span_loss = 0
            type_loss = 0
            rel_loss = 0

            # =========================
            # SPAN DETECTION
            # =========================
            predicted_spans = []

            for i in range(seq_len):
                for j in range(i, min(i + MAX_SPAN_WIDTH, seq_len)):

                    s = embeddings[i]
                    e = embeddings[j]

                    score = model.span_score(s, e).squeeze()  # ✅ FIX HERE

                    label = 0
                    for gs in gold_spans:
                        if i == gs[0] and j == gs[1]:
                            label = 1
                            break

                    label = torch.tensor(label, dtype=torch.float).to(DEVICE)  # scalar

                    span_loss += span_loss_fn(score, label)

                    if score.item() > 0.5:
                        span_vec = model.span_embed(embeddings[i:j+1])
                        predicted_spans.append((i, j, span_vec))

            # =========================
            # TYPE PREDICTION (dummy labels)
            # =========================
            for (_, _, vec) in predicted_spans:
                logits = model.predict_type(vec)
                target = torch.tensor([0]).to(DEVICE)  # placeholder
                type_loss += type_loss_fn(logits.view(1, -1), target)

            # =========================
            # RELATION EXTRACTION
            # =========================
            for (h, t, r) in gold_rels:
                if h < embeddings.shape[0] and t < embeddings.shape[0]:

                    e1 = embeddings[h]
                    e2 = embeddings[t]

                    pred = model.predict_rel(e1, e2)

                    target = torch.tensor([0]).to(DEVICE)
                    rel_loss += rel_loss_fn(pred.view(1, -1), target)

            loss = span_loss + type_loss + rel_loss

            if loss == 0:
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("Total Loss:", total_loss)

    torch.save(model.state_dict(), "rexel_full.pt")
    print("Model saved!")


if __name__ == "__main__":
    train()