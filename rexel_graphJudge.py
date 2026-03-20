import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "roberta-base"

NUM_ENTITY_TYPES = 7
NUM_RELATIONS = 6
MAX_SPAN_WIDTH = 5   # max tokens in entity span


# =========================
# ENCODER
# =========================
class Encoder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(DEVICE)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.squeeze(0), inputs


# =========================
# REXEL MODEL (SPAN-BASED)
# =========================
class REXELModel(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        # span scoring
        self.span_scorer = nn.Linear(hidden_dim * 2, 1)

        # entity type
        self.type_head = nn.Linear(hidden_dim, NUM_ENTITY_TYPES)

        # relation
        self.relation_head = nn.Linear(hidden_dim * 2, NUM_RELATIONS)

        # coreference
        self.coref_head = nn.Linear(hidden_dim * 2, 1)

    def score_span(self, start_emb, end_emb):
        x = torch.cat([start_emb, end_emb], dim=-1)
        return torch.sigmoid(self.span_scorer(x))

    def get_span_embedding(self, span_embs):
        return torch.mean(span_embs, dim=0)

    def predict_relation(self, e1, e2):
        x = torch.cat([e1, e2], dim=-1)
        return self.relation_head(x)

    def predict_coref(self, e1, e2):
        x = torch.cat([e1, e2], dim=-1)
        return torch.sigmoid(self.coref_head(x))


# =========================
# GRAPH JUDGE
# =========================
class GraphJudge(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.scorer = nn.Linear(hidden_dim * 3, 1)

    def forward(self, head, relation, tail):
        x = torch.cat([head, relation, tail], dim=-1)
        return torch.sigmoid(self.scorer(x))


# =========================
# SPAN EXTRACTION
# =========================
def extract_spans(embeddings, tokens, model, threshold=0.5):
    spans = []

    seq_len = embeddings.shape[0]

    for i in range(seq_len):
        for j in range(i, min(i + MAX_SPAN_WIDTH, seq_len)):

            start_emb = embeddings[i]
            end_emb = embeddings[j]

            score = model.score_span(start_emb, end_emb).item()

            if score > threshold:
                span_emb = embeddings[i:j+1]
                span_vec = model.get_span_embedding(span_emb)

                span_text = " ".join(tokens[i:j+1])

                spans.append({
                    "text": span_text,
                    "embedding": span_vec,
                    "start": i,
                    "end": j,
                    "score": score
                })

    return spans


# =========================
# CANDIDATE PAIRS
# =========================
def generate_pairs(spans):
    pairs = []
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            pairs.append((spans[i], spans[j]))
    return pairs


# =========================
# PIPELINE
# =========================
class KGSystem:
    def __init__(self):
        self.encoder = Encoder()
        hidden_dim = self.encoder.model.config.hidden_size

        self.rexel = REXELModel(hidden_dim).to(DEVICE)
        self.judge = GraphJudge(hidden_dim).to(DEVICE)

    def run(self, text):
        print("\n--- Encoding ---")
        embeddings, inputs = self.encoder.encode(text)

        tokens = self.encoder.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0]
        )

        print("\n--- Span Extraction ---")
        spans = extract_spans(embeddings, tokens, self.rexel)

        print(f"Extracted {len(spans)} spans")

        print("\n--- Candidate Pairs ---")
        pairs = generate_pairs(spans)
        print(f"{len(pairs)} pairs")

        triples = []

        print("\n--- Relation + GraphJudge ---")
        for s1, s2 in pairs:

            e1 = s1["embedding"]
            e2 = s2["embedding"]

            # Predict relation logits
            rel_logits = self.rexel.predict_relation(e1, e2)

    # Get relation id
            rel_id = torch.argmax(rel_logits).item()

    # FIX: convert relation logits → same dimension as entity embedding
            rel_emb = torch.zeros_like(e1)
            rel_emb[:rel_logits.shape[0]] = rel_logits

    # GraphJudge scoring
            score = self.judge(e1, rel_emb, e2).item()
            if score > 0.7:
                triples.append({
                    "head": s1["text"],
                    "relation": f"rel_{rel_id}",
                    "tail": s2["text"],
                    "score": score
                })

        print(f"\n--- Final Triples: {len(triples)} ---")
        for t in triples:
            print(t)

        return triples


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    text = """
    Doctor Evelyn Reed is a cardiologist from the Mayo Clinic in Rochester Minnesota.
    She presented research on a new drug called Cardia Care made by Farma Corp.
    The study was published in the New England Journal of Medicine.
    A phase three trial will be conducted in the United States, New Delhi, and London.
    """

    system = KGSystem()
    system.run(text)