import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "roberta-base"
NUM_RELATIONS = 6
MAX_SPAN_WIDTH = 5


class Encoder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(DEVICE)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.squeeze(0), inputs


class REXELModel(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.span_scorer = nn.Linear(hidden_dim * 2, 1)
        self.relation_head = nn.Linear(hidden_dim * 2, NUM_RELATIONS)

    def score_span(self, s, e):
        return torch.sigmoid(self.span_scorer(torch.cat([s, e])))

    def span_embed(self, span):
        return torch.mean(span, dim=0)

    def predict_relation(self, e1, e2):
        return self.relation_head(torch.cat([e1, e2]))


class GraphJudge(nn.Module):
    def __init__(self, hidden_dim, relation_dim):
        super().__init__()
        self.scorer = nn.Linear((hidden_dim * 2) + relation_dim, 1)

    def forward(self, h, r, t):
        return torch.sigmoid(self.scorer(torch.cat([h, r, t])))


class KGModel:
    def __init__(self):
        self.encoder = Encoder()
        h = self.encoder.model.config.hidden_size

        self.rexel = REXELModel(h).to(DEVICE)
        self.judge = GraphJudge(h, NUM_RELATIONS).to(DEVICE)

    def extract(self, text):
        emb, inputs = self.encoder.encode(text)
        tokens = self.encoder.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        spans = []
        for i in range(len(emb)):
            for j in range(i, min(i + MAX_SPAN_WIDTH, len(emb))):
                if self.rexel.score_span(emb[i], emb[j]) > 0.5:
                    span_vec = self.rexel.span_embed(emb[i:j+1])
                    spans.append((tokens[i:j+1], span_vec))

        triples = []
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                (t1, e1), (t2, e2) = spans[i], spans[j]

                rel = self.rexel.predict_relation(e1, e2)
                rid = torch.argmax(rel).item()

                score = self.judge(e1, rel, e2).item()

                if score > 0.7:
                    triples.append({
                        "head": " ".join(t1),
                        "relation": f"rel_{rid}",
                        "tail": " ".join(t2)
                    })

        return triples