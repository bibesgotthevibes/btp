# KG Pipeline - Quick Reference & Cheat Sheet

## One-Minute Overview

```
Noisy Text Input
    ↓
[Denoise] → Remove ads, fillers (entity-aware)
    ↓
[Phonetics] → Fix "knew deli" → "New Delhi" errors
    ↓
[Extract] → Find entity pairs, propose relations
    ↓
[Judge] → "Is this actually in the text?" (VERIFIED ✓ or REJECTED ✗)
    ↓
Neo4j Database (Knowledge Graph)
```

---

## The 3 Key Stages (What We're Explaining)

### 1️⃣ DENOISING (Stage 1)

**What:** Remove noise while keeping facts

**Light Mode** (no entities yet):
```python
prompt = "Clean text: remove 'uh' 'um', fix punctuation. Keep facts."
cleaned = call_ollama(prompt)
```

**Entity-Aware Mode** (after NER):
```python
# Step 1: Keep only sentences mentioning detected entities
kept_sentences = [s for s in sentences if any(e in s for e in entities)]

# Step 2: Polish with LLM
prompt = f"Clean these sentences (keep only about: {entities})"
cleaned = call_ollama(prompt)
```

| Input | Output | Removed |
|-------|--------|---------|
| "uh doctor smith uh works at harvard... BUY NOW 50% OFF" | "doctor smith works at harvard" | filler, ad |

---

### 2️⃣ PHONETIC CORRECTION (Stage 2)

**Problem:**
```
"knew deli" → should be "New Delhi"
"P H C" → should be "PHC"
"A T one receptor" → should be "AT1 receptor"
```

**ECTD Approach** (2 steps):

**Step A: Rough Extraction (as-is)**
```python
step_a_prompt = "List proper nouns from text (don't correct yet). JSON array only."
raw_entities = json.loads(call_ollama(step_a_prompt))
# → ["knew deli", "P H C", "A T one"]
```

**Step B: Context-Based Correction**
```python
step_b_prompt = f"""
Correct these entities using context:
{json.dumps(raw_entities)}

Context:
{text}

Rules:
- "knew deli" + "london england" → "New Delhi"
- "P H C" + "stock jumped" → "PHC" (ticker)
- "A T one" + medical terms → "AT1 receptor"
Return full corrected text."""

corrected = call_ollama(step_b_prompt)
```

**Key Insight:** Context words (place names, "stock", domain terms) guide correction

---

### 3️⃣ GRAPH JUDGE VERIFICATION (Stage 5)

**Problem:** Pipeline extracted triple that's NOT in source text
```
Extracted: (Mayo Clinic, manufactures, Cardia Care)
Source: "Cardia Care is made BY Pharma Corp" ← Mayo didn't make it!
Result: HALLUCINATION → Must reject!
```

**Solution: Form Instruction + Ask Judge**

```
1. Triple → Instruction
   (Doctor Smith, works at, Harvard)
   → "Is this true: Doctor Smith works at Harvard?"

2. Judge Answers
   Backend: BERT / LoRA / Ollama
   Response: "Yes, text says..." or "No, not mentioned..."

3. Parse Response
   if "no" or "false" in first 100 chars:
       return REJECTED
   else:
       return VERIFIED
```

**Three Judge Backends:**

| Backend | Input | Output | Speed | Setup |
|---------|-------|--------|-------|-------|
| **BERT** | "Inst: ... Input: ..." | Binary (0/1) + conf | ⚡ Fast | Needs weights |
| **LoRA+Llama2** | GraphJudge format | Natural text response | 🐢 Slow | GPU + weights |
| **Ollama** (default) | GraphJudge format | Natural text response | ⏱️ Medium | Easy, local |

**Fallback Chain:**
```python
# 1. Try BERT (fast, small)
try:
    return _judge_with_bert(...)
except:
    pass

# 2. Try LoRA (smart, slow)
try:
    return _judge_with_lora(...)
except:
    pass

# 3. Use Ollama (default, always works)
return call_ollama(...)  # ← Falls back here
```

**Example Verification:**

```
Context: "Doctor Smith works at Harvard. He studies malaria."

Triple 1: (Doctor Smith, works at, Harvard)
  Instruction: "Is this true: Doctor Smith works at Harvard?"
  Response: "Yes, the text explicitly says..."
  Result: ✅ VERIFIED

Triple 2: (Doctor Smith, studies, Medicine)
  Instruction: "Is this true: Doctor Smith studies Medicine?"
  Response: "No, the text says he studies malaria specifically"
  Result: ❌ REJECTED

Triple 3: (Harvard, is in, California)
  Instruction: "Is this true: Harvard is in California?"
  Response: "No, this information is not in the provided text"
  Result: ❌ REJECTED (not in source)
```

---

## Pipeline Code Map

| File | Stage | Key Function | Does |
|------|-------|--------------|------|
| `denoiser.py` | 1 | `denoise_text()` | Removes ads, fillers, keeps facts |
| `phonetic_corrector.py` | 2 | `correct_phonetics()` | ECTD: fixes ASR/OCR errors |
| `judge.py` | 5 | `judge_triples()` | Verifies: "Is this in the text?" |
| `main.py` | ALL | `main()` | LangGraph orchestration, runs all stages |

---

## Key Concepts Glossary

| Term | Meaning |
|------|---------|
| **ECTD** | Entity-Centric Text Denoising (from GraphJudge paper) |
| **Denoise** | Remove noise (ads, fillers) while keeping facts |
| **Entity-Aware** | Filtering/processing based on detected entity list |
| **Phonetic Error** | ASR/OCR mistake ("knew" for "new", "P H C" for "PHC") |
| **Draft Triple** | (h, r, t) extracted but NOT YET verified |
| **Verified Triple** | (h, r, t) confirmed to be in source text |
| **Instruction** | "Is this true: h r t?" form for judge backend |
| **Judgment Response** | Backend's answer to instruction (text or binary) |
| **Backend** | Judge implementation (BERT/LoRA/Ollama) |
| **Fallback** | Switch to Ollama if BERT/LoRA unavailable |

---

## Configuration (.env)

```bash
# Judge Backend Selection
JUDGE_BACKEND=ollama              # Current backend (ollama, bert, lora)
JUDGE_FALLBACK_OLLAMA=true        # Try Ollama if primary fails

# Optional BERT Weights
JUDGE_BERT_WEIGHTS=models/bert-base-classifier-rebel-sub

# Optional LoRA
JUDGE_LORA_BASE_MODEL=models/llama-2-7b-hf
JUDGE_LORA_WEIGHTS=models/llama2-7b-lora-genwiki-context

# Ollama
OLLAMA_API_URL=http://localhost:11434/api/generate

# Neo4j
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

---

## Common Tasks

### Run Interactive Demo
```bash
python3 PIPELINE_DEMO.py
# Shows explanations, examples, and calls to Ollama
```

### Run Full Pipeline on Test Data
```bash
python3 -m kg_pipeline.main
# Runs all 6 stages on hardcoded text in main.py
```

### Check Ollama Status
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Check if running
curl http://localhost:11434/api/generate -X POST \
  -d '{"model": "llama3.1:8b", "prompt": "hello", "stream": false}'
```

### Switch Judge Backend
```bash
# Use Ollama (default)
export JUDGE_BACKEND=ollama

# Try BERT first, fallback to Ollama
export JUDGE_BACKEND=bert

# Force LoRA (no fallback, fail if unavailable)
export JUDGE_BACKEND=lora
export JUDGE_FALLBACK_OLLAMA=false
```

### View Denoised Text
```python
# In kg_pipeline/main.py, after denoise stage:
print(state["denoised_text"])
```

### View Verified Triples
```python
# After judge stage:
print(json.dumps(state["verified_triples"], indent=2))
```

### Check Neo4j Results
```bash
# Start Neo4j browser
docker logs neo4j | grep "org.neo4j.server"

# Then visit: http://localhost:7474
# Query: MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 10
```

---

## Typical Outputs

### Denoising Output
```
INPUT:
  "uh doctor smith he uh uh works at harvard ... CLICK HERE SAVE 50% ..."

OUTPUT:
  "Doctor Smith works at Harvard University."
```

### Phonetic Correction Output
```
INPUT:
  "The trial begins in knew deli and london england next year"

OUTPUT:
  "The trial begins in New Delhi and London England next year"
```

### Judge Verification Output
```
Input Triples: 8
Verified:      7 ✓
Rejected:      1 ✗ (hallucination)

Rejected triple: (Harvard, manufactures, Aspirin)
Reason: Not mentioned in source text
```

---

## Troubleshooting Quick Answers

| Problem | Cause | Fix |
|---------|-------|-----|
| "Ollama server not running" | Ollama not started | `ollama serve` in another terminal |
| "LoRA model not found" | Path not configured | Set `JUDGE_BACKEND=ollama` (default) |
| "JSON parse error" | Ollama returned non-JSON | Retry (llm sometimes outputs explanation) |
| "Neo4j connection failed" | DB not running | `docker run neo4j` or check credentials |
| "Slow judge responses" | Using LoRA backend | Switch to Ollama (faster) |

---

## What Each Code Function Does

### Denoising
```python
denoise_text(state):
    if no entities yet:
        light cleanup (remove "uh", "um")
    else:
        filter to entity-relevant sentences
        LLM polishing
    return state with denoised_text
```

### Phonetic Correction
```python
correct_phonetics(state):
    step_a: extract raw entities as-is
    step_b: use context to correct spelling
    return state with corrected denoised_text
```

### Judge
```python
judge_triples(state):
    for each draft triple (h, r, t):
        instruction = f"Is this true: {h} {r} {t}?"
        response = judge_backend(instruction, context)
        if "no"/"false" in response[:100]:
            reject
        else:
            verify
    return state with verified_triples
```

---

## Data Flow Example

```
State: {
  "raw_text": "uh doctor smith he uh works at harvard ... BUY NOW!..."
}

After denoise:
  "denoised_text": "Doctor Smith works at Harvard University"

After phonetics:
  "denoised_text": "Doctor Smith works at Harvard University"  # (no errors)

After candidate gen:
  "draft_triples": [
    {"head": "doctor smith", "relation": "works at", "tail": "harvard"},
    {"head": "harvard", "relation": "is", "tail": "university"}
  ]

After judge (via Ollama):
  "verified_triples": [
    {"head": "doctor smith", "relation": "works at", "tail": "harvard"}
    # (second one rejected: "is" too generic)
  ]

After Neo4j storage:
  "kg_summary": {
    "entities_stored": 2,
    "relations_stored": 1,
    "nodes": ["doctor_smith", "harvard"]
  }
```

---

## Key Insights

1. **Why 3 stages?**
   - Denoise: Make input usable
   - Phonetics: Fix transcription errors
   - Judge: Verify facts (prevent hallucinations)

2. **Why entity-aware denoising?**
   - After NER, we know which entities matter
   - Filter to relevant sentences only
   - Removes off-topic ads and content

3. **Why ECTD (2-step correction)?**
   - Step A captures phonetic version as clue
   - Step B uses context to infer correct spelling
   - Avoids needing pre-built lookup tables

4. **Why multiple judge backends?**
   - BERT: Fast, small, binary
   - LoRA: Smart, explainable, requires GPU
   - Ollama: Balanced, local, no setup

5. **Why Ollama as default?**
   - Local (no API keys)
   - Works on CPU
   - No fine-tuned weights needed
   - Graceful fallback always available

---

## Files to Review

| File | Purpose |
|------|---------|
| `PIPELINE_DEMO.py` | Interactive walkthrough with Ollama examples |
| `PIPELINE_GUIDE.md` | Detailed explanations with diagrams |
| `kg_pipeline/nodes/denoiser.py` | Denoising implementation |
| `kg_pipeline/nodes/phonetic_corrector.py` | ECTD (2-step correction) |
| `kg_pipeline/nodes/judge.py` | GraphJudge verification (BERT/LoRA/Ollama) |
| `kg_pipeline/main.py` | LangGraph pipeline orchestration |

---

## Next: Running the Code

```bash
# 1. Start Ollama
ollama serve

# 2. (Optional) Start Neo4j for storage
docker run -d -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password neo4j

# 3. Run interactive demo
python3 PIPELINE_DEMO.py

# 4. Run full pipeline
python3 -m kg_pipeline.main

# 5. Check results in Neo4j (http://localhost:7474)
```

Good luck! 🚀
