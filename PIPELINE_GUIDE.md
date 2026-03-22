# Knowledge Graph Extraction Pipeline - Complete Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Stage A: Denoising](#stage-a-denoising)
4. [Stage B: Phonetic Correction](#stage-b-phonetic-correction)
5. [Stage C: GraphJudge Verification](#stage-c-graphjudge-verification)
6. [How to Run](#how-to-run)
7. [Troubleshooting](#troubleshooting)

---

## Project Overview

### What Does This Pipeline Do?

This system **extracts structured knowledge from noisy text** and verifies it against the source document.

**Input:** A messy text document (transcribed speech, scanned document, web content)
```
"uh so doctor evelyn reed she is uh a cardiologist from the mayo clinic
in rochester minnesota... the drug showed like a thirty percent reduction...
FLASH SALE THIS WEEK ONLY..."
```

**Output:** Verified knowledge triples stored in a knowledge graph
```
{
  "triples": [
    {"head": "doctor evelyn reed", "relation": "works at", "tail": "mayo clinic"},
    {"head": "farma corp", "relation": "manufactures", "tail": "cardia care"},
    {"head": "cardia care", "relation": "reduces", "tail": "heart incidents"}
  ],
  "stored_in": "Neo4j"
}
```

### Key Challenges Addressed

1. **Noisy Input:** ASR/OCR errors ("knew deli" → "New Delhi"), filler words ("uh", "um"), ads
2. **Phantom Facts:** The system might extract plausible but false triples not in the text
3. **Spelling Errors:** "Pharma corp" written as "farma corp" in transcription
4. **Verification:** Need to confirm extracted facts are actually in the source text

---

## Pipeline Architecture

### Full Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      RAW TEXT INPUT (Noisy)                         │
│  "uh doctor smith he uh... he study diseases... CLICK HERE FOR..."  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
                  ┌──────────────────────────┐
                  │  1. DENOISE TEXT         │  ← Remove ads, fillers
                  │     (Entity-Aware)       │
                  └────────┬─────────────────┘
                           │ denoised_text
                           ▼
                  ┌──────────────────────────┐
                  │  2. CORRECT PHONETICS    │  ← Fix ASR/OCR errors
                  │     (ECTD: 2 steps)      │     farma→Pharma, etc
                  └────────┬─────────────────┘
                           │ denoised_text (updated)
                           ▼
                  ┌──────────────────────────┐
                  │  3. GENERATE CANDIDATES  │  ← Find entity pairs
                  │     Coarse Filtering     │
                  └────────┬─────────────────┘
                           │ draft_triples
                           ▼
                  ┌──────────────────────────┐
                  │  4. JOINT REFINE         │  ← Fine extraction
                  │     (Not shown in demo)  │
                  └────────┬─────────────────┘
                           │ draft_triples (refined)
                           ▼
         ┌─────────────────────────────────────┐
         │  5. GRAPH JUDGE VERIFICATION        │ ◄─────────
         │     "Is this in the text?"          │     KEY FOCUS
         │     (Multiple backends)             │ ◄─────────
         └────────┬────────────────────────────┘
                  │ verified_triples
                  ▼
         ┌─────────────────────────────────────┐
         │  6. STORE TO NEO4J                  │
         │     Create knowledge graph          │
         └────────┬────────────────────────────┘
                  │
                  ▼
         ┌─────────────────────────────────────┐
         │   KNOWLEDGE GRAPH DATABASE          │
         └─────────────────────────────────────┘
```

### Pipeline State (Data Flow)

```python
KGState = TypedDict({
    "raw_text": str,                  # ← Input (noisy)
    "denoised_text": str,             # ← After stage A & B
    "mentions": List[Dict],           # ← Detected entities
    "entity_clusters": List[Dict],    # ← After coreference
    "draft_triples": List[Dict],      # ← Extracted but unverified
    "verified_triples": List[Dict],   # ← After judgment (filtered)
    "kg_summary": Dict,               # ← Neo4j stats
    "error": str                      # ← Error message if any
})
```

Each node reads from state, processes, and writes back to state.

---

## Stage A: Denoising

### What is Denoising?

Denoising removes noise while preserving facts needed for knowledge extraction:
- **Removes:** Ads, promotional content, repeated fillers (uh, um, er)
- **Keeps:** All factual information about entities and their relationships
- **Key Insight:** Does NOT summarize or change meaning

### Two Modes

#### Mode 1: Light Cleaning (before entities are known)

```
INPUT:
"uh so like you know apple is making phones in california
um the ceo tim cook he announced new features"

PROMPT:
"Clean the text by removing filler speech, fixing punctuation.
Do NOT remove factual content or summarize.
Return ONLY cleaned text."

OUTPUT:
"apple is making phones in california
the ceo tim cook announced new features"
```

**Used:** Before NER (named entity recognition) stage

#### Mode 2: Entity-Aware Filtering (after entities detected)

**Idea:** We know which entities matter (Mayo Clinic, Cardia Care, Doctor Smith).
So filter the text to ONLY keep sentences mentioning those entities.

```
INPUT TEXT:
"uh doctor evelyn reed works at mayo clinic... they study cardia care...
FLASH SALE: 50% OFF VITAMINS...
the drug reduced incidents..."

DETECTED ENTITIES:
["doctor evelyn reed", "mayo clinic", "cardia care"]

STEP 1 - Sentence Filtering:
Keep only sentences containing these entities.
(removes the FLASH SALE ad)

STEP 2 - LLM Refinement:
Polish the filtered text for clarity.

OUTPUT:
"doctor evelyn reed works at mayo clinic.
They study cardia care.
The drug reduced incidents..."
```

**Used:** After NER, to clean entity-relevant content only

### Code Example

```python
def denoise_text(state: KGState) -> KGState:
    raw_text = state["raw_text"]
    mentions = state.get("mentions", [])

    if not mentions:
        # LIGHT MODE: generic cleanup
        prompt = f"Clean this text (remove fillers, keep facts): {raw_text}"
        cleaned = call_ollama(prompt)
    else:
        # ENTITY-AWARE MODE: filter to entities only
        entity_list = [m["text"] for m in mentions]

        # Step 1: Sentence filtering
        sentences = raw_text.split(".")
        filtered = [s for s in sentences
                   if any(e.lower() in s.lower() for e in entity_list)]

        # Step 2: LLM refinement
        prompt = f"""Clean document keeping only these entities: {entity_list}
        Remove ads, keep facts about these entities.
        {filtered}"""
        cleaned = call_ollama(prompt)

    state["denoised_text"] = cleaned
    return state
```

### When to Use

| Scenario | Mode | Why |
|----------|------|-----|
| Very noisy input (transcription, OCR) | Light | No entities detected yet |
| After entity detection | Entity-aware | Remove off-topic content |
| Web articles with ads | Entity-aware | Remove promotional sections |
| Medical papers | Entity-aware | Focus on entity-relevant facts |

---

## Stage B: Phonetic Correction

### The Problem

Speech-to-text (ASR) and image-to-text (OCR) produce phonetic errors:

| Error | Intended | Context Clue |
|-------|----------|--------------|
| "knew deli" | "New Delhi" | "london england" (place names nearby) |
| "P H C" | "PHC" | "stock jumped" (looks like ticker symbol) |
| "farma corp" | "Pharma Corp" | "drug, developed, manufactures" |
| "A T one receptor" | "AT1 receptor" | Medical domain |

### The ECTD Approach (Two Steps)

**ECTD** = Entity-Centric Text Denoising (from GraphJudge paper)

#### Step A: Rough Entity Extraction (As-Is)

```
PROMPT:
"List every proper noun in this text.
The text may be misspelled. List them AS-IS (don't correct yet).
Return only JSON array: ["name1", "name2"]"

INPUT TEXT:
"...knew deli india and london england next year..."

OUTPUT:
["knew deli", "london", "england"]
```

**Why as-is?** We capture the phonetic version before context.

#### Step B: Context-Based Correction

```
PROMPT:
"I found these entities (may be misspelled):
["knew deli", "A T one receptor", "P H C"]

Use surrounding context to correct their spelling.
Rules:
1. If near 'stock, traded' → likely company ticker
2. If sounds like place + "london england" → geography
3. If medical domain → domain-specific spelling
Return full corrected text."

CONTEXT:
"...the trial is expected in knew deli india and london england next year..."
"...it targets the A T one receptor..."
"...P H C jumped fifteen percent..."

OUTPUT:
"...the trial is expected in New Delhi India and London England next year..."
"...it targets the AT1 receptor..."
"...PHC jumped fifteen percent..."
```

### Step B Logic

1. **Domain Detection:** What is the subject?
   - Medical? → Look for domain-specific terms
   - Business? → Stock tickers, companies
   - Geography? → Place names

2. **Phonetic Matching:** Does it sound like a real word/name?
   - "knew" sounds like "new"
   - "P H C" spelled out → probably an acronym

3. **Context Usage:** Surrounding words give clues
   - "stock jumped P H C" → ticker symbol
   - "london england knew deli" → city name

### Code Structure

```python
def correct_phonetics(state: KGState) -> KGState:
    denoised_text = state["denoised_text"]

    # STEP A: Extract rough entities (with errors)
    step_a_prompt = f"""Read text, list proper nouns AS-IS (don't correct).
Return JSON array only. {denoised_text}"""
    raw_entities = json.loads(call_ollama(step_a_prompt))

    # STEP B: Use context to correct spelling
    step_b_prompt = f"""Correct these potentially misspelled entities using context:
{json.dumps(raw_entities)}
Context: {denoised_text}
Return fully corrected text."""
    corrected = call_ollama(step_b_prompt)

    state["denoised_text"] = corrected
    return state
```

### When it Works Best

✅ **Strong:** Medical, technical, geographic terms (have conventions)
✅ **Strong:** Company names, acronyms (standard abbreviations)
✅ **Moderate:** Person names (common spellings)
❌ **Weak:** Rare names, novel terms (context can't help)

---

## Stage C: GraphJudge Verification

### The Core Question

> **"Is this extracted triple actually supported by the source text?"**

Without verification, the pipeline would extract plausible-sounding triples that aren't in the document.

### How GraphJudge Works

#### Step 1: Form the Instruction

Every triple becomes a natural language question:

```
Triple: (Doctor Smith, works at, Harvard)
↓
Instruction: "Is this true: Doctor Smith works at Harvard?"

Triple: (Mayo Clinic, manufactures, Cardia Care)
↓
Instruction: "Is this true: Mayo Clinic manufactures Cardia Care?"
```

Format: `"Is this true: {head} {relation} {tail}?"`

#### Step 2: Pass to Judge Backend

```
┌────────────────────────────────┐
│  JUDGE BACKEND (Multiple Options)
├────────────────────────────────┤
│  1. BERT Classifier            │  (Fast, small)
│  2. LoRA+Llama2 (Fine-tuned)   │  (Smart, requires GPU)
│  3. Ollama Llama3.1 (Local LLM) │  (Simpler, CPU-friendly)
└────────────────────────────────┘
         ↓
    call_ollama() or _judge_with_bert()
         ↓
    Response String
```

#### Step 3: Parse Response

```python
def _is_true_from_generated_response(response: str, limit=100) -> bool:
    window = response.strip().lower()[:100]
    # If "no" or "false" in first 100 chars → REJECT
    if "no" in window or "false" in window:
        return False
    return True
```

Examples:

| Response | First 100 chars contain... | Verdict |
|----------|---------------------------|---------|
| "No, mayo clinic does not..." | "no" | ❌ FALSE |
| "Yes, it's true the clinic..." | (no "no") | ✅ TRUE |
| "False, this is not mentioned..." | "false" | ❌ FALSE |
| "The text does not..." | (no match) | ✅ TRUE |

### Three Backends

#### Backend 1: BERT Classifier

```
Input:  "Instruction: Is this true: X Y Z?
         Input: [context text]"

Model:  BERT fine-tuned for entailment/contradiction

Output: Binary classification (0=false, 1=true) + confidence score
        "No, it is not true. (confidence: 0.97)"

Pros:   ⚡ Fast (single forward pass)
        💾 Small model

Cons:   ❌ Requires fine-tuned BERT weights
        ❌ Binary only (no explanation)
```

#### Backend 2: LoRA+Llama2

```
Input:  GraphJudge instruction format (from prepare_KGCom.ipynb)
        ### Instruction: Is this true: X Y Z?
        ### Input: [context]
        ### Response:

Model:  Llama2-7B with GraphJudge LoRA adapter

Output: Natural language response
        "Yes, it is true according to the document.
         The text explicitly mentions..."

Pros:   💡 Generates explanations
        🎯 Fine-tuned on GraphJudge dataset

Cons:   ❌ Slower (generates text)
        ❌ Requires GPU ideally
        ❌ Needs fine-tuned LoRA weights
```

#### Backend 3: Ollama Llama3.1 (Default)

```
Input:  Same GraphJudge format
        ### Instruction: Is this true: X Y Z?
        ### Input: [context]
        ### Response:

Model:  Llama3.1:8B (local, open-source)

Output: Natural language response
        "No, this is not mentioned in the text..."

Pros:   ✅ Local (no API keys)
        ✅ Runs on CPU
        ✅ No fine-tuning needed
        ✅ **DEFAULT & FALLBACK**

Cons:   ⚠️  Requires Ollama installation
        ⚠️  Slightly slower than BERT
        ⚠️  May add verbose explanation
```

### Fallback Chain

```python
def _judge_backend(instruction: str, context_text: str) -> str:
    backend = os.getenv("JUDGE_BACKEND", "ollama")
    fallback = os.getenv("JUDGE_FALLBACK_OLLAMA", "true") == "true"

    try:
        if backend == "bert":
            return _judge_with_bert(instruction, context_text)
        if backend == "lora":
            return _judge_with_lora(instruction, context_text)
    except Exception as exc:
        if not fallback:
            raise
        print(f"{backend} unavailable; falling back to Ollama.")

    return call_ollama(_build_graphjudge_prompt(instruction, context_text))
```

**Priority Order:**
1. Try configured backend (BERT or LoRA)
2. If it fails AND fallback enabled → use Ollama
3. If fallback disabled → raise exception

### Configuration (.env)

```bash
# Which backend to use (default: ollama)
JUDGE_BACKEND=ollama

# Enable fallback to Ollama if primary fails
JUDGE_FALLBACK_OLLAMA=true

# Optional: BERT weights path
JUDGE_BERT_WEIGHTS=models/bert-base-classifier-rebel-sub

# Optional: LoRA base model path
JUDGE_LORA_BASE_MODEL=models/llama-2-7b-hf

# Optional: LoRA adapter path
JUDGE_LORA_WEIGHTS=models/llama2-7b-lora-genwiki-context

# Ollama API
OLLAMA_API_URL=http://localhost:11434/api/generate
```

### Under the Hood: GraphJudge Instruction Format

The instruction follows a specific format learned from the GraphJudge paper (`prepare_KGCom.ipynb`):

```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
Is this true: {head} {relation} {tail}?
### Input:
{context_text}
### Response:
```

This format helps language models understand the verification task clearly.

### Example Judgment Flow

```
CONTEXT TEXT:
"Doctor Smith works at Harvard University.
 He studies malaria transmission."

TRIPLE 1: (Doctor Smith, works at, Harvard)
  Instruction: "Is this true: Doctor Smith works at Harvard?"
  Response: "Yes, the text mentions Doctor Smith works at Harvard University"
  Parsed: Contains no "no"/"false" → ✅ VERIFIED

TRIPLE 2: (Doctor Smith, studies, Psychology)
  Instruction: "Is this true: Doctor Smith studies Psychology?"
  Response: "No, the text says he studies malaria, not psychology"
  Parsed: Contains "no" → ❌ REJECTED

TRIPLE 3: (Harvard, is in, Boston)
  Instruction: "Is this true: Harvard is in Boston?"
  Response: "No, this information is not provided in the given text"
  Parsed: Contains "no" → ❌ REJECTED (not in source)

RESULT: Only [Triple 1] verified
```

---

## How to Run

### Prerequisites

1. **Ollama Running**
   ```bash
   ollama serve
   # In another terminal
   ollama pull llama3.1:8b
   ```

2. **Neo4j Running** (for storage)
   ```bash
   docker run --name neo4j -d \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/password \
     neo4j
   ```

3. **Project Setup**
   ```bash
   cd /Users/bibeksinghdhody/Downloads/btp
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

### Run the Demo (Interactive Tutorial)

```bash
cd /Users/bibeksinghdhody/Downloads/btp
python3 PIPELINE_DEMO.py
```

This script will:
- Walk through each stage with explanations
- Wait for you to press Enter between demos
- Show input/output examples
- Explain backend selection logic

### Run the Full Pipeline

```bash
cd /Users/bibeksinghdhody/Downloads/btp
python3 -m kg_pipeline.main
```

This executes:
1. Denoise the hardcoded test text
2. Correct phonetic errors
3. Generate entity candidates
4. Refine and extract relations
5. Judge each triple with Ollama
6. Store verified triples to Neo4j

### Expected Output

```
Ollama server is running!

---------(1) DENOISING TEXT (ENTITY-AWARE)---------
  >>> Step 1: Sentence Filtering...
  >>> Step 2: LLM Refinement...
Denoising complete.

---(2) CORRECTING PHONETIC ERRORS---
  Step A found 5 rough entities: [...]
Phonetic correction complete.

---(3) GENERATING CANDIDATES---
Candidates generated.

---(4) JOINT REFINING---
Refined 12 candidates to 8 triples.

---(5) GRAPH JUDGE VERIFICATION---
Verified 7 triples out of 8 drafts.
  Rejected: Doctor Smith - teaches - Philosophy

Connected to Neo4j: neo4j://localhost:7687
Stored 7 triples with 5 entities.
Done!
```

---

## Troubleshooting

### Ollama Server Not Running

**Error:**
```
⚠️  Ollama server is not running
    Please start it with: ollama serve
```

**Fix:**
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Run pipeline
cd /Users/bibeksinghdhody/Downloads/btp
python3 -m kg_pipeline.main
```

### LoRA Model Not Found

**Error:**
```
FileNotFoundError: LoRA directory not found: models/llama2-7b-lora-genwiki-context
```

**Fix:**
```bash
# Option 1: Use Ollama backend instead (default)
export JUDGE_BACKEND=ollama

# Option 2: Download the model
python3 -c "
from transformers import AutoTokenizer, LlamaForCausalLM
# Downloads from Hugging Face
"
```

### JSON Parse Error in Phonetic Corrector

**Error:**
```
JSONDecodeError: ... at line 1 column 2
```

**Cause:** Ollama response wasn't valid JSON (maybe returned explanation text)

**Fix:** The script already handles this with a fallback to empty list. To reduce errors:
- Use stricter prompts ("Return ONLY JSON array, no other text")
- Increase `temperature=0` for consistent responses

### Neo4j Connection Failed

**Error:**
```
Failed to connect to Neo4j at neo4j+s://localhost:7687
```

**Fix:**
1. Check Neo4j is running:
   ```bash
   docker ps | grep neo4j
   docker logs neo4j
   ```

2. Check credentials in `.env`:
   ```bash
   NEO4J_URI=neo4j://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=password
   ```

3. Verify port 7687 is open:
   ```bash
   nc -zv localhost 7687
   ```

---

## Key Insights

### Why Three Stages

1. **Denoising:** Make text clean and readable
   - Removes distraction (ads, fillers)
   - Entity-aware mode for precision

2. **Phonetic Correction:** Fix transcription errors
   - Uses context clues (domain, surrounding words)
   - Two-step approach (rough + refined)

3. **Graph Judge:** Verify facts against source
   - Prevents hallucinated triples
   - Multiple backend options (BERT/LoRA/Ollama)
   - Graceful fallback to local LLM

### Why Ollama as Default

| Feature | Ollama | BERT | LoRA |
|---------|--------|------|------|
| Local | ✅ | ✅/❌ | ✅/❌ |
| No GPU Required | ✅ | ✅ | ❌ |
| Explanation | ✅ | ❌ | ✅ |
| Speed | Medium | Fast | Slow |
| Setup | Easy | Needs weights | Needs weights+GPU |
| Default | ✅ YES | - | - |

Ollama balances simplicity, quality, and robustness.

### Pipeline Invariants

These must hold for the pipeline to work:

1. **After denoising:** `denoised_text` is non-empty and fact-preserving
2. **After correction:** `denoised_text` uses canonical entity names
3. **After extraction:** `draft_triples` have all three keys (head, relation, tail)
4. **After judgment:** `verified_triples` ⊆ `draft_triples` (only verified subset)
5. **Before storage:** `verified_triples` all have evidence in `denoised_text`

---

## Next Steps

1. **Run the demo** to see explanations and examples
2. **Run the full pipeline** to extract and verify triples
3. **Check Neo4j** for stored knowledge graph
4. **Customize** test text in `kg_pipeline/main.py`
5. **Experiment** with different judge backends in `.env`
