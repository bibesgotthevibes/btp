# KG Pipeline - Visual Architecture & Flow Diagrams

## Complete System Architecture

```
╔════════════════════════════════════════════════════════════════════════════╗
║                        KNOWLEDGE GRAPH EXTRACTION PIPELINE                 ║
╚════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT: NOISY TEXT                                 │
│                                                                             │
│ "uh so doctor evelyn reed she is uh a cardiologist from the mayo clinic   │
│  in rochester minnesota... FLASH SALE 50% OFF... the drug showed...       │
│  P H C jumped fifteen percent... the trial is in knew deli india...       │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     │ raw_text
                                     ▼
        ╔════════════════════════════════════════════════════════════╗
        ║                                                            ║
        ║  STAGE 1: DENOISE TEXT (Remove Noise)                    ║
        ║  ─────────────────────────────────────────────────────   ║
        ║  • Remove ads: "FLASH SALE..."                           ║
        ║  • Remove fillers: "uh", "um", "er"                      ║
        ║  • Light mode: generic cleanup (no entities yet)         ║
        ║  • Entity-aware mode: filter to entity sentences (if NER ║
        ║    has detected entities)                                ║
        ║  • LLM Polish: call Ollama for final cleanup             ║
        ║                                                            ║
        ║  NODE: denoise_text() in denoiser.py                     ║
        ║  BACKEND: Ollama (call_ollama)                           ║
        ║  OUTPUT STATE: denoised_text                             ║
        ║                                                            ║
        ╚════════════════════════════════════════════════════════════╝
                                     │
                                     │ denoised_text (cleaned)
                                     ▼
        ╔════════════════════════════════════════════════════════════╗
        ║                                                            ║
        ║  STAGE 2: CORRECT PHONETICS (Fix ASR/OCR Errors)         ║
        ║  ─────────────────────────────────────────────────────   ║
        ║  ECTD Approach (Entity-Centric Text Denoising):          ║
        ║                                                            ║
        ║  STEP A: Rough Entity Extraction                         ║
        ║    Input: denoised_text                                  ║
        ║    Output: ["farma corp", "knew deli", "A T one"]       ║
        ║    (AS-IS, no correction yet)                            ║
        ║                                                            ║
        ║  STEP B: Context-Based Correction                        ║
        ║    Input: rough entities + full text                     ║
        ║    Context clues:                                        ║
        ║      - "P H C" + "stock" → "PHC" (ticker)               ║
        ║      - "knew deli" + "london england" → "New Delhi"     ║
        ║      - "A T one" + medical terms → "AT1 receptor"       ║
        ║    Output: Fully corrected text                          ║
        ║                                                            ║
        ║  NODE: correct_phonetics() in phonetic_corrector.py      ║
        ║  BACKEND: Ollama (2x call_ollama for steps A & B)        ║
        ║  OUTPUT STATE: denoised_text (updated with corrections)  ║
        ║                                                            ║
        ╚════════════════════════════════════════════════════════════╝
                                     │
                                     │ denoised_text (phonetically corrected)
                                     ▼
        ╔════════════════════════════════════════════════════════════╗
        ║                                                            ║
        ║  STAGE 3: GENERATE CANDIDATES                            ║
        ║  ─────────────────────────────────────────────────────   ║
        ║  (Limited to REXEL, ECTD variants - not in this demo)    ║
        ║  • Generate entity pairs from text                       ║
        ║  • Coarse filtering: sentence proximity (window=3)      ║
        ║  • Output: Lists of candidate pairs                      ║
        ║                                                            ║
        ║  NODE: generate_candidates() in candidate_generator.py   ║
        ║  OUTPUT STATE: draft_triples (raw, unverified)           ║
        ║                                                            ║
        ╚════════════════════════════════════════════════════════════╝
                                     │
                                     │ draft_triples
                                     ▼
        ╔════════════════════════════════════════════════════════════╗
        ║                                                            ║
        ║  STAGE 4: EXTRACT RELATIONS                              ║
        ║  ─────────────────────────────────────────────────────   ║
        ║  (Not detail-focused but important for full pipeline)    ║
        ║  • Extract relations between entity pairs                ║
        ║  • Fine-grained filtering on relation plausibility      ║
        ║  • LLM-based extraction via Ollama                       ║
        ║                                                            ║
        ║  NODE: extract_relations() in extractor.py               ║
        ║  OUTPUT STATE: draft_triples (with relations)            ║
        ║                                                            ║
        ╚════════════════════════════════════════════════════════════╝
                                     │
                                     │ draft_triples (h, r, t candidates)
                                     ▼
        ╔════════════════════════════════════════════════════════════╗
        ║                                                            ║
        ║  🎯 STAGE 5: GRAPH JUDGE VERIFICATION                    ║
        ║  ─────────────────────────────────────────────────────   ║
        ║  KEY QUESTION: "Is this triple in the source text?"      ║
        ║                                                            ║
        ║  ┌──────────────────────────────────────────────────┐   ║
        ║  │ For each triple (h, r, t):                       │   ║
        ║  │                                                   │   ║
        ║  │ 1. Form Instruction:                             │   ║
        ║  │    "Is this true: {h} {r} {t}?"                 │   ║
        ║  │    Example: "Is this true: Doctor Smith works   │   ║
        ║  │             at Harvard?"                         │   ║
        ║  │                                                   │   ║
        ║  │ 2. Choose Backend (Fallback Chain):              │   ║
        ║  │    ┌─────────────────────────────────────────┐  │   ║
        ║  │    │ 1. Try BERT (fast, small)               │  │   ║
        ║  │    │    if available & JUDGE_BACKEND=bert    │  │   ║
        ║  │    │                                          │  │   ║
        ║  │    │ 2. Try LoRA (smart, slow)               │  │   ║
        ║  │    │    if available & JUDGE_BACKEND=lora    │  │   ║
        ║  │    │                                          │  │   ║
        ║  │    │ 3. DEFAULT: Ollama (always works)       │  │   ║
        ║  │    │    Local LLM, no GPU needed             │  │   ║
        ║  │    └─────────────────────────────────────────┘  │   ║
        ║  │                                                   │   ║
        ║  │ 3. Call Backend with Instruction + Context:      │   ║
        ║  │    Input:  "Is this true: h r t?"               │   ║
        ║  │            + full denoised text context          │   ║
        ║  │    Output: LLM response string                    │   ║
        ║  │                                                   │   ║
        ║  │ 4. Parse Response (Yes/No):                      │   ║
        ║  │    if "no" or "false" in first 100 chars:       │   ║
        ║  │        → REJECT (❌ not in source)               │   ║
        ║  │    else:                                         │   ║
        ║  │        → VERIFY (✅ in source)                   │   ║
        ║  │                                                   │   ║
        ║  │ 5. Repeat for all draft_triples                  │   ║
        ║  └──────────────────────────────────────────────────┘   ║
        ║                                                            ║
        ║  NODE: judge_triples() in judge.py                       ║
        ║  BACKEND: _judge_backend() with fallback logic           ║
        ║  OUTPUT STATE: verified_triples (filtered, evidence-based)║
        ║                                                            ║
        ╚════════════════════════════════════════════════════════════╝
                                     │
                                     │ verified_triples (subset of drafts)
                                     ▼
        ╔════════════════════════════════════════════════════════════╗
        ║                                                            ║
        ║  STAGE 6: STORE TO NEO4J                                 ║
        ║  ─────────────────────────────────────────────────────   ║
        ║  • Create nodes: entities (heads, tails)                 ║
        ║  • Create relationships: (h)--[r]--(t)                   ║
        ║  • Store metadata: extraction confidence, etc.           ║
        ║                                                            ║
        ║  NODE: store_to_neo4j() in neo4j_writer.py               ║
        ║  DATABASE: Neo4j (bolt://neo4j:password@localhost:7687)  ║
        ║  OUTPUT STATE: kg_summary (stats about storage)          ║
        ║                                                            ║
        ╚════════════════════════════════════════════════════════════╝
                                     │
                                     │ kg_summary {entities, relations}
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OUTPUT: KNOWLEDGE GRAPH DATABASE                          │
│                                                                             │
│                          Neo4j (Cypher queries)                             │
│                                                                             │
│  (Doctor Smith)-[WORKS_AT]->(Harvard University)                           │
│  (Harvard University)-[LOCATED_IN]->(Boston)                               │
│  (Doctor Smith)-[STUDIES]->(Malaria)                                       │
│  ...                                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Judge Backend Architecture

```
JUDGE BACKEND SELECTION & FALLBACK
═══════════════════════════════════════════════════════════════════════════════

┌──────────────────────────┐
│   Instruction + Context   │
│  "Is this true: h r t?"  │
└──────────────┬───────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   _judge_backend() Routing Logic                         │
│                                                                          │
│  backend = os.getenv("JUDGE_BACKEND", "ollama")                        │
│  fallback = os.getenv("JUDGE_FALLBACK_OLLAMA", "true")                 │
│                                                                          │
│  try:                                                                    │
│      if backend == "bert": ──────────────────────┐                     │
│      if backend == "lora": ──────────────────────┤ Try primary         │
│  except Exception:          ──────────────────────┘ backend             │
│      if not fallback: raise                                             │
│      # Fallback to Ollama                                               │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
               │
        ┌──────┴──────┬──────────────┬──────────────┐
        │             │              │              │
        ▼             ▼              ▼              ▼
    PRIMARY    FALLBACK 1       FALLBACK 2      DEFAULT
     BERT        LoRA (if       Ollama         Ollama
   (if set)     enabled)      (Fallback)   (Always works)
        │             │              │              │
        │             │              │              │


OPTION 1: BERT CLASSIFIER (Fast)
─────────────────────────────────
Input:  "Instruction: Is this true: h r t?
         Input: [context text]"

Model:  BERT-base fine-tuned for entailment/contradiction
        (from bert-base-classifier-rebel-sub)

Process: 1. Tokenize input
         2. Feed to BERT
         3. Get logits for 2 classes (false=0, true=1)
         4. Compute softmax probabilities
         5. Return argmax + confidence

Output: "Yes, it is true. (confidence: 0.95)"
        OR
        "No, it is not true. (confidence: 0.92)"

Pros:   ✅ Fast (single forward pass)
        ✅ Small model (<500MB)
        ✅ Deterministic (temp=0)

Cons:   ❌ Requires fine-tuned weights
        ❌ Binary only (no detailed explanation)
        ❌ May need GPU for speed


OPTION 2: LoRA + LLAMA2-7B (Smart but Slow)
──────────────────────────────────────────────
Input:  Instruction format (from GraphJudge prepare_KGCom.ipynb):
        "Below is an instruction that describes a task...
         ### Instruction: Is this true: h r t?
         ### Input: [context]
         ### Response:"

Model:  Llama2-7B base + LoRA adapter (llama2-7b-lora-genwiki-context)
        Fine-tuned on GraphJudge dataset

Process: 1. Tokenize prompt with system format
         2. Load base model + LoRA adapter
         3. Generate text with sampling (temp=0.7)
         4. Stop at EOS or max_tokens
         5. Return generated text after "### Response:"

Output: "Yes, the text explicitly states that Doctor Smith works at Harvard.
         This is mentioned in the first sentence."

Pros:   ✅ Generates explanations
        ✅ Fine-tuned on GraphJudge dataset
        ✅ Understands task format

Cons:   ❌ Slower (generates 50+ tokens)
        ❌ Requires GPU ideally
        ❌ Needs fine-tuned LoRA weights
        ❌ More memory (~20GB)


OPTION 3: OLLAMA LLAMA3.1:8B (Local & Balanced) ◄─── DEFAULT
────────────────────────────────────────────────────
Input:  Same GraphJudge instruction format:
        "Below is an instruction...
         ### Instruction: Is this true: h r t?
         ### Input: [context]
         ### Response:"

Model:  Llama3.1:8B (on-device, via Ollama)
        Open-source, no API key needed

Process: 1. Send prompt to Ollama HTTP endpoint
         2. Ollama runs inference locally
         3. Return generated response
         4. Parse for "no"/"false" in first 100 chars

Output: "No, this is not mentioned in the provided text."
        OR
        "Yes, the text states that..."

Pros:   ✅ LOCAL (no API, no internet)
        ✅ Works on CPU (slower but acceptable)
        ✅ No fine-tuning required
        ✅ ALWAYS AVAILABLE (default fallback)
        ✅ Open-source model
        ✅ Easy to use (pre-installed in project)

Cons:   ⚠️  Requires Ollama runtime
        ⚠️  Slightly slower than BERT
        ⚠️  Output includes occasional verbose explanation
        ⚠️  Non-deterministic (sampling-based)


FALLBACK DECISION TREE
───────────────────────────────────────────────────────────────

                    Start Judge Call
                           │
                           ▼
                   What's JUDGE_BACKEND?
                    /        │        \
                BERT?      LoRA?    Ollama?
                  │          │         │
                  ▼          ▼         │
            TRY BERT   TRY LoRA        │
             │     │     │     │       │
          Success? │  Success? │       │
             │     │     │     │       │
          YES  NO YES  NO    NO       │
             │     │     │     │       │
             └─[OK] └─┴──┴─────┴───┐  │
                                    ▼  ▼
                        Can fallback to Ollama?
                         /                \
                      YES                NO
                       │                  │
                       ▼                  ▼
                   call_ollama()      raise Exception
                       │                  │
                       ▼                  ▼
                   Response            Pipeline
                       │               Fails
                       ▼
                  Parse (yes/no)
```

---

## Phonetic Correction Flow (ECTD Step-by-Step)

```
RAW DENOISED TEXT
─────────────────
"...the trial begins in knew deli india...
 ...it targets the A T one receptor...
 ...P H C jumped fifteen percent..."


STEP A: Rough Entity Extraction
─────────────────────────────────

PROMPT:
  "List every proper noun in this text.
   The text may be misspelled - list them AS-IS.
   Return ONLY JSON array: []"

OLLAMA CALL:
  ├─ Input: Denoised text
  └─ Model: Llama3.1:8b

RESPONSE (JSON):
  ["knew deli", "A T one", "P H C"]

WHY AS-IS?
  • Captures phonetic clues
  • Doesn't require external knowledge
  • Preserves exact transcription artifacts


STEP B: Context-Based Correction
─────────────────────────────────

PROMPT:
  "Correct these potentially misspelled entities:
   {json.dumps(["knew deli", "A T one", "P H C"])}

   Rules:
   1. "knew deli" + "london england" nearby → geography
   2. "A T one" + medical terms → "AT1"
   3. "P H C" + "stock jumped" → ticker symbol

   Return full corrected text."

CONTEXT CLUES IN TEXT:
  Line 1: "...knew deli INDIA and LONDON ENGLAND..."
          └─ Geographic context → "New Delhi"

  Line 2: "...A T ONE receptor..."
          └─ Medical context → "AT1"

  Line 3: "...P H C jumped FIFTEEN PERCENT..."
          └─ Financial context (jumped %) → "PHC"

OLLAMA RESULT:
  "...the trial begins in New Delhi India...
   ...it targets the AT1 receptor...
   ...PHC jumped fifteen percent..."


INFERENCE STEPS USED:
──────────────────────
1. Domain Detection
   • Medical terms? Domain = healthcare
   • Stock terms? Domain = finance
   • Place names? Domain = geography

2. Phonetic Pattern Recognition
   • "knew" sounds like "new"
   • "P H C" = spelled-out acronym
   • "A T" = spoken numbers/abbreviations

3. Context Consistency
   • "New Delhi" is India's capital (consistent)
   • "AT1" is real receptor name (medical term)
   • "PHC" matches stock ticker format
```

---

## Graph Judge Verification Examples

```
EXAMPLE 1: CORRECT TRIPLE (Should Verify ✅)
═════════════════════════════════════════════════════════════

CONTEXT TEXT:
  "Doctor Smith works at Harvard University. He studies malaria."

EXTRACTED TRIPLE:
  head = "Doctor Smith"
  relation = "works at"
  tail = "Harvard University"

Instruction Formed:
  "Is this true: Doctor Smith works at Harvard University?"

Judge Backend (Ollama) Receives:
  Instruction: "Is this true: Doctor Smith works at Harvard University?"
  Input: "Doctor Smith works at Harvard University. He studies malaria."

Judge Response:
  "Yes, the text explicitly states that Doctor Smith works at Harvard
   University in the first sentence."

Parse Response Window (first 100 chars):
  "Yes, the text explicitly states that Doctor Smith works at Harvard"
  └─ Does NOT contain "no" or "false"

Result: ✅ VERIFIED (triple kept)


EXAMPLE 2: HALLUCINATED TRIPLE (Should Reject ❌)
═════════════════════════════════════════════════════════════

CONTEXT TEXT:
  "Doctor Smith works at Harvard University. He studies malaria."

EXTRACTED TRIPLE:
  head = "Doctor Smith"
  relation = "studies"
  tail = "Medicine"

Instruction Formed:
  "Is this true: Doctor Smith studies Medicine?"

Judge Backend (Ollama) Receives:
  Instruction: "Is this true: Doctor Smith studies Medicine?"
  Input: "Doctor Smith works at Harvard University. He studies malaria."

Judge Response:
  "No, the text says he studies malaria specifically, not Medicine
   in general. He's a malaria researcher, not a medicine student."

Parse Response Window (first 100 chars):
  "No, the text says he studies malaria specifically, not Medicine"
  └─ CONTAINS "no" at position 0

Result: ❌ REJECTED (triple removed)


EXAMPLE 3: NOT IN SOURCE TEXT (Should Reject ❌)
═════════════════════════════════════════════════════════════

CONTEXT TEXT:
  "Doctor Smith works at Harvard University. He studies malaria."

EXTRACTED TRIPLE:
  head = "Harvard University"
  relation = "is located in"
  tail = "Boston"

Instruction Formed:
  "Is this true: Harvard University is located in Boston?"

Judge Backend (Ollama) Receives:
  Instruction: "Is this true: Harvard University is located in Boston?"
  Input: "Doctor Smith works at Harvard University. He studies malaria."

Judge Response:
  "No, this information is not provided in the given text. While
   Harvard exists, its location is not mentioned in the source document."

Parse Response Window (first 100 chars):
  "No, this information is not provided in the given text. While"
  └─ CONTAINS "no" at position 0

Result: ❌ REJECTED (not grounded in source)


BATCH VERIFICATION EXAMPLE
═════════════════════════════════════════════════════════════

INPUT DRAFT TRIPLES: 5
  1. (Doctor Smith, works at, Harvard)
  2. (Doctor Smith, studies, Malaria)
  3. (Harvard, is in, Boston)
  4. (Doctor Smith, teaches, Psychology)
  5. (Harvard, founded in, 1636)

FOR EACH TRIPLE:
  Triple 1: Check → Contains "works" "Harvard" in text → ✅ VERIFIED
  Triple 2: Check → Contains "studies" "malaria" in text → ✅ VERIFIED
  Triple 3: Check → "Boston" not mentioned → ❌ REJECTED
  Triple 4: Check → "Psychology" not mentioned, says "malaria" → ❌ REJECTED
  Triple 5: Check → "1636" not in text → ❌ REJECTED

OUTPUT VERIFIED TRIPLES: 2
  ├─ (Doctor Smith, works at, Harvard)
  └─ (Doctor Smith, studies, Malaria)

VERIFICATION STATS:
  Total drafted: 5
  Successfully verified: 2 (40%)
  Rejected: 3 (60%) - prevented hallucinations!
```

---

## State Diagram (Data Flow Through Pipeline)

```
INITIAL STATE
─────────────────────────────────────────────────────────────────
{
  "raw_text": "uh doctor ... FLASH SALE ...",
  "denoised_text": "",
  "mentions": [],
  "entity_clusters": [],
  "draft_triples": [],
  "verified_triples": [],
  "kg_summary": {},
  "error": ""
}

AFTER DENOISE
─────────────────────────────────────────────────────────────────
{
  "raw_text": "uh doctor ...",
  "denoised_text": "Doctor Smith works at Harvard University.",
  "mentions": [],
  "entity_clusters": [],
  "draft_triples": [],
  "verified_triples": [],
  "kg_summary": {},
  "error": ""
}

AFTER PHONETIC CORRECTION
─────────────────────────────────────────────────────────────────
{
  "raw_text": "uh doctor ...",
  "denoised_text": "Doctor Smith works at Harvard University.",
  ↑ (updated with corrected entities)
  "mentions": [],
  "entity_clusters": [],
  "draft_triples": [],
  "verified_triples": [],
  "kg_summary": {},
  "error": ""
}

AFTER GENERATE CANDIDATES (implied, by extractor)
─────────────────────────────────────────────────────────────────
{
  "raw_text": "uh doctor ...",
  "denoised_text": "Doctor Smith works at Harvard University.",
  "mentions": [
    {"text": "Doctor Smith", "type": "PERSON", "offset": 0},
    {"text": "Harvard University", "type": "ORG", "offset": 20}
  ],
  "entity_clusters": [
    {"canonical": "Doctor Smith", "variants": ["doctor smith"]},
    {"canonical": "Harvard University", "variants": ["harvard"]}
  ],
  "draft_triples": [
    {"head": "Doctor Smith", "relation": "works at", "tail": "Harvard"}
  ],
  "verified_triples": [],
  "kg_summary": {},
  "error": ""
}

AFTER JUDGE (Final State Before Storage)
─────────────────────────────────────────────────────────────────
{
  "raw_text": "uh doctor ...",
  "denoised_text": "Doctor Smith works at Harvard University.",
  "mentions": [...],
  "entity_clusters": [...],
  "draft_triples": [
    {"head": "Doctor Smith", "relation": "works at", "tail": "Harvard"}
  ],
  "verified_triples": [
    {"head": "Doctor Smith", "relation": "works at", "tail": "Harvard"}
    ↑ Only verified triples kept
  ],
  "kg_summary": {},
  "error": ""
}

AFTER NEO4J STORAGE (Final Output State)
─────────────────────────────────────────────────────────────────
{
  "raw_text": "uh doctor ...",
  "denoised_text": "Doctor Smith works at Harvard University.",
  "mentions": [...],
  "entity_clusters": [...],
  "draft_triples": [
    {"head": "Doctor Smith", "relation": "works at", "tail": "Harvard"}
  ],
  "verified_triples": [
    {"head": "Doctor Smith", "relation": "works at", "tail": "Harvard"}
  ],
  "kg_summary": {
    "entities_stored": 2,
    "relationships_stored": 1,
    "nodes_created": ["Doctor_Smith", "Harvard_University"],
    "edges_created": ["WORKS_AT"]
  },
  "error": ""
}
```

---

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LANGGRAPH ORCHESTRATION                           │
│                                                                             │
│  app = workflow.compile()                                                  │
│  result = app.invoke({"raw_text": input_text})                            │
│                                                                             │
│  LangGraph manages state passing between nodes automatically               │
│                                                                             │
│  ┌──────────┐        ┌──────────┐        ┌──────────┐        ┌──────────┐ │
│  │  NODE 1  │──────▶ │  NODE 2  │──────▶ │  NODE 3  │──────▶ │  NODE 4  │ │
│  │ DENOISE  │ State  │PHONETIC  │ State  │ CANDS    │ State  │ JUDGE    │ │
│  └──────────┘        └──────────┘        └──────────┘        └──────────┘ │
│       │                   │                   │                   │        │
│       └─ reads raw_text   │                   │                   │        │
│       └─ writes           │                   │                   │        │
│          denoised_text    │                   │                   │        │
│                           │                   │                   │        │
│                           └─ reads denoised   │                   │        │
│                           └─ writes updated   │                   │        │
│                              denoised_text    │                   │        │
│                                               │                   │        │
│                                               └─ reads denoised   │        │
│                                               └─ writes           │        │
│                                                  draft_triples    │        │
│                                                                   │        │
│                                                   ┌───────────────┘        │
│                                                   │                        │
│                                                   └─ reads draft_triples   │
│                                                   └─ reads denoised_text  │
│                                                   └─ calls _judge_backend │
│                                                   └─ writes verified_tri. │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        └──────────────────────────┐
                                   ▼
              ┌────────────────────────────────────┐
              │    EXTERNAL BACKENDS / DATABASES   │
              ├────────────────────────────────────┤
              │                                    │
              │  Ollama HTTP Endpoint              │
              │  (localhost:11434)                 │
              │  - denoise_text()                  │
              │  - correct_phonetics() [steps A&B] │
              │  - judge_triples()                 │
              │  - extract_relations()             │
              │                                    │
              │  Neo4j Database                    │
              │  (localhost:7687)                  │
              │  - store_to_neo4j()                │
              │                                    │
              │  Optional: BERT/LoRA Models        │
              │  - judge_triples() [alternative]   │
              │                                    │
              └────────────────────────────────────┘
```

---

## Error Handling Flow

```
CONDITIONAL EDGE: should_continue()
───────────────────────────────────────────────────────────────

After judge_triples() completes, checks:

        Does state have error?
           ├─ YES → "handle_error" → END
           └─ NO  → Continue checking...

        Is denoised_text empty?
           ├─ YES → "handle_error" → END
           └─ NO  → Continue checking...

        Is verified_triples None?
           ├─ YES → "handle_error" → END
           └─ NO  → Proceed...

        All checks pass?
           └─ YES → "store_to_neo4j" → END


ERROR HANDLING
───────────────────────────────────────────────────────────────

If any node raises exception:
  1. State["error"] = exception message
  2. Conditional edge routes to "handle_error"
  3. handle_error() logs error and stops gracefully
  4. Pipeline doesn't crash, exits cleanly

Example:
  Judge fails: connection refused to Ollama
  → Exception caught in judge_triples()
  → state["error"] = "Ollama not running"
  → Routed to handle_error
  → Pipeline stops, user sees error message
```

---

## Summary: The Three Key Stages Visually

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: DENOISE                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input:  "uh doctor smith he uh works at harvard ... BUY NOW 50%!"        │
│                                                                             │
│  Process:                                                                   │
│    [Light]       Remove fillers, fix punctuation                          │
│    OR                                                                       │
│    [Entity-aware] Keep sentences with detected entities,                  │
│                  remove ads, Polish with LLM                              │
│                                                                             │
│  Output: "Doctor Smith works at Harvard University"                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: PHONETIC CORRECTION (ECTD)                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input:  "The trial begins in knew deli india... A T one receptor...    │
│           P H C jumped 15%..."                                             │
│                                                                             │
│  Step A: Extract rough entities                                            │
│    ["knew deli", "A T one", "P H C"]                                      │
│    (as-is, with errors)                                                    │
│                                                                             │
│  Step B: Use context to correct                                            │
│    - "knew deli" + "london england" → "New Delhi"                         │
│    - "A T one" + medical → "AT1 receptor"                                 │
│    - "P H C" + "jumped 15%" → "PHC" (stock ticker)                        │
│                                                                             │
│  Output: "The trial begins in New Delhi India... AT1 receptor...        │
│           PHC jumped 15%..."                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: GRAPH JUDGE VERIFICATION                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input:  Draft Triples                                                    │
│    (Doctor Smith, works at, Harvard)        ← Check against text          │
│    (Harvard, manufactures, Medicine)        ← Check against text          │
│    (Doctor Smith, studies, Malaria)         ← Check against text          │
│                                                                             │
│  For Each Triple:                                                          │
│    1. Instruction: "Is this true: {h} {r} {t}?"                          │
│    2. Backend: BERT/LoRA/Ollama (with fallback)                          │
│    3. Response: "Yes..." / "No..."                                        │
│    4. Parse: "no"/"false" in response? REJECT : VERIFY                   │
│                                                                             │
│  Output: Verified Triples (only those in source text)                     │
│    ✅ (Doctor Smith, works at, Harvard)                                   │
│    ✅ (Doctor Smith, studies, Malaria)                                    │
│    ❌ (Harvard, manufactures, Medicine) - NOT in text!                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
