#!/usr/bin/env python3
"""
================================================================================
  KNOWLEDGE GRAPH EXTRACTION PIPELINE - DEMONSTRATION & EXPLANATION
================================================================================

PROJECT OVERVIEW:
This pipeline extracts structured knowledge (triples: head-relation-tail) from
unstructured text documents. It handles noisy input (ASR/OCR errors) and verifies
extracted triples against the source text.

PIPELINE STAGES:
    1. DENOISE TEXT        — Clean raw input (remove ads, fix transcription errors)
    2. CORRECT PHONETICS   — Fix ASR/OCR errors using entity-centric context
    3. GENERATE CANDIDATES — Propose entity pairs as relation candidates
    4. JOINT REFINE        — Perform coarse & fine filtering on candidates
    5. GRAPH JUDGE         — Verify each triple: "Is this actually in the text?"
    6. NEO4J STORAGE       — Store verified triples in knowledge graph

FOCUS AREAS (This Demo):
    A. DENOISING          → entity-aware text cleaning
    B. PHONETIC CORRECTION → ECTD approach (rough extraction → context correction)
    C. GRAPH JUDGE         → GraphJudge-style triple verification

================================================================================
"""

import json
import sys
from typing import List, Dict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from kg_pipeline.utils.ollama import call_ollama
from kg_pipeline.state import KGState


# ============================================================================
# DEMO DATA & UTILITIES
# ============================================================================

def print_section(title: str):
    """Pretty-print section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subsection(title: str):
    """Pretty-print subsection header."""
    print(f"\n  >>> {title}")
    print("  " + "-" * 76)


def print_highlight(label: str, content: str, max_chars: int = 200):
    """Print labeled content with line wrapping."""
    prefix = f"  📌 {label}: "
    if len(content) > max_chars:
        content = content[:max_chars] + "..."
    print(f"{prefix}\n     {content.replace(chr(10), chr(10) + '     ')}")


class DemoData:
    """Demo dataset with realistic noisy text examples."""

    # Example 1: Medical article with speech errors, ads, phonetic mistakes
    MEDICAL_EXAMPLE = """
uh so basically doctor evelyn reed she is uh a cardiologist from the mayo clinic
in rochester minnesota she was presenting her research on this new drug called
cardia care um its made by farma corp i think thats how you spell it
the drug showed like a thirty percent reduction in heart related incidents
or something like that

FLASH SALE THIS WEEK ONLY GET 50 PERCENT OFF ALL SUPPLEMENTS VISIT WWW DOT HEALTHDEALS DOT COM

so the study had over five thousand patients and it was published in the
new england journal of medicine on may first twenty twenty four
farma corp stock which trades as P H C jumped fifteen percent after the announcement

doctor reed also mentioned that cardia care works differently from
existing beta blockers she said it targets a specific receptor called
the A T one receptor which existing drugs dont focus on

buy one get one free on all vitamins and minerals limited time offer

the mayo clinic and farma corp are now planning a phase three trial
which will be conducted across twelve hospitals in the united states
doctor reed said the results were very promising
the trial is expected to begin in knew deli india and london england next year
"""

    # Example 2: Technology company news (simpler for first demo)
    TECH_EXAMPLE = """
so apple announced new iphone features in san francisco yesterday
tim cook the ceo presented improvements to the a seventeen chip
microsoft is also competing with their new cortex processor
both companies are focused on a i development for smartphones
"""


# ============================================================================
# STAGE A: DENOISING TEXT
# ============================================================================

def explain_denoising():
    """Explain the denoising stage."""
    print_section("STAGE A: DENOISING TEXT (Entity-Aware)")

    print("""
WHAT IS IT?
  Text denoising removes noise while preserving facts:
  • Advertisements and promotional content
  • Repeated filler speech (uh, um, er, etc.)
  • False starts and incomplete sentences
  • Irrelevant promotional sections

TWO MODES:
  1. LIGHT CLEANING: If no entities detected yet
     → Generic cleanup: remove fillers, fix punctuation

  2. ENTITY-AWARE FILTERING: Once entities are known
     → Keep ONLY sentences mentioning detected entities
     → Remove all irrelevant sections
     → Preserve factual relationships between entities

WHY ENTITY-AWARE?
  By the time we denoise, we've already detected entities (persons, orgs, drugs).
  So we filter the text to ONLY keep sentences relevant to those entities.
  This removes ads, off-topic content, and noise while preserving what matters.

EXAMPLE FLOW:
  Raw Text    →  [Detect Entities]  →  Entity-specific Cleaning  →  Clean Text
    (noisy)       (mayo clinic,              (keep sentences about      (facts-only)
                   farma corp,               these entities)
                   cardia care)
""")


def demo_denoise_light_mode():
    """Demo light denoising (without entities)."""
    print_subsection("DEMO 1: Light Mode (No Entity Filter)")

    simple_text = """
uh so like you know apple is making phones in california
um the ceo tim cook he announced new features
er basically they have a new chip called a seventeen
the a seventeen is faster than before
"""

    print_highlight("INPUT (Noisy)", simple_text)

    prompt = f"""
Clean the following text by:
- Removing advertisements, promotional content
- Removing repeated filler speech (uh, um, etc.)
- Fixing sentence structure slightly

DO NOT remove factual content.
DO NOT summarize.

Return only cleaned text.

Text:
{simple_text}
"""

    print("\n  🔄 Calling Ollama for light denoising...")
    cleaned = call_ollama(prompt)
    print_highlight("OUTPUT (Cleaned)", cleaned)


def demo_denoise_entity_aware():
    """Demo entity-aware denoising."""
    print_subsection("DEMO 2: Entity-Aware Mode (Medical Example)")

    # Simulate that we detected these entities
    detected_entities = ["doctor evelyn reed", "mayo clinic", "farma corp", "cardia care"]

    print_highlight("DETECTED ENTITIES", str(detected_entities))
    print_highlight("INPUT TEXT", DemoData.MEDICAL_EXAMPLE, max_chars=300)

    # Step 1: Sentence filtering
    sentences = DemoData.MEDICAL_EXAMPLE.split(".")
    filtered_sentences = [
        s for s in sentences
        if any(entity.lower() in s.lower() for entity in detected_entities)
    ]

    filtered_text = ". ".join(filtered_sentences)
    print("\n  📋 Step 1: Sentence Filtering (keep only entity-relevant sentences)")
    print(f"     Original: {len(sentences)} sentences")
    print(f"     Filtered: {len(filtered_sentences)} sentences (removed ads, filler)")

    # Step 2: LLM refinement
    entity_list = ", ".join(detected_entities)
    prompt = f"""
You are cleaning a document for knowledge graph extraction.

Entities detected in the document:
{entity_list}

Your task:
1. Keep ONLY sentences relevant to these entities
2. Remove advertisements, promotional text, irrelevant sections
3. Preserve all factual relationships between entities
4. Do NOT summarize or paraphrase too much
5. Keep original meaning intact

Return ONLY cleaned text.

Document:
{filtered_text}
"""

    print("\n  🔄 Step 2: LLM Refinement (Polish the filtered text)")
    cleaned = call_ollama(prompt)
    print_highlight("OUTPUT (Entity-Aware Cleaned)", cleaned, max_chars=400)


# ============================================================================
# STAGE B: PHONETIC CORRECTION
# ============================================================================

def explain_phonetic_correction():
    """Explain the phonetic correction stage."""
    print_section("STAGE B: PHONETIC CORRECTION (ECTD Approach)")

    print("""
WHAT IS IT?
  Phonetic correction fixes ASR (speech-to-text) and OCR (image-to-text) errors.
  Uses entity-centric context to infer correct spellings.

WHY TWO STEPS?

  STEP A — Rough Entity Extraction:
    → Extract all proper nouns from noisy text AS-IS (don't correct yet)
    → Get: ["farma corp", "knew deli", "A T one receptor", ...]
    → These may be misspelled but we have the phonetic anchors

  STEP B — Context-Based Correction:
    → Use sentence context to infer correct spelling
    → "knew deli" + context "london england" → "New Delhi"
    → "P H C" spelled out + context "stock traded farma corp" → "PHC"
    → "cardia care" medical context → stays "Cardia Care"

INTUITION:
  We use surrounding words (stock, drug, london, etc.) to figure out:
  - Is this a company name? A place? A drug?
  - What domain is it in?
  - What does it sound like when read aloud?

RESULT: Automatically corrected entities without manual fix lists!
""")


def demo_phonetic_correction():
    """Demo phonetic correction."""
    print_subsection("DEMO 3: Phonetic Correction (ECTD)")

    denoised_text = DemoData.MEDICAL_EXAMPLE

    # STEP A: Rough entity extraction
    print("\n  📌 STEP A: Rough Entity Extraction (as-is, no correction)")

    step_a_prompt = f"""Read the following text and list every proper noun you can find.
This includes: people names, company names, drug names, place names, organization names.
The text may contain speech transcription errors so spellings may be wrong.
List them exactly as they appear in the text, do not correct them yet.

Return ONLY a JSON array of strings. No explanation. No markdown.
Example: ["farma corp", "doctor evelyn reed", "knew deli", "cardia care"]

Text:
{denoised_text}"""

    print("  🔄 Calling Ollama to extract rough entities...")
    raw_entities_response = call_ollama(step_a_prompt)

    try:
        raw_entities = json.loads(raw_entities_response)
    except json.JSONDecodeError:
        print(f"  ⚠️  JSON parse failed, showing raw response:")
        print(f"     {raw_entities_response[:200]}...")
        raw_entities = []

    print_highlight("RAW ENTITIES (As-Is)", json.dumps(raw_entities, indent=2), max_chars=500)

    # STEP B: Context-based correction
    print("\n  📌 STEP B: Context-Based Correction")

    step_b_prompt = f"""You are correcting a speech transcription that contains phonetic errors.

I found these potential entity names in the text (they may be misspelled):
{json.dumps(raw_entities, indent=2)}

For each misspelled entity, use the surrounding context in the text to figure out
what the correct spelling should be.

Rules:
1. Use context clues — what words appear near the entity? What domain is the text about?
2. If an entity appears near words like "stock", "traded", "drug", "developed"
   it is likely a company or product name — correct it to proper capitalization and spelling
3. If an entity sounds like a real place name when read aloud, correct it
   Example: "knew deli" sounds like "New Delhi" → correct it
   Example: "P H C" spelled out letter by letter → correct to "PHC"
   Example: "A T one receptor" → "AT1 receptor"
4. Do NOT invent new facts or change the meaning
5. Return the full corrected text only, no explanation

Original text:
{denoised_text}"""

    print("  🔄 Calling Ollama to correct phonetic errors...")
    corrected_text = call_ollama(step_b_prompt)
    print_highlight("CORRECTED TEXT", corrected_text, max_chars=400)


# ============================================================================
# STAGE C: GRAPH JUDGE VERIFICATION
# ============================================================================

def explain_graphjudge():
    """Explain the GraphJudge verification stage."""
    print_section("STAGE C: GRAPH JUDGE - TRIPLE VERIFICATION")

    print("""
WHAT IS IT?
  Given an extracted triple (head-relation-tail), GraphJudge answers:
  "Is this triple actually supported by the source text?"

HOW IT WORKS:
  1. EXTRACT CANDIDATES from text
     Example: (Mayo Clinic, is-associated-with, Cardia Care)

  2. FORM INSTRUCTION
     Format: "Is this true: {head} {relation} {tail}?"
     Example: "Is this true: Mayo Clinic is-associated-with Cardia Care?"

  3. JUDGE THE TRIPLE
     Pass instruction + context text to backend (BERT/LoRA/Ollama)
     Backend generates a response string

  4. PARSE RESPONSE
     If response contains "no" or "false" in first 100 chars → REJECT
     Otherwise → ACCEPT

MULTIPLE BACKENDS (With Fallback):

  🔹 BERT Classifier (Fast, Small):
     Input:  "Instruction: Is this true: X Y Z? Input: context..."
     Output: (label=0 or 1, confidence score)
     → Binary classification

  🔹 LoRA+Llama2 (Fine-tuned, Smart):
     Input:  GraphJudge instruction format (from prepare_KGCom.ipynb)
     Output: "Yes, it is..." or "No, it is not..."
     → Generates natural language response

  🔹 Ollama Llama3.1 (Local LLM, Default):
     Input:  Same GraphJudge instruction format
     Output: Yes/No response from local model
     → No GPU required, runs on CPU if needed
     → FALLBACK if BERT/LoRA unavailable

FALLBACK CHAIN:
  1. Try BERT (if available, configured)
  2. Try LoRA (if available, configured)
  3. **DEFAULT: Ollama** (local, always available)

ADVANTAGES:
  ✅ Verifies extractedtriples against source (evidence-based)
  ✅ Filters hallucinations & guesses
  ✅ Supports multiple backends (choose speed/accuracy)
  ✅ Graceful fallback (local LLM always works)
""")


def demo_graphjudge_instruction_generation():
    """Demo how GraphJudge instructions are generated."""
    print_subsection("DEMO 4a: GraphJudge Instruction Generation")

    print("""
  EXAMPLE EXTRACTED TRIPLES (Before Judgment):
""")

    # Simulated extracted triples
    example_triples = [
        {"head": "doctor evelyn reed", "relation": "works at", "tail": "mayo clinic"},
        {"head": "mayo clinic", "relation": "conducted research on", "tail": "cardia care"},
        {"head": "farma corp", "relation": "manufactures", "tail": "cardia care"},
        {"head": "cardia care", "relation": "reduces", "tail": "heart incidents"},
        {"head": "cardia care", "relation": "targets", "tail": "AT1 receptor"},
    ]

    for i, triple in enumerate(example_triples, 1):
        h = triple["head"]
        r = triple["relation"]
        t = triple["tail"]
        print(f"    {i}. {h} → {r} → {t}")

    print_subsection("DEMO 4b: Converting Triples to Instructions")

    print("""
  Each triple is converted to a single instruction:
  Format: "Is this true: {head} {relation} {tail}?"

""")

    for i, triple in enumerate(example_triples[:2], 1):
        h = triple["head"]
        r = triple["relation"]
        t = triple["tail"]
        instruction = f"Is this true: {h} {r} {t}?"
        print(f"    Triple {i}:")
        print(f"      {instruction}")

    print("\n    ...")


def demo_graphjudge_verification():
    """Demo the actual judgment process."""
    print_subsection("DEMO 4c: Full Judgment Process (Ollama Backend)")

    # Example triples
    test_triples = [
        {
            "head": "doctor evelyn reed",
            "relation": "works at",
            "tail": "mayo clinic",
            "description": "Should be TRUE (explicitly in text)"
        },
        {
            "head": "farma corp",
            "relation": "manufactures",
            "tail": "cardia care",
            "description": "Should be TRUE (explicitly in text)"
        },
        {
            "head": "apple",
            "relation": "manufactures",
            "tail": "cardia care",
            "description": "Should be FALSE (not in text)"
        }
    ]

    # Context text from the medical example (cleaned)
    context = """
doctor evelyn reed is a cardiologist from the mayo clinic in rochester minnesota.
she was presenting her research on a new drug called cardia care.
it is made by farma corp.
the drug showed a thirty percent reduction in heart related incidents.
the study had over five thousand patients.
cardia care works differently from existing beta blockers.
it targets a specific receptor called the AT1 receptor.
mayo clinic and farma corp are planning a phase three trial.
"""

    print("  📌 CONTEXT TEXT (source document):")
    print(f"     {context[:150]}...")

    print("\n  📌 TEST TRIPLES & JUDGMENTS:")

    verified_count = 0
    rejected_count = 0

    for i, triple in enumerate(test_triples, 1):
        h = triple["head"]
        r = triple["relation"]
        t = triple["tail"]
        instruction = f"Is this true: {h} {r} {t}?"

        print(f"\n    Triple {i}: {h} → {r} → {t}")
        print(f"      Expected: {triple['description']}")
        print(f"      Instruction: {instruction}")

        # Build GraphJudge prompt
        graphjudge_prompt = f"""
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{context}
### Response:
"""

        print(f"      🔄 Calling Ollama judge backend...")
        response = call_ollama(graphjudge_prompt)

        # Parse response (any "no" or "false" in first 100 chars = FALSE)
        window = response.strip().lower()[:100]
        is_true = not ("no" in window or "false" in window)
        verdict = "✅ VERIFIED" if is_true else "❌ REJECTED"

        if is_true:
            verified_count += 1
        else:
            rejected_count += 1

        print(f"      Response: {response[:80]}...")
        print(f"      {verdict}")

    print(f"\n  📊 Summary: {verified_count} verified, {rejected_count} rejected")


def demo_graphjudge_backend_selection():
    """Demo how the backend selection works."""
    print_subsection("DEMO 4d: Backend Selection & Fallback Mechanism")

    print("""
  BACKEND SELECTION LOGIC (from judge.py):

  def _judge_backend(instruction: str, context_text: str) -> str:
      backend = os.getenv("JUDGE_BACKEND", "ollama").strip().lower()
      fallback = os.getenv("JUDGE_FALLBACK_OLLAMA", "true").strip().lower() == "true"

      try:
          if backend == "bert":
              return _judge_with_bert(instruction, context_text)
          if backend == "lora":
              return _judge_with_lora(instruction, context_text)
      except Exception as exc:
          if not fallback:
              raise
          print(f"{backend.upper()} backend unavailable; falling back to Ollama.")

      prompt = _build_graphjudge_prompt(instruction, context_text)
      return call_ollama(prompt)  # ← Fallback to Ollama

  ENVIRONMENT VARIABLES (.env):
    JUDGE_BACKEND=ollama              # Default backend
    JUDGE_FALLBACK_OLLAMA=true        # Enable fallback
    JUDGE_BERT_WEIGHTS=models/bert... # Optional BERT path
    JUDGE_LORA_BASE_MODEL=models/...  # Optional LoRA base
    JUDGE_LORA_WEIGHTS=models/...     # Optional LoRA adapter

  FALLBACK BEHAVIOR:
    1. Try to use configured backend (BERT/LoRA)
    2. If loading fails → catch exception
    3. If fallback=true → use Ollama instead
    4. If fallback=false → raise exception (fail hard)

  YOUR CURRENT CONFIG:
    ✅ Backend: Ollama (default, fast, local)
    ✅ Fallback: Enabled (robust)
    → No BERT/LoRA models required to run!
""")


# ============================================================================
# FULL PIPELINE DEMO
# ============================================================================

def demo_full_pipeline():
    """Demo a simplified version of the full pipeline."""
    print_section("FULL PIPELINE WALKTHROUGH")

    print("""
  PIPELINE STATE (TypedDict):
    {
      raw_text: str,              # Input (noisy)
      denoised_text: str,         # After stage A
      mentions: List[Dict],       # Detected entities
      entity_clusters: List,      # Coreference resolved
      draft_triples: List[Dict],  # Extracted (unverified)
      verified_triples: List,     # After judgment
      kg_summary: Dict,           # Neo4j storage summary
      error: str                  # Error message if any
    }

  PIPELINE FLOW:

    raw_text (noisy speech/OCR)
        ↓
    [1. DENOISE TEXT] ← Remove ads, fillers
        ↓ denoised_text
    [2. CORRECT PHONETICS] ← Fix ASR errors (farma → Pharma, etc)
        ↓ denoised_text (updated)
    [3. GENERATE CANDIDATES] ← Find entity pairs
        ↓ draft_triples (unverified)
    [4. JUDGE TRIPLES] ← GraphJudge: "Is this in the text?"
        ↓ verified_triples (filtered)
    [5. STORE TO NEO4J] ← Save knowledge graph
        ↓ kg_summary

  KEY INSIGHT:
    Each stage reads state["key"] and writes state["key"] = output
    LangGraph orchestrates the edges between stages
    Conditional routing after judge (if error → handle_error → END)
""")

    # Simulate simple pipeline execution
    print("\n  📋 EXAMPLE STATE TRANSITIONS:")

    state = {
        "raw_text": "uh so doctor smith works at harvard he studies malaria",
        "denoised_text": "",
        "mentions": [],
        "draft_triples": [],
        "verified_triples": [],
    }

    print("\n    1️⃣  INITIAL STATE (raw_text):")
    print(f"       raw_text = '{state['raw_text'][:60]}...'")

    print("\n    2️⃣  AFTER DENOISE:")
    state["denoised_text"] = "doctor smith works at harvard he studies malaria"
    print(f"       denoised_text = '{state['denoised_text']}'")

    print("\n    3️⃣  AFTER PHONETIC CORRECTION:")
    state["denoised_text"] = "doctor smith works at harvard university he studies malaria"
    print(f"       denoised_text = '{state['denoised_text']}'")

    print("\n    4️⃣  AFTER CANDIDATE GENERATION:")
    state["draft_triples"] = [
        {"head": "doctor smith", "relation": "works at", "tail": "harvard"},
        {"head": "doctor smith", "relation": "studies", "tail": "malaria"},
    ]
    print(f"       draft_triples = {state['draft_triples']}")

    print("\n    5️⃣  AFTER JUDGE:")
    # Simulate judgment (both accepted for this example)
    state["verified_triples"] = state["draft_triples"]
    print(f"       verified_triples = {state['verified_triples']}")

    print("\n    6️⃣  AFTER NEO4J STORAGE:")
    state["kg_summary"] = {
        "entities_stored": 3,
        "relations_stored": 2,
        "nodes_created": ["doctor_smith", "harvard", "malaria"],
    }
    print(f"       kg_summary = {state['kg_summary']}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all demos."""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                        KG PIPELINE DEMONSTRATION                          ║
║                   Denoising | Phonetic Correction | Judge                 ║
║                            (Interactive Tutorial)                         ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

    # Check Ollama
    from kg_pipeline.utils.ollama import check_ollama_server
    if not check_ollama_server():
        print("""
⚠️  OLLAMA SERVER NOT RUNNING
    Please start it with: ollama serve
    (in another terminal, from the project directory)

    Continuing with explanations (some demos will be skipped)...
""")
        run_explanations_only = True
    else:
        print("✅ Ollama server is running!\n")
        run_explanations_only = False

    # Part 1: Denoising
    explain_denoising()
    if not run_explanations_only:
        input("\n  Press Enter to see light denoising demo...")
        demo_denoise_light_mode()

        input("\n  Press Enter to see entity-aware denoising demo...")
        demo_denoise_entity_aware()

    # Part 2: Phonetic Correction
    explain_phonetic_correction()
    if not run_explanations_only:
        input("\n  Press Enter to see phonetic correction demo...")
        demo_phonetic_correction()

    # Part 3: GraphJudge
    explain_graphjudge()

    if not run_explanations_only:
        input("\n  Press Enter to see instruction generation...")
        demo_graphjudge_instruction_generation()

        input("\n  Press Enter to see verification process...")
        demo_graphjudge_verification()

    demo_graphjudge_backend_selection()

    # Part 4: Full Pipeline
    demo_full_pipeline()

    # Conclusion
    print_section("NEXT STEPS")
    print("""
  1. RUN THE FULL PIPELINE:
     $ python3 -m kg_pipeline.main

     This will execute all stages on the hardcoded test text in main.py:
     ✓ Denoise the medical article
     ✓ Correct phonetic errors (knew deli → New Delhi, etc)
     ✓ Generate entity pairs as candidates
     ✓ Extract relations from candidates
     ✓ Judge each triple with Ollama backend
     ✓ Store verified triples to Neo4j

  2. EXPLORE THE CODEBASE:
     kg_pipeline/nodes/denoiser.py          ← Entity-aware text cleaning
     kg_pipeline/nodes/phonetic_corrector.py ← ECTD (rough + context)
     kg_pipeline/nodes/judge.py              ← GraphJudge tri verification
     kg_pipeline/main.py                    ← Pipeline orchestration (LangGraph)

  3. CUSTOMIZE:
     • Change test_text in main.py for different input
     • Adjust judge backend in .env (JUDGE_BACKEND=bert/lora/ollama)
     • Add evidence-span validation in judge_triples()
     • Optimize extractor to use surviving_pairs for efficiency

  4. VALIDATE RESULTS:
     • Check Neo4j for stored triples (bolt://localhost:7687)
     • Verify verified_triples match expected facts
     • Examine rejected triples and why they were filtered
""")

    print_section("SUMMARY")
    print("""
  🧹 DENOISING:
     Removes noise (fillers, ads) while keeping facts.
     Entity-aware mode filters to entity-relevant sentences only.

  🔊 PHONETIC CORRECTION:
     Step A: Extract rough entities (with errors)
     Step B: Use context to infer correct spellings

  ⚖️  GRAPH JUDGE:
     Verifies each triple: "Is this actually in the source text?"
     Uses instruction-response format, parses yes/no
     Supports BERT/LoRA/Ollama backends with fallback

  🕸️  FULL PIPELINE:
     Sequential: denoise → phonetic → candidates → refine → judge → neo4j
     Uses LangGraph for orchestration
     Stores verified triples in knowledge graph database
""")

    print("\n" + "=" * 80)
    print("  Demo complete! See comments in this script for detailed explanations.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
