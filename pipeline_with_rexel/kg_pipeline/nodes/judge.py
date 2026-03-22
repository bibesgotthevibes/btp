from kg_pipeline.state import KGState
from kg_pipeline.utils.ollama import call_ollama
import json
import os
import re
from typing import List, Optional


_PID_LABELS = {
    "P17": "country",
    "P19": "place of birth",
    "P20": "place of death",
    "P27": "country of citizenship",
    "P31": "instance of",
    "P36": "capital",
    "P39": "position held",
    "P54": "member of sports team",
    "P69": "educated at",
    "P106": "occupation",
    "P108": "employer",
    "P112": "founded by",
    "P123": "publisher",
    "P127": "owned by",
    "P131": "located in the administrative territorial entity",
    "P136": "genre",
    "P137": "operator",
    "P159": "headquarters location",
    "P161": "cast member",
    "P166": "award received",
    "P171": "parent taxon",
    "P176": "manufacturer",
    "P178": "developer",
    "P190": "sister city",
    "P194": "legislative body",
    "P276": "location",
    "P279": "subclass of",
    "P355": "subsidiary",
    "P361": "part of",
    "P364": "original language of work",
    "P400": "platform",
    "P407": "language of work or name",
    "P449": "original broadcaster",
    "P463": "member of",
    "P495": "country of origin",
    "P527": "has part(s)",
    "P530": "diplomatic relation",
    "P551": "residence",
    "P569": "date of birth",
    "P570": "date of death",
    "P571": "inception",
    "P576": "dissolved, abolished or demolished date",
    "P577": "publication date",
    "P607": "conflict",
    "P625": "coordinate location",
    "P641": "sport",
    "P674": "characters",
    "P706": "located on terrain feature",
    "P710": "participant",
    "P740": "location of formation",
    "P749": "parent organization",
    "P800": "notable work",
    "P840": "narrative location",
    "P937": "work location",
    "P1001": "applies to jurisdiction",
    "P1056": "product or material produced",
    "P1270": "number of episodes",
    "P1344": "participant in",
    "P1365": "replaces",
    "P1366": "replaced by",
    "P1376": "capital of",
    "P1412": "languages spoken, written or signed",
    "P150": "contains the administrative territorial entity",
    "P155": "follows",
    "P156": "followed by",
    "P169": "chief executive officer",
    "P2632": "place of detention",
    "P264": "record label",
}


def _build_graphjudge_prompt(instruction: str, context_text: str) -> str:
    return f"""
You are verifying whether a candidate knowledge-graph triple is supported by the provided passage.

Return STRICT JSON only with this schema:
{{"is_true": true|false, "reason": "short reason"}}

Rules:
- Use only evidence from the passage.
- If uncertain or unsupported, set is_true to false.
- Do not include markdown or extra text.

Instruction:
{instruction}

Passage:
{context_text}
"""


def _norm_key(text: str) -> str:
    value = (text or "").lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text or "")
    return [p.strip() for p in parts if p and p.strip()]


def _collect_aliases(entity: str, entity_clusters: List[dict]) -> List[str]:
    aliases = {(entity or "").strip()}
    target = _norm_key(entity)
    if not target:
        return [a for a in aliases if a]

    for cluster in entity_clusters or []:
        canonical = cluster.get("canonical") or cluster.get("text") or ""
        mentions = cluster.get("mentions") or []

        candidates = [canonical] + [m for m in mentions if isinstance(m, str)]
        norm_candidates = {_norm_key(c) for c in candidates if isinstance(c, str) and c.strip()}
        if target in norm_candidates:
            for c in candidates:
                if isinstance(c, str) and c.strip():
                    aliases.add(c.strip())

    # Prefer longer aliases first so exact phrase matching is stronger.
    return sorted([a for a in aliases if a], key=len, reverse=True)


def _sentence_has_alias(sentence: str, aliases: List[str]) -> bool:
    s_key = _norm_key(sentence)
    if not s_key:
        return False
    for alias in aliases:
        a_key = _norm_key(alias)
        if a_key and a_key in s_key:
            return True
    return False


def _has_joint_lexical_support(raw_text: str, head_aliases: List[str], tail_aliases: List[str]) -> bool:
    for sentence in _split_sentences(raw_text):
        if _sentence_has_alias(sentence, head_aliases) and _sentence_has_alias(sentence, tail_aliases):
            return True
    return False


def _build_evidence_context(raw_text: str, head_aliases: List[str], tail_aliases: List[str], max_chars: int = 2600) -> str:
    """Select focused evidence snippets to reduce LLM false negatives on long passages."""
    sentences = _split_sentences(raw_text)
    if not sentences:
        return (raw_text or "")[:max_chars]

    scored = []
    for idx, sent in enumerate(sentences):
        score = 0
        if _sentence_has_alias(sent, head_aliases):
            score += 2
        if _sentence_has_alias(sent, tail_aliases):
            score += 2
        if score > 0:
            scored.append((idx, score))

    if not scored:
        return (raw_text or "")[:max_chars]

    selected_idx = set()
    for idx, _ in scored:
        selected_idx.add(idx)
        if idx - 1 >= 0:
            selected_idx.add(idx - 1)
        if idx + 1 < len(sentences):
            selected_idx.add(idx + 1)

    selected = [sentences[i] for i in sorted(selected_idx)]
    context = " ".join(selected)
    if len(context) <= max_chars:
        return context
    return context[:max_chars]


def _parse_judge_response(response_text: str) -> Optional[bool]:
    """Parse judge response.

    Returns:
    - True / False when confidently parsed.
    - None when undecidable (used by non-strict mode to keep recall high).
    """
    text = (response_text or "").strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            if isinstance(obj.get("is_true"), bool):
                return obj["is_true"]
            # Backward compatibility with older prompt schema.
            if isinstance(obj.get("valid"), bool):
                return obj["valid"]
    except Exception:
        pass

    # Fallback: extract first JSON object if model wrapped output.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                if isinstance(obj.get("is_true"), bool):
                    return obj["is_true"]
                if isinstance(obj.get("valid"), bool):
                    return obj["valid"]
        except Exception:
            pass

    # Final conservative fallback for legacy free-text answers.
    lowered = text.lower()[:120]
    if "yes" in lowered or "true" in lowered:
        return True
    if "no" in lowered or "false" in lowered:
        return False
    return None


def _judge_strict_mode() -> bool:
    """Return True if uncertain judge outputs should be rejected."""
    raw = os.getenv("JUDGE_STRICT", "true").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _relation_for_judge(relation: str) -> str:
    rel = (relation or "").strip()
    if not rel:
        return rel
    upper_rel = rel.upper()
    if re.fullmatch(r"P\d+", upper_rel):
        label = _PID_LABELS.get(upper_rel)
        if label:
            return f"{upper_rel} ({label})"
        return f"{upper_rel} (Wikidata property)"
    return rel


def _judge_backend(instruction: str, context_text: str) -> str:
    prompt = _build_graphjudge_prompt(instruction, context_text)
    return call_ollama(prompt)


def judge_triples(state: KGState) -> KGState:
    """
    GraphJudge-style triple verification.
    Uses instruction form from prepare_KGCom.ipynb:
    "Is this true: head relation tail?"
    """
    print("---(5) GRAPH JUDGE VERIFICATION---")
    
    # Allow disabling the judge entirely
    judge_enabled = os.getenv("JUDGE_ENABLED", "true").strip().lower() in {"1", "true", "yes"}
    if not judge_enabled:
        print("Judge disabled (JUDGE_ENABLED=false)")
        draft_triples = state.get("draft_triples", [])
        state["verified_triples"] = draft_triples
        print(f"Keeping all {len(draft_triples)} extracted triples without verification.")
        return state
    
    draft_triples = state.get("draft_triples", [])
    entity_clusters = state.get("entity_clusters", [])
    
    # Use original text for judging (preserve full context beyond denoiser edits).
    judge_text = state.get("raw_text", state.get("denoised_text", ""))

    if not draft_triples:
        print("No draft triples to verify.")
        state["verified_triples"] = []
        return state

    strict_mode = _judge_strict_mode()
    print(f"Judge strict mode: {'ON' if strict_mode else 'OFF'}")

    verified_triples = []

    for triple in draft_triples:
        head = triple.get("head")
        relation = triple.get("relation")
        tail = triple.get("tail")

        if not head or not relation or not tail:
            continue

        relation_hint = _relation_for_judge(relation)
        head_aliases = _collect_aliases(head, entity_clusters)
        tail_aliases = _collect_aliases(tail, entity_clusters)
        focused_context = _build_evidence_context(judge_text, head_aliases, tail_aliases)
        instruction = (
            "Is this triple supported by the passage? "
            f"head='{head}', relation='{relation_hint}', tail='{tail}'. "
            f"Known aliases: head={head_aliases[:5]}, tail={tail_aliases[:5]}."
        )

        try:
            response = _judge_backend(instruction, focused_context)
        except Exception as exc:
            print(f"Judge call failed for triple {triple}: {exc}")
            continue

        decision = _parse_judge_response(response)
        if decision is True:
            verified_triples.append({
                "head": head,
                "relation": relation,
                "tail": tail,
            })
        elif decision is None and not strict_mode:
            # In non-strict mode, keep undecidable triples to maximize recall.
            verified_triples.append({
                "head": head,
                "relation": relation,
                "tail": tail,
            })
            print(f"Kept (uncertain): {head} - {relation} - {tail}")
        elif decision is False and not strict_mode and _has_joint_lexical_support(judge_text, head_aliases, tail_aliases):
            # Non-strict recovery path: if both entities co-occur in at least one
            # sentence, keep candidate to avoid over-conservative LLM rejection.
            verified_triples.append({
                "head": head,
                "relation": relation,
                "tail": tail,
            })
            print(f"Kept (lexical support): {head} - {relation} - {tail}")
        else:
            print(f"Rejected: {head} - {relation} - {tail}")

    state["verified_triples"] = verified_triples
    print(f"Verified {len(verified_triples)} triples out of {len(draft_triples)} drafts.")
    return state