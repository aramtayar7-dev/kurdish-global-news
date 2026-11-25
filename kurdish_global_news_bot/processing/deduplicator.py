"""
Kurdish Global News Bot — Zero-API Local Cross-Source Semantic Event Deduplicator

Replaces external LLM calls with fast, deterministic, zero-API local cross-source
event deduplication against recent PUBLISHED headlines and already-accepted batch items:
  - Stage A (Fast title similarity): Normalized title fuzzy ratio >= 0.60
  - Stage B (Local cross-source semantic event match): Stopword-filtered token
    overlap + canonical multilingual named-entity / event-concept matching +
    fuzzy token-set similarity across English, Kurdish Sorani, and Arabic.
Consumes ZERO Gemini API requests.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from rapidfuzz import fuzz

from config import DEDUP_WINDOW_HOURS
from storage.database import Database
from storage.models import Article

logger = logging.getLogger("processing.deduplicator")

_STOPWORDS: set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "has", "have",
    "in", "into", "is", "it", "its", "of", "on", "or", "over", "said", "says", "than", "that",
    "the", "their", "this", "to", "under", "was", "were", "will", "with", "amid", "after",
    "before", "near", "new", "more", "report", "reported", "reports", "official", "officials",
    "news", "today", "yesterday", "live", "breaking", "update",
    # Arabic & Kurdish common function words
    "في", "من", "على", "إلى", "عن", "مع", "بعد", "قبل", "بين", "أن", "إن", "الذي", "التي", "هذا", "هذه",
    "لە", "بۆ", "بە", "لەسەر", "لەگەڵ", "دوای", "پێش", "ئەم", "ئەو", "کە", "بوو", "دەڵێت", "ڕایگەیاند",
}

# Canonical multilingual actors, places, and event concepts for paraphrase alignment
_CONCEPT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("loc:erbil", ("erbil", "hewler", "هەولێر", "اربيل", "أربيل")),
    ("loc:kirkuk", ("kirkuk", "کەرکوک", "كركوك")),
    ("loc:sulaymaniyah", ("sulaymaniyah", "slemani", "سلێمانی", "سلیمانیة")),
    ("loc:duhok", ("duhok", "دهۆک", "دهوك")),
    ("loc:kurdistan", ("kurdistan", "krg", "peshmerga", "کوردستان", "پێشمەرگە", "کوردستانی")),
    ("loc:baghdad_iraq", ("iraq", "iraqi", "baghdad", "anbar", "basra", "عێراق", "بەغدا", "العراق", "بغداد")),
    ("loc:iran", ("iran", "iranian", "tehran", "ئێران", "تاران", "إيران", "طهران")),
    ("loc:israel", ("israel", "israeli", "tel aviv", "idf", "ئیسرائیل", "إسرائيل", "اسرائيل")),
    ("loc:lebanon_beirut", ("beirut", "lebanon", "lebanese", "hezbollah", "بەیروت", "لوبنان", "بیروت", "بيروت", "لبنان", "اللبناني", "اللبنانية")),
    ("loc:gaza", ("gaza", "rafah", "hamas", "غەززە", "غزة", "رفح")),
    ("loc:syria", ("syria", "syrian", "damascus", "qamishli", "سووریا", "سوریا", "دیمەشق", "سوريا", "دمشق")),
    ("loc:turkey", ("turkey", "turkish", "ankara", "تورکیا", "ئەنقەرە", "تركيا", "أنقرة")),
    ("loc:usa", ("united states", "american", "pentagon", "washington", "white house", "ئەمریکا", "واشنتن", "الأمريكي", "أمريكا", "واشنطن")),
    ("loc:russia", ("russia", "russian", "moscow", "kremlin", "رووسیا", "مۆسکۆ", "روسيا", "موسكو")),
    ("loc:ukraine", ("ukraine", "ukrainian", "kyiv", "ئۆکرانیا", "کیێڤ", "أوكرانيا", "كييف")),
    ("loc:red_sea", ("red sea", "houthi", "bab el mandeb", "البحر الأحمر", "دەریای سوور")),
    ("act:strike_explosion", (
        "bomb", "bombed", "bombing", "bombardment", "explosion", "explosions", "blast",
        "shelling", "strike", "strikes", "attack", "attacks",
        "هێرش", "تەقینەوە", "بۆردومان", "تۆپباران", "قصف", "انفجار", "انفجارات", "ضربات",
    )),
    ("act:air_ops", ("airstrike", "airstrikes", "warplane", "warplanes", "aerial", "fighter jet", "غارات", "فڕۆکەی جەنگی")),
    ("act:missile_drone", ("missile", "missiles", "ballistic", "rocket", "drone", "drones", "مووشەک", "درۆن", "صاروخ", "مسيرة", "طائرات")),
    ("act:ceasefire", ("ceasefire", "truce", "armistice", "ئاگربەست", "وقف إطلاق النار", "هدنة")),
    ("act:summit_talks", ("summit", "talks", "negotiation", "negotiations", "diplomatic", "treaty", "accord", "کۆبوونەوە", "گفتوگۆ", "ڕێککەوتن", "مباحثات", "قمة", "اتفاق")),
    ("act:sanctions", ("sanctions", "embargo", "tariff", "tariffs", "سزا", "عقوبات")),
    ("act:earthquake", ("earthquake", "quake", "tsunami", "tremor", "بوومەلەرزە", "زلزال")),
    ("act:airport_base", ("airport", "airbase", "depot", "depots", "base", "bases", "فڕۆکەخانە", "بنکە", "مطار", "قاعدة")),
)

_TOKEN_SYNONYM_MAP: dict[str, str] = {}
for _concept_name, _forms in _CONCEPT_GROUPS:
    for _form in _forms:
        if " " not in _form:
            _TOKEN_SYNONYM_MAP[_form] = _concept_name


@dataclass
class DedupResult:
    """Outcome of a zero-API cross-source deduplication check."""
    is_duplicate: bool
    matched_id: Optional[int] = None
    reason: str = ""
    stage: str = "none"
    used_fallback: bool = False


def _normalize_unicode_text(text: str) -> str:
    """Normalize Arabic/Kurdish character variants and lowercase text cleanly."""
    t = (text or "").lower()
    t = (
        t.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ة", "ه")
        .replace("ى", "ي")
        .replace("ك", "ک")
    )
    cleaned = re.sub(r"[^\w\s\u0621-\u064A\u0671-\u06D5]", " ", t)
    return re.sub(r"\s+", " ", cleaned).strip()


def _normalize_for_match(text: str) -> str:
    """Lowercase and strip punctuation/extra whitespace for fast title comparison."""
    cleaned = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _stem_token(tok: str) -> str:
    if len(tok) > 5 and tok.isascii():
        if tok.endswith("ies"):
            return tok[:-3] + "y"
        if tok.endswith("ing") or tok.endswith("ment"):
            pass
        elif tok.endswith("ed") and len(tok) > 5:
            return tok[:-2]
        elif tok.endswith("es") and len(tok) > 4:
            return tok[:-2]
        elif tok.endswith("s") and not tok.endswith("ss") and len(tok) > 4:
            return tok[:-1]
    return tok


def _tokenize_content(text: str) -> set[str]:
    norm = _normalize_unicode_text(text)
    tokens: set[str] = set()
    for raw in norm.split():
        if raw in _STOPWORDS or len(raw) <= 2:
            continue
        stemmed = _stem_token(raw)
        if stemmed not in _STOPWORDS:
            tokens.add(stemmed)
    return tokens


def _extract_event_concepts(text: str) -> tuple[set[str], set[str], set[str]]:
    """
    Extract (place_or_actor_concepts, action_concepts, proper_entities_or_numbers)
    using zero-API multilingual canonical mapping.
    """
    norm = _normalize_unicode_text(text)
    padded = f" {norm} "
    places: set[str] = set()
    actions: set[str] = set()

    for concept, forms in _CONCEPT_GROUPS:
        matched = False
        for form in forms:
            norm_form = _normalize_unicode_text(form)
            if " " in norm_form:
                if f" {norm_form} " in padded:
                    matched = True
                    break
            elif norm_form in padded.split():
                matched = True
                break
        if matched:
            if concept.startswith("loc:"):
                places.add(concept)
            else:
                actions.add(concept)

    # Multi-char capitalized entities & numeric anchors from raw text (excluding tokens already mapped as places/actions)
    entities: set[str] = set()
    for cap in re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", text or ""):
        low = cap.lower()
        if low not in _STOPWORDS and low not in _TOKEN_SYNONYM_MAP:
            entities.add(low)
    for num in re.findall(r"\b\d{2,}\b", text or ""):
        entities.add(num)

    return places, actions, entities


class SemanticDeduplicator:
    """
    Zero-API cross-source event deduplicator combining Stage A normalized title
    similarity and Stage B local token/entity/concept overlap.
    """

    def __init__(
        self,
        db: Database,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        window_hours: Optional[int] = None,
        fast_threshold: float = 0.60,
        semantic_threshold: float = 0.52,
        max_candidates: int = 25,
        base_backoff: float = 0.0,
    ) -> None:
        self._db = db
        self._window_hours = window_hours if window_hours is not None else DEDUP_WINDOW_HOURS
        self._fast_threshold = fast_threshold
        self._semantic_threshold = semantic_threshold
        self._max_candidates = max_candidates
        # Retained only as a safe metadata indicator (consumes 0 Gemini requests)
        self.uses_gemini = False

    async def cleanup(self) -> None:
        """No-op cleanup (no external API session held)."""
        return None

    @staticmethod
    def _compute_title_similarity(title_a: str, title_b: str) -> float:
        """Return normalized [0.0, 1.0] lexical similarity between two titles."""
        norm_a = _normalize_for_match(title_a)
        norm_b = _normalize_for_match(title_b)
        if not norm_a or not norm_b:
            return 0.0
        ratio_seq = fuzz.ratio(norm_a, norm_b) / 100.0
        ratio_tok = fuzz.token_sort_ratio(norm_a, norm_b) / 100.0
        return max(ratio_seq, ratio_tok)

    def _fast_similarity_check(
        self,
        article: Article,
        recent_headlines: list[dict[str, Any]],
    ) -> Optional[DedupResult]:
        """Stage A: zero-cost lexical/token similarity check (>= 0.60)."""
        candidate_titles = [
            t for t in (article.original_title, article.translated_title) if t and t.strip()
        ]
        if not candidate_titles:
            return None

        best_score = 0.0
        best_item: Optional[dict[str, Any]] = None

        for item in recent_headlines:
            ref_titles = [
                t for t in (item.get("original_title", ""), item.get("translated_title", ""))
                if t and t.strip()
            ]
            for cand in candidate_titles:
                for ref in ref_titles:
                    sim = self._compute_title_similarity(cand, ref)
                    if sim > best_score:
                        best_score = sim
                        best_item = item

        if best_item is not None and best_score >= self._fast_threshold:
            return DedupResult(
                is_duplicate=True,
                matched_id=int(best_item["id"]) if best_item.get("id") is not None else None,
                reason=f"Fast title similarity {best_score:.2f} >= {self._fast_threshold:.2f}",
                stage="fast",
            )
        return None

    def _local_semantic_match_check(
        self,
        article: Article,
        recent_headlines: list[dict[str, Any]],
    ) -> Optional[DedupResult]:
        """
        Stage B (zero-API): Compare candidate title/summary against recent published
        headlines using token overlap, multilingual canonical entities/actions, and token-set ratio.
        """
        cand_text = " ".join(
            p for p in (article.original_title, article.translated_title, (article.original_summary or "")[:240])
            if p and p.strip()
        ).strip()
        if not cand_text:
            return None

        cand_tokens = _tokenize_content(f"{article.original_title or ''} {article.translated_title or ''}")
        cand_places, cand_actions, cand_entities = _extract_event_concepts(cand_text)
        cand_norm_title = _normalize_unicode_text(article.original_title or article.translated_title or "")

        best_score = 0.0
        best_item: Optional[dict[str, Any]] = None
        best_detail = ""

        for item in recent_headlines:
            ref_title = " ".join(
                t for t in (item.get("original_title", ""), item.get("translated_title", ""))
                if t and t.strip()
            )
            ref_text = " ".join(
                t for t in (ref_title, (item.get("original_summary") or "")[:240]) if t and t.strip()
            )
            if not ref_text:
                continue

            ref_tokens = _tokenize_content(ref_title or ref_text)
            ref_places, ref_actions, ref_entities = _extract_event_concepts(ref_text)
            ref_norm_title = _normalize_unicode_text(item.get("original_title", "") or item.get("translated_title", ""))

            shared_places = cand_places & ref_places
            shared_actions = cand_actions & ref_actions
            shared_entities = cand_entities & ref_entities
            shared_tokens = cand_tokens & ref_tokens
            # Distinguish location names from specific event words so two unrelated Erbil/Baghdad items don't collide
            shared_non_loc_tokens = {
                tok for tok in shared_tokens
                if not (_TOKEN_SYNONYM_MAP.get(tok, "").startswith("loc:"))
            }

            token_overlap = (
                len(shared_tokens) / max(1, min(len(cand_tokens), len(ref_tokens)))
                if cand_tokens and ref_tokens
                else 0.0
            )
            token_jaccard = (
                len(shared_tokens) / max(1, len(cand_tokens | ref_tokens))
                if cand_tokens and ref_tokens
                else 0.0
            )
            token_set = (
                fuzz.token_set_ratio(cand_norm_title, ref_norm_title) / 100.0
                if cand_norm_title and ref_norm_title
                else 0.0
            )

            all_cand_concepts = cand_places | cand_actions | cand_entities
            all_ref_concepts = ref_places | ref_actions | ref_entities
            shared_concept_count = len(shared_places) + len(shared_actions) + len(shared_entities)
            concept_jaccard = (
                shared_concept_count / max(1, len(all_cand_concepts | all_ref_concepts))
                if (all_cand_concepts and all_ref_concepts)
                else 0.0
            )

            combined = (0.38 * token_overlap) + (0.32 * token_set) + (0.30 * concept_jaccard)

            # High-precision paraphrase & lexical rules:
            # Never merge two articles purely for sharing a region/city without action/topic evidence.
            concept_anchors = len(shared_places) + len(shared_actions) + (1 if len(shared_entities) >= 1 else 0)
            paraphrase_hit = bool(
                (shared_places or len(shared_entities) >= 2)
                and shared_actions
                and (concept_anchors >= 3 or token_overlap >= 0.28 or token_set >= 0.42)
            )
            lexical_entity_hit = bool(
                len(shared_non_loc_tokens) >= 2
                and len(shared_tokens) >= 3
                and token_overlap >= 0.50
                and token_set >= 0.58
            )
            combined_hit = bool(
                combined >= self._semantic_threshold
                and (len(shared_non_loc_tokens) >= 2 or (bool(shared_actions) and len(shared_tokens) >= 2))
            )

            if paraphrase_hit or lexical_entity_hit or combined_hit:
                effective = max(combined, 0.72 if paraphrase_hit else combined)
                if effective > best_score:
                    best_score = effective
                    best_item = item
                    matched_anchors = sorted(shared_places | shared_actions | shared_entities)[:4]
                    best_detail = (
                        f"Local cross-source match (score={effective:.2f}, jaccard={token_jaccard:.2f}, "
                        f"anchors={','.join(matched_anchors) or 'tokens'})"
                    )

        if best_item is not None:
            return DedupResult(
                is_duplicate=True,
                matched_id=int(best_item["id"]) if best_item.get("id") is not None else None,
                reason=best_detail,
                stage="local_semantic",
            )
        return None

    async def check_duplicate(
        self,
        article: Article,
        extra_recent: Optional[list[dict[str, Any]]] = None,
    ) -> DedupResult:
        """
        Check whether `article` duplicates an event published in the last
        `DEDUP_WINDOW_HOURS` hours (or already accepted in `extra_recent`)
        using zero Gemini API calls.
        """
        recent_headlines = await self._db.get_recent_published_headlines(hours=self._window_hours)
        if extra_recent:
            recent_headlines = list(extra_recent) + list(recent_headlines)
        if not recent_headlines:
            return DedupResult(
                is_duplicate=False,
                reason="No recent published headlines in lookback window",
                stage="none",
            )

        # Stage A: fast normalized headline similarity
        fast_match = self._fast_similarity_check(article, recent_headlines)
        if fast_match is not None:
            return fast_match

        # Stage B: zero-API local semantic / entity / concept matching
        local_match = self._local_semantic_match_check(article, recent_headlines)
        if local_match is not None:
            return local_match

        return DedupResult(
            is_duplicate=False,
            matched_id=None,
            reason="Distinct event (local zero-API dedup)",
            stage="local_semantic",
        )
