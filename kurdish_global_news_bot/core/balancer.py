"""
Kurdish Global News Bot — 5-of-8 Southern Kurdistan (Bashur) Quota Balancer

Enforces a target mix of 5 Southern Kurdistan (Bashur) articles + 3 Global/Other
articles per 8 published articles (62.5% Bashur / 37.5% Global) with:
  - Priority 1A (Bashur Politics, War, Security, Drone/Missile, Peshmerga, Erbil-Baghdad)
    ranked strictly ahead of Priority 1B (other major Bashur news like salaries/economy)
  - Priority 2 (Global/Regional Politics & War: Iraq, Iran, Turkey, Syria, US, Middle East)
    ranked ahead of non-political/other global stories (Priority 3)
  - Elastic backfill: when fewer than 5 Bashur articles exist (0, 1, 2, 3, or 4),
    backfills remaining slots up to N with the highest-scoring Global articles
  - Rolling 8-article window ratio alignment when batch limit N differs from 8
  - Deterministic publishing order: genuine breaking emergencies first ->
    Southern Kurdistan (Politics & War first) -> Global articles
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import config
from config import (
    BASHUR_QUOTA_PER_8,
    BASHUR_TARGET_RATIO,
    BATCH_PUBLISH_LIMIT,
    ECON_CYBER_MAX_PER_8,
    ECON_CYBER_MIN_SCORE,
    ENABLE_SOURCE_DOMINANCE_CAP,
    MAX_POSTS_PER_SOURCE_GROUP_IN_WINDOW,
)
from core.importance_ranker import (
    PRIORITY_TIER_1A,
    article_selection_sort_key,
    classify_priority_tier,
    classify_topic_bucket,
    is_bashur_article,
    is_regional_geopolitics,
    is_strict_breaking,
)
from core.source_verifier import get_source_group, get_source_tier, TIER_3
from storage.models import Article


def _normalized_article_score(article: Article) -> float:
    """Normalize article importance_score onto the 0..10 scale."""
    raw = float(getattr(article, "importance_score", 0.0) or 0.0)
    if 0.0 < raw <= 1.0:
        return raw * 10.0
    return raw


def _is_dominance_cap_exempt(article: Article) -> bool:
    """
    Corroborated Tier-1 / Tier-2 genuine breaking emergencies may bypass the rolling
    per-source-group dominance cap. Tier-3 sources (e.g. Sabereen) are NEVER exempt.
    """
    if not is_strict_breaking(article):
        return False
    if get_source_tier(getattr(article, "source_name", "") or "") == TIER_3:
        return False
    if getattr(article, "_is_uncorroborated_tier3", False):
        return False
    return True


def _qualifies_for_econ_cyber_slot(article: Article) -> bool:
    """Return True if an economy/cybersecurity story meets the threshold for the 1-of-8 secondary slot."""
    if is_strict_breaking(article):
        return True
    return _normalized_article_score(article) >= ECON_CYBER_MIN_SCORE


@dataclass
class BalanceSelectionResult:
    """Structured result of 5-of-8 Bashur + 7-of-8 Politics/War balanced batch selection."""
    selected: list[Article]
    deferred: list[Article]
    bashur_selected: int
    global_selected: int
    bashur_quota_target: int
    reason: str
    politics_war_selected: int = 0
    econ_cyber_selected: int = 0


def batch_publish_sort_key(article: Article) -> tuple:
    """
    Sort selected batch strictly for Telegram publishing order:
      1. Genuine breaking emergencies (`is_breaking=True` and `score >= 9.0`), Kurdistan first
      2. Material THREAD_UPDATEs & Tier 1 Southern Kurdistan (Bashur) Politics & War
      3. Other Bashur stories, then Global Politics & War (regional first), then Economy/Cyber
    """
    breaking_rank = 0 if is_strict_breaking(article) else 1
    bashur_rank = 0 if is_bashur_article(article) else 1
    is_thread = bool(
        getattr(article, "is_thread_update", False)
        or getattr(article, "thread_root_message_id", None) is not None
    )
    tier = classify_priority_tier(article)
    bucket = classify_topic_bucket(article)
    if bucket == "politics_war":
        tier_rank = 0 if tier == PRIORITY_TIER_1A else (1 if bashur_rank == 0 else 2)
    else:
        tier_rank = 1 if bashur_rank == 0 and bucket not in ("economy", "cybersecurity") else 3

    if breaking_rank == 0:
        band_rank = 0
    elif is_thread or (bashur_rank == 0 and tier_rank == 0):
        band_rank = 0
    elif bashur_rank == 0:
        band_rank = 1
    elif tier_rank <= 2:
        band_rank = 2
    else:
        band_rank = 3

    regional_rank = 0 if is_regional_geopolitics(article) else 1
    score = -float(getattr(article, "importance_score", 0.0) or 0.0)
    pub = getattr(article, "published_at", "") or getattr(article, "publication_date", "") or ""
    return (breaking_rank, band_rank, bashur_rank, tier_rank, regional_rank, score, pub)


class QuotaBalancer:
    """
    Combined Elastic Quota & Topic Balancer:
      - Regional target: 5 Southern Kurdistan (Bashur) + 3 Global per 8 posts
      - Topic target: minimum 7 Politics/War + maximum 1 combined Economy OR Cybersecurity per 8 posts
        (fills 8/8 Politics & War when no high-impact Economy/Cybersecurity exists)
    """

    def __init__(
        self,
        batch_limit: int = BATCH_PUBLISH_LIMIT,
        bashur_per_8: int = BASHUR_QUOTA_PER_8,
        econ_cyber_max_per_8: int = ECON_CYBER_MAX_PER_8,
    ) -> None:
        self._batch_limit = max(1, batch_limit)
        self._bashur_per_8 = max(0, min(8, bashur_per_8))
        self._econ_cyber_max_per_8 = max(0, min(8, econ_cyber_max_per_8))
        self._ratio = self._bashur_per_8 / 8.0 if self._bashur_per_8 != 5 else BASHUR_TARGET_RATIO

    def compute_bashur_budget(
        self,
        limit: int,
        recent_published: Optional[list[Article]] = None,
    ) -> int:
        """
        Compute the target Bashur article budget for `limit` slots.
        - For `limit == 8`: returns `self._bashur_per_8` (5 of 8 = 62.5%).
        - For `limit != 8` when recent history exists: aligns `recent_window + batch`
          towards the 5:3 Bashur/Global ratio over the rolling 8-article window.
        """
        if limit <= 0:
            return 0
        if limit == 8:
            return self._bashur_per_8

        if recent_published:
            window_hist_len = max(0, 8 - min(limit, 8))
            history_slice = recent_published[:window_hist_len]
            if history_slice:
                recent_bashur = sum(1 for a in history_slice if is_bashur_article(a))
                total_window = len(history_slice) + limit
                target_window_bashur = round(total_window * self._ratio)
                return max(0, min(limit, target_window_bashur - recent_bashur))

        return max(0, min(limit, int(round(limit * self._ratio))))

    def compute_econ_cyber_budget(
        self,
        limit: int,
        recent_published: Optional[list[Article]] = None,
    ) -> int:
        """
        Compute remaining Economy/Cybersecurity combined budget (0 or 1) across the
        rolling 8-article window (`limit` upcoming + up to `8 - limit` recent posts).
        Never allows > 1 Economy/Cybersecurity article per 8 published posts.
        """
        if limit <= 0 or self._econ_cyber_max_per_8 <= 0:
            return 0
        if limit >= 8:
            return self._econ_cyber_max_per_8

        if recent_published:
            window_hist_len = max(0, 8 - min(limit, 8))
            history_slice = recent_published[:window_hist_len]
            recent_ec = sum(
                1 for a in history_slice if classify_topic_bucket(a) in ("economy", "cybersecurity")
            )
            return max(0, self._econ_cyber_max_per_8 - recent_ec)

        return self._econ_cyber_max_per_8

    def select_batch(
        self,
        candidates: list[Article],
        limit: Optional[int] = None,
        recent_published: Optional[list[Article]] = None,
    ) -> BalanceSelectionResult:
        """
        Select up to `limit` articles enforcing BOTH:
          - 5-of-8 Southern Kurdistan (Bashur) / Global quota (with elastic backfill)
          - Minimum 7 Politics & War + Maximum 1 combined Economy OR Cybersecurity per 8 posts
            (all 8/8 Politics & War when no qualifying Economy/Cybersecurity exists)
        """
        effective_limit = self._batch_limit if limit is None else max(0, limit)
        if not candidates or effective_limit == 0:
            return BalanceSelectionResult(
                selected=[],
                deferred=list(candidates),
                bashur_selected=0,
                global_selected=0,
                bashur_quota_target=0,
                reason="No eligible candidates or zero batch limit",
                politics_war_selected=0,
                econ_cyber_selected=0,
            )

        bashur_budget = self.compute_bashur_budget(effective_limit, recent_published=recent_published)
        ec_budget = self.compute_econ_cyber_budget(effective_limit, recent_published=recent_published)

        dom_cap_enabled = bool(getattr(config, "ENABLE_SOURCE_DOMINANCE_CAP", ENABLE_SOURCE_DOMINANCE_CAP))
        dom_cap = max(
            0,
            int(getattr(config, "MAX_POSTS_PER_SOURCE_GROUP_IN_WINDOW", MAX_POSTS_PER_SOURCE_GROUP_IN_WINDOW)),
        )
        window_hist_len = max(0, 8 - min(effective_limit, 8))
        history_slice = (recent_published or [])[:window_hist_len]
        group_counts: dict[str, int] = {}
        for hist_art in history_slice:
            grp = get_source_group(getattr(hist_art, "source_name", "") or "")
            if grp and grp != "unknown":
                group_counts[grp] = group_counts.get(grp, 0) + 1

        pw_bashur_pool = sorted(
            [a for a in candidates if classify_topic_bucket(a) == "politics_war" and is_bashur_article(a)],
            key=article_selection_sort_key,
        )
        pw_global_pool = sorted(
            [a for a in candidates if classify_topic_bucket(a) == "politics_war" and not is_bashur_article(a)],
            key=article_selection_sort_key,
        )
        ec_candidates = sorted(
            [
                a
                for a in candidates
                if classify_topic_bucket(a) in ("economy", "cybersecurity")
                and _qualifies_for_econ_cyber_slot(a)
            ],
            key=lambda a: (
                0 if is_strict_breaking(a) else 1,
                -_normalized_article_score(a),
                0 if is_bashur_article(a) else 1,
                getattr(a, "published_at", "") or getattr(a, "publication_date", "") or "",
            ),
        )

        # At most `ec_budget` (0 or 1) combined Economy/Cybersecurity article may enter
        chosen_ec = ec_candidates[:ec_budget]
        ec_bashur = [a for a in chosen_ec if is_bashur_article(a)]
        ec_global = [a for a in chosen_ec if not is_bashur_article(a)]

        picked_bashur: list[Article] = []
        picked_global: list[Article] = []
        selected: list[Article] = []
        selected_ids: set[int] = set()

        def _can_admit(art: Article, allow_tier12_scarcity_overflow: bool = False) -> bool:
            if id(art) in selected_ids:
                return False
            if not dom_cap_enabled or dom_cap <= 0:
                return True
            if _is_dominance_cap_exempt(art):
                return True
            grp = get_source_group(getattr(art, "source_name", "") or "")
            if not grp or grp == "unknown":
                return True
            if group_counts.get(grp, 0) < dom_cap:
                return True
            # Tier-3 (e.g. Sabereen) is STRICTLY hard-capped at `dom_cap` even during scarcity.
            if allow_tier12_scarcity_overflow and get_source_tier(getattr(art, "source_name", "") or "") != TIER_3:
                return True
            return False

        def _admit(art: Article, is_bashur: bool) -> None:
            selected.append(art)
            selected_ids.add(id(art))
            if is_bashur:
                picked_bashur.append(art)
            else:
                picked_global.append(art)
            grp = get_source_group(getattr(art, "source_name", "") or "")
            if grp and grp != "unknown":
                group_counts[grp] = group_counts.get(grp, 0) + 1

        # 1. Fill Bashur slots respecting dominance cap (P&W strictly first; Bashur Econ if needed)
        for art in pw_bashur_pool:
            if len(picked_bashur) >= bashur_budget:
                break
            if _can_admit(art, allow_tier12_scarcity_overflow=False):
                _admit(art, is_bashur=True)

        if len(picked_bashur) < bashur_budget:
            for art in ec_bashur:
                if len(picked_bashur) >= bashur_budget:
                    break
                if _can_admit(art, allow_tier12_scarcity_overflow=False):
                    _admit(art, is_bashur=True)

        # 2. Allocate Global slots respecting dominance cap & max 1 Econ/Cyber budget
        global_budget = max(0, effective_limit - bashur_budget)
        current_ec_count = sum(
            1 for a in selected if classify_topic_bucket(a) in ("economy", "cybersecurity")
        )
        global_ec_taken: list[Article] = []
        if current_ec_count < ec_budget and (global_budget >= 1 or len(pw_bashur_pool) < bashur_budget):
            for art in ec_global:
                if _can_admit(art, allow_tier12_scarcity_overflow=False):
                    global_ec_taken = [art]
                    break

        pw_global_needed = max(0, global_budget - len(global_ec_taken))
        pw_global_taken_count = 0
        for art in pw_global_pool:
            if pw_global_taken_count >= pw_global_needed or len(selected) >= effective_limit:
                break
            if _can_admit(art, allow_tier12_scarcity_overflow=False):
                _admit(art, is_bashur=False)
                pw_global_taken_count += 1

        for art in global_ec_taken:
            if len(selected) < effective_limit and _can_admit(art, allow_tier12_scarcity_overflow=False):
                _admit(art, is_bashur=False)

        # 3. Elastic backfill while still enforcing rolling dominance cap across all groups
        for art in pw_global_pool:
            if len(selected) >= effective_limit:
                break
            if _can_admit(art, allow_tier12_scarcity_overflow=False):
                _admit(art, is_bashur=False)

        for art in pw_bashur_pool:
            if len(selected) >= effective_limit:
                break
            if _can_admit(art, allow_tier12_scarcity_overflow=False):
                _admit(art, is_bashur=True)

        current_ec_count = sum(
            1 for a in selected if classify_topic_bucket(a) in ("economy", "cybersecurity")
        )
        if len(selected) < effective_limit and current_ec_count < ec_budget:
            for art in chosen_ec:
                if len(selected) >= effective_limit:
                    break
                if _can_admit(art, allow_tier12_scarcity_overflow=False):
                    _admit(art, is_bashur=is_bashur_article(art))
                    break

        # 4. Scarcity-only backfill for Tier-1 / Tier-2 (NEVER Tier-3!) if slots remain empty
        if len(selected) < effective_limit:
            for art in pw_bashur_pool:
                if len(picked_bashur) >= bashur_budget or len(selected) >= effective_limit:
                    break
                if _can_admit(art, allow_tier12_scarcity_overflow=True):
                    _admit(art, is_bashur=True)

            for art in pw_global_pool:
                if len(selected) >= effective_limit:
                    break
                if _can_admit(art, allow_tier12_scarcity_overflow=True):
                    _admit(art, is_bashur=False)

            for art in pw_bashur_pool:
                if len(selected) >= effective_limit:
                    break
                if _can_admit(art, allow_tier12_scarcity_overflow=True):
                    _admit(art, is_bashur=True)

            current_ec_count = sum(
                1 for a in selected if classify_topic_bucket(a) in ("economy", "cybersecurity")
            )
            if len(selected) < effective_limit and current_ec_count < ec_budget:
                for art in chosen_ec:
                    if len(selected) >= effective_limit:
                        break
                    if _can_admit(art, allow_tier12_scarcity_overflow=True):
                        _admit(art, is_bashur=is_bashur_article(art))
                        break

        deferred = [a for a in candidates if id(a) not in selected_ids]

        # Genuine corroborated Tier-1/2 breaking emergencies never stay suppressed
        deferred_breaking = [
            a for a in deferred
            if is_strict_breaking(a)
            and (
                not dom_cap_enabled
                or dom_cap <= 0
                or _is_dominance_cap_exempt(a)
            )
            and (
                classify_topic_bucket(a) == "politics_war"
                or (
                    classify_topic_bucket(a) in ("economy", "cybersecurity")
                    and ec_budget > 0
                )
            )
        ]
        if deferred_breaking:
            for brk in sorted(deferred_breaking, key=article_selection_sort_key):
                non_breaking_idx = next(
                    (i for i in range(len(selected) - 1, -1, -1) if not is_strict_breaking(selected[i])),
                    None,
                )
                if non_breaking_idx is not None:
                    evicted = selected.pop(non_breaking_idx)
                    ev_grp = get_source_group(getattr(evicted, "source_name", "") or "")
                    if ev_grp in group_counts and group_counts[ev_grp] > 0:
                        group_counts[ev_grp] -= 1
                    deferred.append(evicted)
                    deferred.remove(brk)
                    selected.append(brk)
                    brk_grp = get_source_group(getattr(brk, "source_name", "") or "")
                    if brk_grp and brk_grp != "unknown":
                        group_counts[brk_grp] = group_counts.get(brk_grp, 0) + 1

        ordered_selected = sorted(selected, key=batch_publish_sort_key)
        bashur_count = sum(1 for a in ordered_selected if is_bashur_article(a))
        global_count = len(ordered_selected) - bashur_count
        pw_count = sum(1 for a in ordered_selected if classify_topic_bucket(a) == "politics_war")
        ec_count = sum(1 for a in ordered_selected if classify_topic_bucket(a) in ("economy", "cybersecurity"))

        reason = (
            f"Selected {len(ordered_selected)}/{effective_limit}: "
            f"{bashur_count} Bashur (target {bashur_budget}) + {global_count} Global | "
            f"Topics: {pw_count} Politics/War + {ec_count} Econ/Cyber (EC budget {ec_budget})"
        )
        return BalanceSelectionResult(
            selected=ordered_selected,
            deferred=deferred,
            bashur_selected=bashur_count,
            global_selected=global_count,
            bashur_quota_target=bashur_budget,
            reason=reason,
            politics_war_selected=pw_count,
            econ_cyber_selected=ec_count,
        )
