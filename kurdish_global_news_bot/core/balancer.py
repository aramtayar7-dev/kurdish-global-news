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

from config import (
    BASHUR_QUOTA_PER_8,
    BASHUR_TARGET_RATIO,
    BATCH_PUBLISH_LIMIT,
    ECON_CYBER_MAX_PER_8,
    ECON_CYBER_MIN_SCORE,
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
from storage.models import Article


def _normalized_article_score(article: Article) -> float:
    """Normalize article importance_score onto the 0..10 scale."""
    raw = float(getattr(article, "importance_score", 0.0) or 0.0)
    if 0.0 < raw <= 1.0:
        return raw * 10.0
    return raw


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
      1. Genuine breaking emergencies (`is_breaking=True` and `score >= 9.0`)
      2. Selected Southern Kurdistan (Bashur) articles — Politics & War (1A) first, then 1B
      3. Selected Global articles — Politics & War (Priority 2, regional first),
         then Economy/Cybersecurity slot (Priority 3)
    """
    breaking_rank = 0 if is_strict_breaking(article) else 1
    bashur_rank = 0 if is_bashur_article(article) else 1
    tier = classify_priority_tier(article)
    bucket = classify_topic_bucket(article)
    if bucket == "politics_war":
        tier_rank = 0 if tier == PRIORITY_TIER_1A else (1 if bashur_rank == 0 else 2)
    else:
        tier_rank = 1 if bashur_rank == 0 and bucket not in ("economy", "cybersecurity") else 3
    regional_rank = 0 if is_regional_geopolitics(article) else 1
    score = -float(getattr(article, "importance_score", 0.0) or 0.0)
    pub = getattr(article, "published_at", "") or getattr(article, "publication_date", "") or ""
    return (breaking_rank, bashur_rank, tier_rank, regional_rank, score, pub)


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

        # 1. Fill Bashur slots (Politics & War strictly first; Bashur Econ only if Bashur P&W < quota)
        picked_bashur.extend(pw_bashur_pool[:bashur_budget])
        if len(picked_bashur) < bashur_budget and ec_bashur:
            picked_bashur.extend(ec_bashur[: bashur_budget - len(picked_bashur)])

        # 2. Allocate Global slots: up to 1 slot reserved for strongest qualifying Econ/Cyber if budget permits
        global_budget = max(0, effective_limit - bashur_budget)
        global_ec_taken: list[Article] = []
        already_picked_ec = any(classify_topic_bucket(a) in ("economy", "cybersecurity") for a in picked_bashur)
        if not already_picked_ec and ec_global and (global_budget >= 1 or len(pw_bashur_pool) < bashur_budget):
            global_ec_taken = ec_global[:1]

        pw_global_needed = max(0, global_budget - len(global_ec_taken))
        picked_global.extend(pw_global_pool[:pw_global_needed])
        picked_global.extend(global_ec_taken)

        selected = picked_bashur + picked_global
        selected_ids = {id(a) for a in selected}

        # 3. Elastic backfill: empty slots are filled with Politics & War first (Global then extra Bashur)
        remaining_slots = effective_limit - len(selected)
        if remaining_slots > 0:
            extra_pw_global = [a for a in pw_global_pool if id(a) not in selected_ids]
            take = extra_pw_global[:remaining_slots]
            picked_global.extend(take)
            selected.extend(take)
            selected_ids.update(id(a) for a in take)
            remaining_slots = effective_limit - len(selected)

        if remaining_slots > 0:
            extra_pw_bashur = [a for a in pw_bashur_pool if id(a) not in selected_ids]
            take = extra_pw_bashur[:remaining_slots]
            picked_bashur.extend(take)
            selected.extend(take)
            selected_ids.update(id(a) for a in take)
            remaining_slots = effective_limit - len(selected)

        # If a Global slot couldn't take chosen Econ/Cyber earlier and space opened via scarcity, admit at most 1
        current_ec_count = sum(1 for a in selected if classify_topic_bucket(a) in ("economy", "cybersecurity"))
        if remaining_slots > 0 and current_ec_count < ec_budget:
            for art in chosen_ec:
                if id(art) not in selected_ids and remaining_slots > 0:
                    picked_global.append(art) if not is_bashur_article(art) else picked_bashur.append(art)
                    selected.append(art)
                    selected_ids.add(id(art))
                    remaining_slots -= 1
                    break

        deferred = [a for a in candidates if id(a) not in selected_ids]

        # Genuine breaking emergencies (from Politics/War, or Econ/Cyber if EC budget permits) never stay suppressed
        deferred_breaking = [
            a for a in deferred
            if is_strict_breaking(a) and (
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
                    deferred.append(evicted)
                    deferred.remove(brk)
                    selected.append(brk)

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
