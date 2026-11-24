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
)
from core.importance_ranker import (
    PRIORITY_TIER_1A,
    article_selection_sort_key,
    classify_priority_tier,
    is_bashur_article,
    is_regional_geopolitics,
    is_strict_breaking,
)
from storage.models import Article


@dataclass
class BalanceSelectionResult:
    """Structured result of 5-of-8 balanced batch selection."""
    selected: list[Article]
    deferred: list[Article]
    bashur_selected: int
    global_selected: int
    bashur_quota_target: int
    reason: str


def batch_publish_sort_key(article: Article) -> tuple:
    """
    Sort selected batch strictly for Telegram publishing order:
      1. Genuine breaking emergencies (`is_breaking=True` and `score >= 9.0`)
      2. Selected Southern Kurdistan (Bashur) articles — Politics & War (1A) first, then 1B
      3. Selected Global/Other articles — Politics & War (Priority 2, regional first), then Priority 3
    """
    breaking_rank = 0 if is_strict_breaking(article) else 1
    bashur_rank = 0 if is_bashur_article(article) else 1
    tier = classify_priority_tier(article)
    tier_rank = 0 if tier == PRIORITY_TIER_1A else (1 if bashur_rank == 0 else (2 if tier == "2" else 3))
    regional_rank = 0 if is_regional_geopolitics(article) else 1
    score = -float(getattr(article, "importance_score", 0.0) or 0.0)
    pub = getattr(article, "published_at", "") or getattr(article, "publication_date", "") or ""
    return (breaking_rank, bashur_rank, tier_rank, regional_rank, score, pub)


class QuotaBalancer:
    """Elastic 5-of-8 Southern Kurdistan (Bashur) quota & priority balancer."""

    def __init__(
        self,
        batch_limit: int = BATCH_PUBLISH_LIMIT,
        bashur_per_8: int = BASHUR_QUOTA_PER_8,
    ) -> None:
        self._batch_limit = max(1, batch_limit)
        self._bashur_per_8 = max(0, min(8, bashur_per_8))
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

    def select_batch(
        self,
        candidates: list[Article],
        limit: Optional[int] = None,
        recent_published: Optional[list[Article]] = None,
    ) -> BalanceSelectionResult:
        """
        Select up to `limit` articles balancing Southern Kurdistan (5/8) and Global (3/8)
        with elastic backfill and strict publishing order.
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
            )

        bashur_pool = sorted(
            [a for a in candidates if is_bashur_article(a)],
            key=article_selection_sort_key,
        )
        global_pool = sorted(
            [a for a in candidates if not is_bashur_article(a)],
            key=article_selection_sort_key,
        )

        bashur_budget = self.compute_bashur_budget(effective_limit, recent_published=recent_published)
        global_budget = max(0, effective_limit - bashur_budget)

        # Primary 5:3 selection
        picked_bashur = list(bashur_pool[:bashur_budget])
        picked_global = list(global_pool[:global_budget])

        # Elastic backfill: if fewer Bashur than quota (0-4), fill empty slots with top Global
        remaining_slots = effective_limit - (len(picked_bashur) + len(picked_global))
        if remaining_slots > 0:
            extra_global = global_pool[len(picked_global): len(picked_global) + remaining_slots]
            picked_global.extend(extra_global)
            remaining_slots = effective_limit - (len(picked_bashur) + len(picked_global))

        # Symmetric elastic backfill: if fewer Global than budget, fill remaining with extra Bashur
        if remaining_slots > 0:
            extra_bashur = bashur_pool[len(picked_bashur): len(picked_bashur) + remaining_slots]
            picked_bashur.extend(extra_bashur)

        selected = picked_bashur + picked_global
        selected_ids = {id(a) for a in selected}
        deferred = [a for a in candidates if id(a) not in selected_ids]

        # Genuine breaking emergencies must never be silently suppressed if deferred
        deferred_breaking = [a for a in deferred if is_strict_breaking(a)]
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

        reason = (
            f"Selected {len(ordered_selected)}/{effective_limit}: "
            f"{bashur_count} Bashur (target {bashur_budget}) + {global_count} Global"
        )
        return BalanceSelectionResult(
            selected=ordered_selected,
            deferred=deferred,
            bashur_selected=bashur_count,
            global_selected=global_count,
            bashur_quota_target=bashur_budget,
            reason=reason,
        )
