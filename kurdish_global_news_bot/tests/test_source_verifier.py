"""
Comprehensive automated tests for Idea 4:
Telegram Source Balancing, Credibility Tiers, Single-Source Verification Guard
& Rolling Per-Source-Group Dominance Cap (`kurdish_global_news_bot`).

All external network, Telegram, Gemini, and filesystem calls are mocked.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import config
from config import (
    ArticleStatus,
    ENABLE_SOURCE_DOMINANCE_CAP,
    ENABLE_SOURCE_VERIFICATION,
    MAX_POSTS_PER_SOURCE_GROUP_IN_WINDOW,
    Priority,
    SOURCE_CREDIBILITY,
    TELEGRAM_SOURCE_CHANNELS,
)
from core.balancer import QuotaBalancer
from core.engine import NewsEngine
from core.importance_ranker import (
    RelevanceEvaluation,
    check_blocked_topic,
    check_pre_filter_off_topic,
    is_strict_breaking,
)
from core.pipeline import ArticlePipeline
from core.source_verifier import (
    CorroborationResult,
    TIER_1,
    TIER_2,
    TIER_3,
    apply_verification_guard,
    compute_corroboration,
    get_source_group,
    get_source_tier,
    select_preferred_source,
)
from monitoring.health import HealthMonitor
from monitoring.metrics import Metrics
from processing.content_formatter import (
    TIER3_ATTRIBUTION_PREFIX,
    ContentFormatter,
    prepend_tier3_attribution,
)
from processing.deduplicator import SemanticDeduplicator
from processing.gemini_client import GeminiClient
from processing.translator import is_valid_kurdish_sorani
from sources.base import RawArticle
from sources.telegram_source import TelegramSource, clean_telegram_message_text
from storage.database import Database
from storage.models import Article
from tg_bot.publisher import TelegramPublisher


def _iso_minutes_ago(minutes: float = 5.0) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


def _make_article(
    art_id: int | None = 1,
    title: str = "Iraqi forces reinforce security positions near Baghdad",
    source_name: str = "Reuters",
    category: str = "politics_world",
    region: str = "",
    url: str = "",
    score: float = 8.4,
    is_breaking: bool = False,
    event_key: str = "",
    event_entities: list[str] | None = None,
    summary: str = "",
    minutes_ago: float = 5.0,
) -> Article:
    pub_iso = _iso_minutes_ago(minutes_ago)
    art = Article(
        id=art_id,
        original_url=url or f"https://example.com/{source_name.lower().replace(' ', '-')}/{art_id or 1}",
        canonical_url=url or f"https://example.com/{source_name.lower().replace(' ', '-')}/{art_id or 1}",
        original_title=title,
        normalized_title=Article.normalize_title(title),
        original_summary=summary or (
            f"{title}. Senior defense and diplomatic officials confirmed urgent security "
            "coordination, monitoring operations, and joint government briefings across key zones."
        ),
        source_name=source_name,
        category=category,
        region=region,
        priority=Priority.CRITICAL if is_breaking else Priority.NORMAL,
        importance_score=score,
        is_breaking=is_breaking,
        event_key=event_key,
        event_entities=list(event_entities or []),
        publication_date=pub_iso,
        discovered_at=pub_iso,
    )
    art.compute_content_hash()
    art.compute_event_fingerprint()
    return art


@pytest.fixture
async def temp_db(tmp_path):
    db = Database(str(tmp_path / "test_source_verifier.db"))
    await db.connect()
    yield db
    await db.close()


# ═══════════════════════════════════════════════════
# 1. Telegram Channel Roster & Metadata Validation
# ═══════════════════════════════════════════════════
class TestChannelRosterAndMetadata:
    def test_active_telegram_channels_have_required_metadata_and_ajanews_replaces_sabereen(self):
        assert len(TELEGRAM_SOURCE_CHANNELS) == 6
        usernames = {ch["username"] for ch in TELEGRAM_SOURCE_CHANNELS}
        assert usernames == {
            "AjaNews",
            "rudawdigital",
            "Kurdistan24news",
            "nrttvofficial",
            "liveuamap",
            "ClashReport",
        }
        assert "SabrenNewss" not in usernames
        assert "Sabren_News1" not in usernames

        by_user = {ch["username"]: ch for ch in TELEGRAM_SOURCE_CHANNELS}
        aja = by_user["AjaNews"]
        assert aja["name"] == "Al Jazeera"
        assert aja["region_hint"] == "middle_east"
        assert aja["lang"] == "ar"
        assert aja["credibility_tier"] == "1"
        assert aja["source_group"] == "aljazeera"

        required_keys = {"username", "name", "region_hint", "lang", "credibility_tier", "source_group"}
        for ch in TELEGRAM_SOURCE_CHANNELS:
            assert required_keys.issubset(ch.keys())
            assert ch["lang"] in {"ckb", "en", "ar"}
            assert ch["credibility_tier"] in {"1", "2", "3"}
            assert ch["source_group"]
            # Also ensure credibility score exists for configured display name
            assert ch["name"].lower() in SOURCE_CREDIBILITY

    def test_verification_and_cap_config_defaults(self):
        assert ENABLE_SOURCE_VERIFICATION is True
        assert ENABLE_SOURCE_DOMINANCE_CAP is True
        assert MAX_POSTS_PER_SOURCE_GROUP_IN_WINDOW == 2

    def test_kurdish_telegram_signatures_stripped_cleanly(self):
        raw = (
            "هێزەکانی پێشمەرگە و حکوومەتی هەرێم لە هەولێر کۆبوونەوە.\n"
            "#Rudaw\n"
            "تۆڕی میدیایی رووداو\n"
            "#Kurdistan24\n"
            "#NRT\n"
            "بۆ بینینی زیاتر سەردانی ماڵپەڕەکەمان بکەن"
        )
        cleaned = clean_telegram_message_text(raw)
        assert "پێشمەرگە" in cleaned
        assert "تۆڕی میدیایی رووداو" not in cleaned
        assert "Rudaw" not in cleaned
        assert "Kurdistan24" not in cleaned
        assert "NRT" not in cleaned


# ═══════════════════════════════════════════════════
# 2. Source Group Unification & Credibility Tiers
# ═══════════════════════════════════════════════════
class TestSourceGroupAndTierClassification:
    def test_paired_outlets_unify_to_same_source_group(self):
        assert get_source_group("Sabereen News") == "sabereen"
        assert get_source_group("Sabereen Plus") == "sabereen"

        assert get_source_group("Rudaw") == "rudaw"
        assert get_source_group("Rudaw Digital") == "rudaw"
        assert get_source_group("Rudaw English") == "rudaw"

        assert get_source_group("Kurdistan24") == "kurdistan24"
        assert get_source_group("Kurdistan24 EN") == "kurdistan24"
        assert get_source_group("Kurdistan24 Sorani") == "kurdistan24"
        assert get_source_group("K24") == "kurdistan24"

        assert get_source_group("NRT") == "nrt"
        assert get_source_group("NRT TV") == "nrt"

        assert get_source_group("Reuters") == "reuters"
        assert get_source_group("Reuters World") == "reuters"
        assert get_source_group("Reuters Business") == "reuters"

        assert get_source_group("BBC") == "bbc"
        assert get_source_group("BBC World") == "bbc"

    def test_tier_assignments(self):
        for t1 in ("Reuters", "Reuters World", "BBC World", "AP News", "Al Jazeera", "UN News", "NATO News"):
            assert get_source_tier(t1) == TIER_1
        for t2 in ("Rudaw", "Kurdistan24", "Kurdistan24 EN", "NRT", "Liveuamap", "Clash Report", "Defense News"):
            assert get_source_tier(t2) == TIER_2
        for t3 in ("Sabereen News", "Sabereen Plus", "Unknown Telegram Blog"):
            assert get_source_tier(t3) == TIER_3


# ═══════════════════════════════════════════════════
# 3 & 4 & 5. Single-Source Tier-3 Guard & Cross-Source Corroboration
# ═══════════════════════════════════════════════════
class TestVerificationGuardAndCorroboration:
    def test_single_source_tier3_breaking_downgrade_and_attribution(self):
        sab_art = _make_article(
            art_id=10,
            title="Explosions reported near US military convoy in western Iraq",
            source_name="Sabereen News",
            category="military_conflict",
            url="https://t.me/SabrenNewss/5001",
            score=9.5,
            is_breaking=True,
            event_key="western_iraq_convoy_blast",
            event_entities=["iraq", "convoy", "explosion"],
        )

        corrob = compute_corroboration(sab_art, batch_articles=[sab_art], recent_history=[])
        assert corrob.corroboration_count == 1
        assert corrob.highest_corroborating_tier == TIER_3
        assert corrob.is_uncorroborated_tier3 is True

        apply_verification_guard(sab_art, corrob)
        assert sab_art.is_breaking is False
        assert sab_art.priority == Priority.HIGH
        assert is_strict_breaking(sab_art) is False

        raw_kurdish_summary = "تەقینەوەیەک لە نزیک کاروانێکی سەربازی لە ڕۆژئاوای عێراق ڕوویدا."
        attributed = prepend_tier3_attribution(raw_kurdish_summary)
        assert attributed.startswith(TIER3_ATTRIBUTION_PREFIX)
        assert is_valid_kurdish_sorani(attributed) is True
        # Calling again never double-prefixes
        assert prepend_tier3_attribution(attributed) == attributed

    def test_sabereen_news_plus_sabereen_plus_does_not_corroborate(self):
        sab1 = _make_article(
            art_id=11,
            title="Rocket fire hits desert logistics depot in Anbar",
            source_name="Sabereen News",
            url="https://t.me/SabrenNewss/5011",
            score=9.4,
            is_breaking=True,
            event_key="anbar_depot_rocket_attack",
            event_entities=["anbar", "depot", "rocket"],
        )
        sab2 = _make_article(
            art_id=12,
            title="Missiles target Anbar logistics depot overnight",
            source_name="Sabereen Plus",
            url="https://t.me/Sabren_News1/5012",
            score=9.2,
            is_breaking=True,
            event_key="anbar_depot_rocket_attack",
            event_entities=["anbar", "depot", "rocket"],
        )

        corrob = compute_corroboration(sab1, batch_articles=[sab1, sab2], recent_history=[])
        assert corrob.corroborating_sources == ["sabereen"]
        assert corrob.corroboration_count == 1
        assert corrob.highest_corroborating_tier == TIER_3
        assert corrob.is_uncorroborated_tier3 is True

        apply_verification_guard(sab1, corrob)
        assert sab1.is_breaking is False

    def test_cross_source_corroboration_and_higher_tier_preference(self):
        sab = _make_article(
            art_id=21,
            title="غارة جوية تستهدف منشأة عسكرية قرب بغداد",
            source_name="Sabereen News",
            category="military_conflict",
            url="https://t.me/SabrenNewss/6001",
            score=9.4,
            is_breaking=True,
            event_key="baghdad_military_airstrike",
            event_entities=["baghdad", "military", "airstrike"],
        )
        reuters = _make_article(
            art_id=22,
            title="Airstrike hits military facility outside Baghdad, officials say",
            source_name="Reuters",
            category="military_conflict",
            url="https://reuters.com/world/baghdad-airstrike-6001",
            score=9.5,
            is_breaking=True,
            event_key="baghdad_military_airstrike",
            event_entities=["baghdad", "military", "airstrike"],
        )

        corrob = compute_corroboration(sab, batch_articles=[sab, reuters], recent_history=[])
        assert corrob.corroboration_count == 2
        assert corrob.corroborating_sources == ["reuters", "sabereen"]
        assert corrob.highest_corroborating_tier == TIER_1
        assert corrob.is_uncorroborated_tier3 is False

        preferred = select_preferred_source([sab, reuters])
        assert len(preferred) == 1
        assert preferred[0].source_name == "Reuters"

    @pytest.mark.asyncio
    async def test_corroboration_uses_sqlite_recent_history_source_name(self, temp_db):
        pub_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        reuters = _make_article(
            art_id=None,
            title="US and Iraqi ministers hold emergency talks in Erbil",
            source_name="Reuters World",
            category="kurdistan",
            url="https://reuters.com/world/erbil-talks-101",
            score=8.8,
            event_key="erbil_emergency_ministerial_talks",
            event_entities=["erbil", "iraq", "ministers"],
        )
        reuters.published_at = pub_time
        reuters.status = ArticleStatus.PUBLISHED
        rid = await temp_db.insert_article(reuters)
        await temp_db.update_article_status(
            rid,
            ArticleStatus.PUBLISHED,
            published_at=pub_time,
            event_key=reuters.event_key,
            event_entities=reuters.event_entities,
        )

        history = await temp_db.get_recent_published_headlines(hours=24)
        assert len(history) == 1
        assert history[0]["source_name"] == "Reuters World"

        sab_follow = _make_article(
            art_id=99,
            title="اجتماع طارئ في أربيل بين وزراء العراق وأمريكا",
            source_name="Sabereen News",
            category="kurdistan",
            url="https://t.me/SabrenNewss/7099",
            score=9.1,
            is_breaking=True,
            event_key="erbil_emergency_ministerial_talks",
            event_entities=["erbil", "ministers", "talks"],
        )
        corrob = compute_corroboration(sab_follow, batch_articles=[sab_follow], recent_history=history)
        assert corrob.corroboration_count == 2
        assert corrob.highest_corroborating_tier == TIER_1
        assert corrob.is_uncorroborated_tier3 is False

    @pytest.mark.asyncio
    async def test_pipeline_end_to_end_downgrades_uncorroborated_tier3_and_prefers_tier1(self, temp_db):
        mock_bot = AsyncMock()
        msg_counter = {"n": 8000}

        async def _send_photo(*args, **kwargs):
            msg_counter["n"] += 1
            m = MagicMock()
            m.message_id = msg_counter["n"]
            return m

        mock_bot.send_photo = AsyncMock(side_effect=_send_photo)
        publisher = TelegramPublisher()
        publisher._channel_id = "@KurdishGlobalNews"
        publisher._bot = mock_bot
        publisher._image_handler.download_image = AsyncMock(return_value=None)

        pipeline = ArticlePipeline(temp_db, publisher, HealthMonitor(), Metrics())
        pipeline._semantic_dedup = SemanticDeduplicator(temp_db, api_key="", base_backoff=0.0)

        # Uncorroborated Tier-3 claim marked breaking 9.5 by Gemini -> MUST be downgraded & attributed
        sab_rumor = _make_article(
            art_id=None,
            title="Unconfirmed explosion reported at western desert compound",
            source_name="Sabereen Plus",
            category="military_conflict",
            url="https://unverified-blog.example/rumor-9501",
            score=9.5,
            is_breaking=True,
        )
        sab_rumor.id = await temp_db.insert_article(sab_rumor)

        # Duplicate cross-language pair: Tier-3 channel + Reuters (English) with same event_key -> Reuters wins!
        sab_dup = _make_article(
            art_id=None,
            title="غارة تستهدف مستودعا عسكريا شرق بغداد",
            source_name="Sabereen News",
            category="military_conflict",
            url="https://unverified-blog.example/raid-9502",
            score=9.2,
            is_breaking=True,
        )
        sab_dup.id = await temp_db.insert_article(sab_dup)

        reuters_main = _make_article(
            art_id=None,
            title="Airstrike destroys military warehouse east of Baghdad",
            source_name="Reuters",
            category="military_conflict",
            url="https://reuters.com/world/baghdad-warehouse-9503",
            score=9.3,
            is_breaking=True,
        )
        reuters_main.id = await temp_db.insert_article(reuters_main)

        async def _fake_batch_eval(articles, batch_size=12):
            evals = []
            for a in articles:
                if "9501" in a.original_url:
                    evals.append(RelevanceEvaluation(
                        score=9, is_breaking=True, category="military_conflict",
                        event_key="desert_compound_blast",
                        event_entities=["desert", "compound", "explosion"],
                        reason="Military incident", passed=True,
                    ))
                else:
                    evals.append(RelevanceEvaluation(
                        score=9, is_breaking=True, category="military_conflict",
                        event_key="east_baghdad_warehouse_airstrike",
                        event_entities=["baghdad", "warehouse", "airstrike"],
                        reason="Confirmed military airstrike", passed=True,
                    ))
            return evals

        async def _fake_translate(title, summary):
            if "Unconfirmed explosion" in title:
                return (
                    "تەقینەوەیەک لە بنکەیەکی بیاباندا ڕوویدا",
                    "ڕاپۆرتەکان ئاماژە بە بیستنی دەنگی تەقینەوەیەک لە ناوچە بیابانییەکە دەکەن.",
                )
            return (
                "هێرشێکی ئاسمانی کۆگایەکی سەربازی لە ڕۆژهەڵاتی بەغدا کردە ئامانج",
                "بەرپرسان پشتڕاستیان کردەوە کە هێرشێکی ئاسمانی کۆگایەکی سەربازی لە ڕۆژهەڵاتی بەغدا وێران کردووە.",
            )

        with patch.object(pipeline._ranker, "evaluate_batch_relevance", new=_fake_batch_eval), \
             patch.object(pipeline._extractor, "fetch_article_content_and_image", new=AsyncMock(return_value=("", None))), \
             patch.object(pipeline._translator, "translate_article", new=_fake_translate):
            res = await pipeline.process_batch([sab_dup, sab_rumor, reuters_main], max_publish=8)

        assert res["published"] == 2
        assert mock_bot.send_photo.await_count == 2

        # Reuters won the Baghdad warehouse event; Sabereen duplicate was rejected
        db_sab_dup = await temp_db.get_article_by_canonical_url(sab_dup.canonical_url)
        db_reuters = await temp_db.get_article_by_canonical_url(reuters_main.canonical_url)
        db_sab_rumor = await temp_db.get_article_by_canonical_url(sab_rumor.canonical_url)

        assert db_sab_dup is not None and db_sab_dup.status == ArticleStatus.REJECTED_DUPLICATE
        assert db_reuters is not None and db_reuters.status == ArticleStatus.PUBLISHED
        assert db_reuters.is_breaking is True
        assert not db_reuters.translated_summary.startswith(TIER3_ATTRIBUTION_PREFIX)

        assert db_sab_rumor is not None and db_sab_rumor.status == ArticleStatus.PUBLISHED
        assert db_sab_rumor.is_breaking is False
        assert db_sab_rumor.translated_summary.startswith(TIER3_ATTRIBUTION_PREFIX)

        # Verify captions delivered to Telegram (Reuters web link sits in inline button; Sabereen t.me stays in caption)
        sent_calls = mock_bot.send_photo.call_args_list
        captions_by_url = {c.kwargs["caption"]: c.kwargs["disable_notification"] for c in sent_calls}
        reuters_cap = next(cap for cap in captions_by_url if "Reuters" in cap)
        rumor_cap = next(cap for cap in captions_by_url if "Sabereen Plus" in cap)
        assert "🔴 بەپەلە | " in reuters_cap
        assert captions_by_url[reuters_cap] is False
        assert "🔴 بەپەلە | " not in rumor_cap
        assert captions_by_url[rumor_cap] is True
        assert TIER3_ATTRIBUTION_PREFIX in rumor_cap


# ═══════════════════════════════════════════════════
# 6. Rolling 2-of-8 Per-Source Dominance Cap
# ═══════════════════════════════════════════════════
class TestPerSourceDominanceCap:
    def test_five_sabereen_posts_capped_strictly_at_2(self):
        balancer = QuotaBalancer(batch_limit=8, bashur_per_8=5)
        sab_posts = [
            _make_article(
                art_id=100 + i,
                title=f"Sabereen report on Erbil KRG military coordination {i}",
                source_name="Sabereen News" if i % 2 == 0 else "Sabereen Plus",
                category="kurdistan",
                region="South Kurdistan",
                score=8.9 - i * 0.1,
            )
            for i in range(5)
        ]

        res = balancer.select_batch(sab_posts, limit=8)
        assert len(res.selected) == 2
        assert len(res.deferred) == 3

    def test_balanced_8_post_batch_respects_max_2_per_source_group(self):
        balancer = QuotaBalancer(batch_limit=8, bashur_per_8=5)
        candidates = (
            [
                _make_article(
                    art_id=10 + i,
                    title=f"Sabereen report on Baghdad Erbil talks {i}",
                    source_name="Sabereen News" if i % 2 == 0 else "Sabereen Plus",
                    category="kurdistan",
                    score=9.0 - i * 0.05,
                )
                for i in range(5)
            ]
            + [
                _make_article(
                    art_id=20 + i,
                    title=f"Rudaw reports Erbil parliament security law {i}",
                    source_name="Rudaw",
                    category="kurdistan",
                    score=8.5 - i * 0.05,
                )
                for i in range(3)
            ]
            + [
                _make_article(
                    art_id=30 + i,
                    title=f"Kurdistan24 reports Peshmerga Kirkuk deployment {i}",
                    source_name="Kurdistan24",
                    category="kurdistan",
                    score=8.4 - i * 0.05,
                )
                for i in range(3)
            ]
            + [
                _make_article(
                    art_id=40 + i,
                    title=f"Reuters covers NATO defense summit in Brussels {i}",
                    source_name="Reuters",
                    category="politics_world",
                    score=8.3 - i * 0.05,
                )
                for i in range(3)
            ]
        )

        res = balancer.select_batch(candidates, limit=8)
        assert len(res.selected) == 8
        counts: dict[str, int] = {}
        for art in res.selected:
            grp = get_source_group(art.source_name)
            counts[grp] = counts.get(grp, 0) + 1
        assert all(cnt <= 2 for cnt in counts.values())
        assert counts.get("sabereen", 0) <= 2

    def test_rolling_window_history_blocks_excess_sabereen_across_cycles(self):
        balancer = QuotaBalancer(batch_limit=8, bashur_per_8=5)
        recent_5 = [
            _make_article(1, "Sabereen security post A", source_name="Sabereen News", category="military_conflict"),
            _make_article(2, "Sabereen security post B", source_name="Sabereen Plus", category="military_conflict"),
            _make_article(3, "Erbil diplomatic summit", source_name="Rudaw", category="kurdistan"),
            _make_article(4, "Peshmerga patrol update", source_name="Kurdistan24", category="kurdistan"),
            _make_article(5, "UN Security Council talks", source_name="Reuters", category="politics_world"),
        ]
        candidates = [
            _make_article(10, "Third Sabereen post in window", source_name="Sabereen News", category="military_conflict", score=9.0),
            _make_article(11, "Fourth Sabereen post in window", source_name="Sabereen Plus", category="kurdistan", score=9.0),
            _make_article(12, "Sulaymaniyah security talks", source_name="Rudaw", category="kurdistan", score=8.2),
            _make_article(13, "NRT reports Duhok border coordination", source_name="NRT", category="kurdistan", score=8.1),
            _make_article(14, "BBC reports Geneva peace talks", source_name="BBC World", category="politics_world", score=8.0),
        ]

        # Next 3 slots + recent 5 = rolling 8-post window; Sabereen already at 2/8!
        res = balancer.select_batch(candidates, limit=3, recent_published=recent_5)
        assert len(res.selected) == 3
        assert all(get_source_group(a.source_name) != "sabereen" for a in res.selected)

    def test_tier3_breaking_cannot_bypass_cap_while_corroborated_tier1_can(self):
        balancer = QuotaBalancer(batch_limit=8, bashur_per_8=5)
        sab_candidates = [
            _make_article(
                art_id=200 + i,
                title=f"Sabereen breaking strike claim {i}",
                source_name="Sabereen News",
                category="military_conflict",
                score=9.6,
                is_breaking=True,  # Even if marked breaking, Tier-3 is NEVER exempt
            )
            for i in range(4)
        ]
        res = balancer.select_batch(sab_candidates, limit=8)
        assert len(res.selected) == 2  # Strictly capped at 2

        # Corroborated Tier-1 breaking item bypasses cap when replacing non-breaking
        recent_5 = [
            _make_article(1, "Reuters story A", source_name="Reuters", category="politics_world"),
            _make_article(2, "Reuters story B", source_name="Reuters", category="politics_world"),
            _make_article(3, "Rudaw Erbil news", source_name="Rudaw", category="kurdistan"),
            _make_article(4, "K24 Duhok news", source_name="Kurdistan24", category="kurdistan"),
            _make_article(5, "NRT Kirkuk news", source_name="NRT", category="kurdistan"),
        ]
        brk_reuters = _make_article(
            30, "Reuters emergency missile strike", source_name="Reuters",
            category="military_conflict", score=9.7, is_breaking=True,
        )
        other_candidates = [
            _make_article(31, "Duhok talks", source_name="Kurdistan24", category="kurdistan", score=8.2),
            _make_article(32, "BBC ceasefire update", source_name="BBC World", category="politics_world", score=8.1),
            _make_article(33, "UN Security Council session", source_name="UN News", category="politics_world", score=8.0),
        ]
        res2 = balancer.select_batch([brk_reuters] + other_candidates, limit=3, recent_published=recent_5)
        assert brk_reuters in res2.selected


# ═══════════════════════════════════════════════════
# 7. Kurdish Cultural / Art / Sports / Mundane Rejection
# ═══════════════════════════════════════════════════
class TestKurdishContentRejectionGuards:
    @pytest.mark.parametrize(
        "source_name,title",
        [
            ("Rudaw", "هونەرمەندێک 30 تابلۆی هونەری لە پێشانگایەکدا نمایش کرد"),
            ("Kurdistan24", "پێشانگای هونەری شێوەکاری لە شاری هەولێر کرایەوە"),
            ("NRT", "هەڵبژاردەی تۆپی پێی عێراق لە خولی نایاب سەرکەوتنی بەدەستهێنا"),
            ("Kurdistan24", "فیستیڤاڵی هونەری و کۆنسێرتی گۆرانی لە سلێمانی سازکرا"),
            ("NRT", "ڕووداوی هاتووچۆ لەسەر ڕێگای سەرەکی دهۆک ڕوویدا"),
        ],
    )
    def test_kurdish_art_culture_sports_traffic_rejected(self, source_name: str, title: str):
        art = _make_article(
            art_id=400,
            title=title,
            source_name=source_name,
            category="kurdistan",
            region="kurdistan_bashur",
            summary=f"{title}. چالاکییەکی ناوخۆیی بوو و خەڵکێکی زۆر بەشدارییان تێدا کرد.",
        )
        assert check_blocked_topic(art) is not None
        assert check_pre_filter_off_topic(art) is not None

    def test_genuine_kurdish_security_and_politics_passes(self):
        art = _make_article(
            art_id=401,
            title="هێزەکانی پێشمەرگە و سوپای عێراق لەسەر ئاسایشی سنووری کەرکووک کۆبوونەوە",
            source_name="Rudaw",
            category="kurdistan",
            region="kurdistan_bashur",
            summary="فەرماندەکانی پێشمەرگە و وەزارەتی بەرگری عێراق لە هەولێر پلانێکی ئەمنییان پەسەند کرد.",
        )
        assert check_blocked_topic(art) is None
        assert check_pre_filter_off_topic(art) is None


# ═══════════════════════════════════════════════════
# 8. Telegram Multi-Channel Failure Isolation
# ═══════════════════════════════════════════════════
class TestTelegramChannelFailureIsolation:
    @pytest.mark.asyncio
    async def test_one_channel_timeout_does_not_block_other_channels(self):
        import asyncio

        src = TelegramSource()
        src._session = MagicMock(closed=False)
        fresh_ts = _iso_minutes_ago(15)

        def _channel_html(username: str, post_id: int, text: str) -> str:
            return f"""
            <div class="tgme_widget_message_wrap">
              <div class="tgme_widget_message" data-post="{username}/{post_id}">
                <div class="tgme_widget_message_text">{text}</div>
                <div class="tgme_widget_message_footer">
                  <a class="tgme_widget_message_date" href="https://t.me/{username}/{post_id}">
                    <time datetime="{fresh_ts}">12:00</time>
                  </a>
                </div>
              </div>
            </div>"""

        async def _fake_fetch_channel(ch_cfg: dict[str, str]):
            uname = ch_cfg["username"]
            if uname == "nrttvofficial":
                raise asyncio.TimeoutError("Simulated timeout for NRT")
            return src.parse_channel_html(
                _channel_html(
                    uname,
                    101,
                    f"{ch_cfg['name']} dispatch confirming military coordination and border talks in Erbil and Baghdad.",
                ),
                ch_cfg,
            )

        src.fetch_channel = _fake_fetch_channel  # type: ignore[method-assign]
        articles = await src.fetch_articles()

        sources_returned = {a.source_name for a in articles}
        assert len(articles) == 5
        assert "NRT" not in sources_returned
        assert {"Al Jazeera", "Rudaw", "Kurdistan24", "Liveuamap", "Clash Report"}.issubset(sources_returned)


# ═══════════════════════════════════════════════════
# 9. Zero Extra Gemini API Calls & Engine Wiring
# ═══════════════════════════════════════════════════
class TestZeroExtraGeminiCallsAndEngineWiring:
    @pytest.mark.asyncio
    async def test_source_verifier_makes_zero_gemini_calls_during_engine_cycle(self, temp_db, tmp_path):
        mock_bot = AsyncMock()
        mock_bot.send_photo = AsyncMock(return_value=MagicMock(message_id=9001))

        publisher = TelegramPublisher()
        publisher._channel_id = "@KurdishGlobalNews"
        publisher._bot = mock_bot
        publisher._image_handler.download_image = AsyncMock(return_value=None)
        fake_card = tmp_path / "card.jpg"
        fake_card.write_bytes(b"JPG")
        publisher._card_generator.generate_card_file = AsyncMock(return_value=fake_card)

        fresh_ts = _iso_minutes_ago(2)
        raw_items = [
            RawArticle(
                url="https://blog.example.org/airbase-strike-8801",
                title="Missiles strike regional airbase near Baghdad",
                summary=(
                    "Local channel reports emergency missile strikes at a regional airbase near Baghdad "
                    "triggering air defense sirens across neighboring sectors and heightened military "
                    "deployment around key installations in central Iraq overnight."
                ),
                source_name="Sabereen News",
                publication_date=fresh_ts,
            ),
            RawArticle(
                url="https://reuters.com/world/airbase-strike-8802",
                title="Missiles strike regional airbase near Baghdad, defense ministry says",
                summary=(
                    "Reuters reports Iraq's defense ministry confirmed missile strikes hitting an airbase "
                    "near Baghdad overnight, prompting emergency response units and radar inspections across "
                    "nearby security perimeters in coordination with regional commands."
                ),
                source_name="Reuters",
                publication_date=fresh_ts,
            ),
        ]
        mock_src = MagicMock()
        mock_src.name = "MockSource"
        mock_src.source_type = "rss"
        mock_src.enabled = True
        mock_src.fetch_articles = AsyncMock(return_value=raw_items)

        engine = NewsEngine()
        engine._db = temp_db
        engine._publisher = publisher
        engine._sources = [mock_src]
        engine._pipeline = ArticlePipeline(temp_db, publisher, engine._health, engine._metrics)
        engine._pipeline._extractor.fetch_article_content_and_image = AsyncMock(return_value=("", None))
        engine._pipeline._semantic_dedup = SemanticDeduplicator(temp_db, api_key="", base_backoff=0.0)

        async def _fake_batch_eval(articles, batch_size=12):
            return [
                RelevanceEvaluation(
                    score=9, is_breaking=True, category="military_conflict",
                    event_key="baghdad_airbase_missile_strike",
                    event_entities=["baghdad", "airbase", "missile"],
                    reason="Strike confirmed", passed=True,
                )
                for _ in articles
            ]

        with patch.object(GeminiClient, "generate_json", new=AsyncMock()) as mock_gemini_json, \
             patch.object(
                 engine._pipeline._ranker,
                 "evaluate_batch_relevance",
                 new=_fake_batch_eval,
             ), \
             patch.object(
                 engine._pipeline._translator,
                 "translate_article",
                 new=AsyncMock(return_value=(
                     "هێرشی مووشەکی کرایە سەر بنکەیەکی ئاسمانی نزیک بەغدا",
                     "وەزارەتی بەرگری پشتڕاستی کردەوە کە مووشەکەکان بەر بنکە ئاسمانییەکە کەوتوون.",
                 )),
             ):
            await engine._discovery_cycle()

        # Zero direct GeminiClient.generate_json calls from source_verifier / dedup / pacing / card_generator
        mock_gemini_json.assert_not_awaited()
        # Reuters published cleanly as Tier-1 breaking (corroborated by Tier-3 blog), duplicate dropped
        reuters_db = await temp_db.get_article_by_canonical_url("https://reuters.com/world/airbase-strike-8802")
        sab_db = await temp_db.get_article_by_canonical_url("https://blog.example.org/airbase-strike-8801")
        assert reuters_db is not None and reuters_db.status == ArticleStatus.PUBLISHED
        assert reuters_db.is_breaking is True
        assert sab_db is not None and sab_db.status == ArticleStatus.REJECTED_DUPLICATE
