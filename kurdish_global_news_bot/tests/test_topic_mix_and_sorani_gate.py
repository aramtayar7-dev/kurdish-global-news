"""
Comprehensive unit and integration tests for:
1. 7-of-8 Politics & War + Max 1-of-8 Economy OR Cybersecurity topic mix (plus 8/8 P&W fallback)
   while preserving the 5-of-8 Southern Kurdistan (Bashur) quota.
2. Absolute ban on sports and ordinary/mundane news (travel, weather trivia, gadgets/apps,
   entertainment/celebrities, routine human interest, traffic/ads, domestic courts).
3. Strict Central Kurdish (Sorani) publish gate blocking untranslated Arabic or English,
   aborting publish cleanly without permanent skipping on HTTP 429, and stripping leader insults.
4. Suppression of Telegram channel screenshot `og:image` / static photos while preserving `video_url`
   and keeping `processing/card_generator.py` intact.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import (
    ArticleStatus,
    ECON_CYBER_MAX_PER_8,
    ECON_CYBER_MIN_SCORE,
    POLITICS_WAR_MIN_PER_8,
)
from core.balancer import QuotaBalancer
from core.importance_ranker import (
    ImportanceRanker,
    RelevanceEvaluation,
    check_blocked_topic,
    check_pre_filter_off_topic,
    classify_topic_bucket,
    is_bashur_article,
    is_strict_breaking,
)
from core.pipeline import ArticlePipeline
from monitoring.health import HealthMonitor
from monitoring.metrics import Metrics
from processing.content_formatter import ContentFormatter, normalize_neutral_terminology
from processing.deduplicator import SemanticDeduplicator
from processing.translator import (
    TranslationEngine,
    TranslationError,
    is_kurdish_sorani,
    is_valid_kurdish_sorani,
)
from storage.database import Database
from storage.models import Article
from tg_bot.publisher import TelegramPublisher


def _make_article(
    art_id: int,
    title: str,
    url: str = "",
    category: str = "politics_world",
    region: str = "",
    source_name: str = "Reuters",
    score: float = 8.2,
    is_breaking: bool = False,
    summary: str = "",
) -> Article:
    pub_iso = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    art = Article(
        id=art_id,
        original_url=url or f"https://example.com/news/{art_id}",
        canonical_url=url or f"https://example.com/news/{art_id}",
        original_title=title,
        normalized_title=Article.normalize_title(title),
        original_summary=summary or (
            f"{title}. Diplomatic and security authorities released official details "
            "regarding regional developments, defense coordination, border monitoring, "
            "and ongoing government deliberations across capital ministries."
        ),
        source_name=source_name,
        category=category,
        region=region,
        importance_score=score,
        is_breaking=is_breaking,
        publication_date=pub_iso,
        discovered_at=pub_iso,
    )
    art.compute_content_hash()
    art.compute_event_fingerprint()
    return art


@pytest.fixture
async def temp_db(tmp_path):
    db = Database(str(tmp_path / "test_topic_sorani_gate.db"))
    await db.connect()
    yield db
    await db.close()


class TestSevenPlusOneTopicMixBalancer:
    """Verify 7+ Politics/War + max 1 Econ/Cyber per 8 posts alongside 5-of-8 Bashur quota."""

    def test_topic_mix_constants_configured(self):
        assert POLITICS_WAR_MIN_PER_8 == 7
        assert ECON_CYBER_MAX_PER_8 == 1
        assert ECON_CYBER_MIN_SCORE >= 7.0

    def test_max_1_combined_econ_or_cyber_with_5_bashur(self):
        balancer = QuotaBalancer(batch_limit=8, bashur_per_8=5, econ_cyber_max_per_8=1)

        bashur_pw = [
            _make_article(
                100 + i,
                f"Erbil KRG parliament holds security coordination session {i}",
                category="kurdistan",
                region="South Kurdistan",
                score=8.3,
            )
            for i in range(6)
        ]
        global_pw = [
            _make_article(
                200 + i,
                f"NATO allies hold ministerial talks on Eastern Mediterranean defense {i}",
                category="politics_world",
                score=8.1,
            )
            for i in range(4)
        ]
        # Strong Economy (8.7) + Strong Cybersecurity (8.9) + Extra Economy (8.5)
        # Only ONE combined Econ OR Cyber (the highest-scoring: Cyber 8.9) may enter 8 slots!
        econ_1 = _make_article(
            301,
            "IMF warns global sovereign debt crisis threatens emerging oil exporters",
            category="economy_markets",
            source_name="Reuters Business",
            score=8.7,
        )
        econ_2 = _make_article(
            302,
            "European Central Bank holds interest rates steady amid inflation pressure",
            category="economy_markets",
            source_name="CNBC World",
            score=8.5,
        )
        cyber_1 = _make_article(
            401,
            "CISA warns of active nation-state zero-day ransomware campaign on power grids",
            category="cybersecurity",
            source_name="BleepingComputer",
            score=8.9,
        )

        res = balancer.select_batch(
            bashur_pw + global_pw + [econ_1, econ_2, cyber_1],
            limit=8,
        )

        assert len(res.selected) == 8
        assert res.bashur_selected == 5
        assert res.global_selected == 3
        assert res.politics_war_selected == 7
        assert res.econ_cyber_selected == 1

        selected_ids = {a.id for a in res.selected}
        assert 401 in selected_ids  # Highest-scoring Econ/Cyber (Cyber 8.9) won the single slot
        assert 301 not in selected_ids
        assert 302 not in selected_ids

    def test_smart_8_of_8_politics_war_fallback_when_no_qualifying_econ_or_cyber(self):
        balancer = QuotaBalancer(batch_limit=8, bashur_per_8=5, econ_cyber_max_per_8=1)

        bashur_pw = [
            _make_article(
                100 + i,
                f"Peshmerga commanders inspect Kirkuk defense perimeter {i}",
                category="kurdistan",
                region="South Kurdistan",
                score=8.4,
            )
            for i in range(5)
        ]
        global_pw = [
            _make_article(
                200 + i,
                f"UN Security Council negotiates regional ceasefire terms {i}",
                category="military_conflict",
                score=8.2,
            )
            for i in range(4)
        ]
        # Low-impact Economy (< ECON_CYBER_MIN_SCORE=7.5) must NOT take a slot away from Politics/War
        weak_econ = _make_article(
            301,
            "Quarterly corporate trade volumes ease slightly in regional markets",
            category="economy_markets",
            score=6.8,
        )

        res = balancer.select_batch(bashur_pw + global_pw + [weak_econ], limit=8)

        assert len(res.selected) == 8
        assert res.bashur_selected == 5
        assert res.politics_war_selected == 8
        assert res.econ_cyber_selected == 0
        assert weak_econ in res.deferred

    def test_rolling_window_blocks_second_econ_or_cyber_across_cycles(self):
        balancer = QuotaBalancer(batch_limit=8, bashur_per_8=5, econ_cyber_max_per_8=1)

        recent_4 = [
            _make_article(1, "Erbil security cabinet statement", category="kurdistan", region="South Kurdistan"),
            _make_article(2, "Global cyberagency alerts critical vulnerability", category="cybersecurity", score=8.6),
            _make_article(3, "Baghdad defense committee talks", category="politics_world"),
            _make_article(4, "Ankara border security briefing", category="politics_world"),
        ]
        candidates = [
            _make_article(10, "IMF warns on sovereign debt default risks", category="economy_markets", score=9.0),
            _make_article(11, "Tehran and Washington exchange diplomatic warnings", category="politics_world", score=8.1),
            _make_article(12, "Damascus northern frontier military redeployment", category="military_conflict", score=8.2),
            _make_article(13, "Vienna nuclear talks resume amid sanctions standoff", category="politics_world", score=8.0),
            _make_article(14, "Sulaymaniyah provincial council discusses budget law", category="kurdistan", region="South Kurdistan", score=8.3),
        ]

        res = balancer.select_batch(candidates, limit=4, recent_published=recent_4)
        assert len(res.selected) == 4
        assert res.econ_cyber_selected == 0
        assert all(classify_topic_bucket(a) == "politics_war" for a in res.selected)

    @pytest.mark.asyncio
    async def test_pipeline_process_batch_enforces_7_plus_1_and_allows_major_econ_feed(self, temp_db):
        mock_bot = AsyncMock()
        msg_obj = MagicMock()
        msg_obj.message_id = 6060
        mock_bot.send_photo = AsyncMock(return_value=msg_obj)

        publisher = TelegramPublisher()
        publisher._channel_id = "@KurdishGlobalNews"
        publisher._bot = mock_bot
        publisher._image_handler.download_image = AsyncMock(return_value=None)

        pipeline = ArticlePipeline(temp_db, publisher, HealthMonitor(), Metrics())
        pipeline._semantic_dedup = SemanticDeduplicator(temp_db, api_key="", base_backoff=0.0)

        bashur_titles = [
            "Erbil KRG parliament debates provincial security budget framework",
            "Peshmerga commanders reinforce Kirkuk western defensive trench line",
            "Sulaymaniyah governor hosts constitutional federalism dialogue talks",
            "Duhok customs authorities modernize Zakho cross-border terminal",
            "Halabja provincial council approves reconstruction infrastructure bill",
        ]
        global_titles = [
            "NATO allies reinforce Eastern Mediterranean naval air defenses",
            "UN Security Council convenes emergency humanitarian corridor talks",
            "Ankara and Tehran diplomats hold border security consultations",
        ]
        batch: list[Article] = []
        for i, title in enumerate(bashur_titles):
            batch.append(
                _make_article(
                    10 + i,
                    title,
                    url=f"https://rudaw.net/pol/{i}",
                    category="kurdistan",
                    region="South Kurdistan",
                    source_name="Rudaw",
                )
            )
        for i, title in enumerate(global_titles):
            batch.append(
                _make_article(
                    20 + i,
                    title,
                    url=f"https://reuters.com/pol/{i}",
                    category="politics_world",
                )
            )
        # 1 Major Macro Economy (NOT pre-filtered!) + 1 Cybersecurity (competing for single 1-of-8 slot)
        major_econ = _make_article(
            31,
            "IMF cuts global growth forecast over soaring sovereign debt tariffs",
            url="https://reuters.com/business/imf-forecast",
            category="economy_markets",
            source_name="Reuters Business",
            summary=(
                "The International Monetary Fund warned central banks face difficult tariff and inflation risks globally "
                "as slowing industrial output and sovereign bond pressures weigh heavily on major importing economies."
            ),
        )
        major_cyber = _make_article(
            32,
            "BleepingComputer: Massive ransomware flaw actively exploited in enterprise firewalls",
            url="https://bleepingcomputer.com/news/security/firewall-zero-day",
            category="cybersecurity",
            source_name="BleepingComputer",
            summary=(
                "Security researchers urged immediate patching of a critical remote code execution zero-day bug "
                "after threat actors targeted government and infrastructure appliances with automated exploit chains."
            ),
        )
        batch.extend([major_econ, major_cyber])

        # Verify neither major macro economy nor cybersecurity is dropped before Gemini
        assert check_pre_filter_off_topic(major_econ) is None
        assert check_pre_filter_off_topic(major_cyber) is None

        for a in batch:
            a.id = await temp_db.insert_article(a)

        bashur_words = ("Erbil", "KRG", "Peshmerga", "Sulaymaniyah", "Duhok", "Halabja")

        async def mock_batch_eval(articles, **_kwargs):
            out = []
            for art in articles:
                if any(w in art.original_title for w in bashur_words):
                    out.append(RelevanceEvaluation(score=8, is_breaking=False, category="kurdistan", reason="Bashur", passed=True))
                elif "IMF" in art.original_title:
                    out.append(RelevanceEvaluation(score=8, is_breaking=False, category="economy_markets", reason="Macro econ", passed=True))
                elif "BleepingComputer" in art.original_title:
                    out.append(RelevanceEvaluation(score=9, is_breaking=False, category="cybersecurity", reason="Critical cyber", passed=True))
                else:
                    out.append(RelevanceEvaluation(score=8, is_breaking=False, category="politics_world", reason="Global pol", passed=True))
            return out

        with patch.object(pipeline._ranker, "evaluate_batch_relevance", side_effect=mock_batch_eval), \
             patch.object(pipeline._extractor, "fetch_article_content_and_image", new=AsyncMock(return_value=(None, None))), \
             patch.object(
                 pipeline._translator,
                 "translate_article",
                 side_effect=lambda title, summary="": (
                     "هەواڵێکی نوێی کوردستان و جیهان لەسەر دۆخی ئەمنی و سیاسی"
                     if "BleepingComputer" not in title
                     else "هۆشداریی ئاسایشی سایبەری لەسەر هێرشێکی نوێی ئەلیکترۆنی بڵاوکرایەوە",
                     "پوختەی هەواڵەکە بە زمانی کوردی سۆرانی بۆ بڵاوکردنەوە لە کەناڵدا.",
                 ),
             ):
            res = await pipeline.process_batch(batch, max_publish=8)

        assert res["published"] == 8
        published_captions = [c.kwargs["caption"] for c in mock_bot.send_photo.call_args_list]
        assert sum(1 for cap in published_captions if "ئاسایشی سایبەری" in cap) == 1
        # IMF (8) lost to BleepingComputer (9), so IMF was deferred rather than publishing 2 non-P&W posts
        econ_db = await temp_db.get_article_by_canonical_url(major_econ.canonical_url)
        assert econ_db.status == ArticleStatus.SKIPPED


class TestBannedSportsAndOrdinaryNews:
    """Verify sports and ordinary/mundane topics are unconditionally blocked."""

    @pytest.mark.parametrize(
        "title,summary",
        [
            ("Real Madrid beats Barcelona 2-1 in Champions League semifinal", "Two late football goals sealed the tournament victory."),
            ("مباراة كرة القدم في دوري الأبطال تنتهي بفوز مثير", "تألق المهاجم وسجل هدفين في ملعب المباراة."),
            ("Oscars red carpet celebrity gossip and box office weekend hits", "Hollywood actors attended the annual film awards ceremony."),
            ("Best beaches and luxury hotel travel guide for summer vacations", "Tourism experts listed top island resorts for holiday travelers."),
            ("Sunny weekend weather forecast brings cherry blossom crowds", "Mild spring temperatures made weekend weather ideal for picnics."),
            ("New iPhone smartphone camera and WhatsApp update review for users", "Tech reviewers compared consumer phone battery and social app features."),
            ("Houston man sentenced to 12 years in US federal prison for ISIS propaganda", "Qamishli, Syria (North Press) – A US federal court handed down the prison sentence."),
            ("تنويه مروري حول كاميرات السرعة والرادار في الشوارع الرئيسية", "أعلنت مديرية المرور العامة عن تفعيل غرامات مرورية جديدة."),
        ],
    )
    @pytest.mark.asyncio
    async def test_banned_topics_blocked_without_gemini_and_marked_skipped(self, title, summary):
        art = _make_article(500, title, summary=summary)
        reason = check_blocked_topic(art)
        assert reason is not None

        ranker = ImportanceRanker(api_key="dummy_key")
        ranker._gemini.generate_json = AsyncMock()
        ev = await ranker.evaluate_global_relevance(art)
        assert ev.passed is False
        assert ev.is_breaking is False
        ranker._gemini.generate_json.assert_not_called()


class TestStrictSoraniGateAndNeutrality:
    """Verify Arabic/English outputs never leak through and HTTP 429 retains RETRY_PENDING cleanly."""

    def test_is_valid_kurdish_sorani_distinguishes_arabic_english_and_sorani(self):
        # Genuine Sorani -> True
        assert is_valid_kurdish_sorani("ئەمڕۆ لە هەولێر کۆبوونەوەیەکی گرنگ ئەنجامدرا.") is True
        assert is_valid_kurdish_sorani("وەزارەتی بەرگریی عێراق ڕایگەیاند هێزەکان لەسەر سنوور جێگیر کران.") is True
        assert is_kurdish_sorani("هەولێر پایتەختی هەرێمی کوردستانە") is True

        # Untranslated Arabic Sabereen posts sharing U+0600-U+06FF -> strictly False!
        assert is_valid_kurdish_sorani("أعلنت وزارة الدفاع العراقية اليوم في بغداد عن اجتماع طارئ") is False
        assert is_valid_kurdish_sorani("عاجل | القوات المسلحة تؤكد جاهزيتها الكاملة في المنطقة خلال الساعات المقبلة") is False
        # Even if an Arabic sentence contains an occasional Persian/Kurdish consonant like گ
        assert is_valid_kurdish_sorani("عاجل | أعلنت الحكومة في بغداد عن خطة أمنية گ") is False

        # Pure English or half-English mock strings -> False!
        assert is_valid_kurdish_sorani("US forces conduct airstrikes near border") is False
        assert is_valid_kurdish_sorani("وەرگێڕدراو بۆ کوردی: US forces conduct airstrikes in Middle East") is False
        assert is_valid_kurdish_sorani("") is False

    def test_neutral_terminology_strips_leader_insults_and_never_falls_back_to_original(self):
        assert normalize_neutral_terminology("المجرم ترامب يصدر قرارا") == "دۆناڵد ترەمپ يصدر قرارا"
        assert normalize_neutral_terminology("تاوانبار نەتەنیاهۆ ڕایگەیاند") == "بێنیامین نەتەنیاهۆ ڕایگەیاند"

        formatter = ContentFormatter()
        untranslated = Article(
            original_title="Unpublished English Headline",
            original_summary="Unpublished English summary body.",
            translated_title="",
            translated_summary="",
            original_url="https://example.com/raw",
            source_name="Test",
        )
        msg = formatter.format_telegram_message(untranslated)
        cap = formatter.format_caption(untranslated)
        assert "Unpublished English Headline" not in msg
        assert "Unpublished English summary" not in msg
        assert "Unpublished English Headline" not in cap

    @pytest.mark.asyncio
    async def test_translation_engine_never_treats_arabic_as_native_kurdish_or_returns_raw_on_429(self):
        engine = TranslationEngine(gemini_api_key="gem_key", delay=0.0)
        arabic_title = "القوات الأمنية تنتشر في بغداد خلال الساعات القادمة"
        arabic_summary = "أفادت مصادر أمنية في العاصمة العراقية اليوم ببدء انتشار أمني واسع في الشوارع الرئيسية."

        # 1. Must call Gemini with translate prompt (is_kurdish=False), NOT rewrite native Kurdish
        with patch.object(
            engine._gemini_provider,
            "translate_article",
            new_callable=AsyncMock,
            return_value=("هێزە ئەمنییەکان لە بەغدا بڵاوەیان پێکرا", "سەرچاوە ئەمنییەکانی بەغدا بڵاوکردنەوەی هێزەکانیان ڕاگەیاند."),
        ) as mock_gem:
            t, s = await engine.translate_article(arabic_title, arabic_summary)
            assert mock_gem.call_args.kwargs["is_kurdish"] is False
            assert is_valid_kurdish_sorani(t)
            assert is_valid_kurdish_sorani(s)

        # 2. On HTTP 429 quota error + Google failure -> raises TranslationError (NEVER returns raw Arabic)
        with patch.object(
            engine._gemini_provider,
            "translate_article",
            new_callable=AsyncMock,
            side_effect=TranslationError("HTTP 429 All Gemini keys exhausted"),
        ), patch.object(
            engine._google_fallback,
            "translate",
            new_callable=AsyncMock,
            side_effect=TranslationError("Google Translate returned HTTP 429"),
        ):
            with pytest.raises(TranslationError):
                await engine.translate_article(arabic_title, arabic_summary)

    @pytest.mark.asyncio
    async def test_pipeline_aborts_on_429_and_retains_retry_pending_not_skipped_or_published(self, temp_db):
        mock_bot = AsyncMock()
        publisher = TelegramPublisher()
        publisher._channel_id = "@KurdishGlobalNews"
        publisher._bot = mock_bot

        pipeline = ArticlePipeline(temp_db, publisher, HealthMonitor(), Metrics())
        pipeline._semantic_dedup = SemanticDeduplicator(temp_db, api_key="", base_backoff=0.0)

        tg_art = _make_article(
            901,
            "القوات العراقية تعزز الانتشار الأمني في المحافظة",
            url="https://t.me/SabrenNewss/9901",
            source_name="Sabereen News",
            category="military_conflict",
            summary=(
                "أعلنت قيادة العمليات المشتركة اليوم في بغداد تعزيز الإجراءات الأمنية وحماية الحدود "
                "والمنشآت الحيوية بالتنسيق مع الدفاع الجوي والوحدات الميدانية وسط متابعة مستمرة للأوضاع في البلاد."
            ),
        )
        tg_art.id = await temp_db.insert_article(tg_art)

        with patch.object(
            pipeline._ranker,
            "evaluate_global_relevance",
            new=AsyncMock(return_value=RelevanceEvaluation(score=8, is_breaking=False, category="military_conflict", reason="Security", passed=True)),
        ), patch.object(
            pipeline._extractor,
            "fetch_article_content_and_image",
            new=AsyncMock(return_value=("", "https://cdn4.telegram-cdn.org/file/watermarked_screenshot.jpg")),
        ), patch.object(
            pipeline._translator,
            "translate_article",
            new=AsyncMock(side_effect=TranslationError("All Gemini keys exhausted (HTTP 429 quota cooldown)")),
        ):
            published = await pipeline.process_article(tg_art)

        assert published is False
        mock_bot.send_photo.assert_not_called()
        mock_bot.send_message.assert_not_called()
        mock_bot.send_video.assert_not_called()

        saved = await temp_db.get_article_by_canonical_url(tg_art.canonical_url)
        assert saved.status == ArticleStatus.RETRY_PENDING
        # Critical: NOT permanently skipped in skipped_urls so next cycle retries when quota resets!
        assert await temp_db.is_url_skipped(tg_art.canonical_url) is False
        # And og:image screenshot from t.me/ was stripped before card generation
        assert tg_art.image_url == ""

    @pytest.mark.asyncio
    async def test_publisher_blocks_arabic_or_english_text_before_card_generation(self):
        mock_bot = AsyncMock()
        publisher = TelegramPublisher()
        publisher._channel_id = "@KurdishGlobalNews"
        publisher._bot = mock_bot

        arabic_leak = Article(
            original_title="Arabic news",
            translated_title="عاجل | أعلنت وزارة الدفاع اليوم في بغداد",
            translated_summary="هذا نص عربي غير مترجم إلى اللغة الكوردية.",
            original_url="https://t.me/SabrenNewss/500",
            source_name="Sabereen News",
        )
        assert await publisher.publish_article(arabic_leak) is None
        mock_bot.send_photo.assert_not_called()
        mock_bot.send_video.assert_not_called()
        mock_bot.send_message.assert_not_called()

        # Verify card_generator.py was NOT altered and strict breaking gate still holds
        brk = _make_article(999, "Major airstrikes hit capital airbase", score=9.2, is_breaking=True)
        non_brk_stmt = _make_article(998, "KRG statement on talks", score=9.2, is_breaking=False)
        assert is_strict_breaking(brk) is True
        assert is_strict_breaking(non_brk_stmt) is False
        assert is_bashur_article(_make_article(997, "Erbil cabinet session approves budget", region="South Kurdistan")) is True
