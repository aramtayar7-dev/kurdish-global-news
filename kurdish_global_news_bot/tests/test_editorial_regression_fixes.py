"""
Regression tests for the 4 production editorial/content fixes:
  1. Content-based Kurdistan classification (NEVER classify by source name/feed tag alone).
  2. Reject celebrity / Hollywood / actor / entertainment obituaries across EN/Sorani/Arabic.
  3. Strip RSS footer boilerplate ('Continue reading...', 'بەردەوام بە لە خوێندنەوە...', '[...]')
     and reject multi-story news roundups / podcast briefings (e.g., NPR Up First).
  4. Translation & Sorani cleanup (no ', و,' literal artifacts, no boilerplate footers,
     single-person domestic legal/immigration cases blocked).
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from config import ArticleStatus
from core.balancer import QuotaBalancer
from core.importance_ranker import (
    ImportanceRanker,
    RelevanceEvaluation,
    check_blocked_topic,
    check_pre_filter_off_topic,
    is_bashur_article,
    is_genuine_kurdistan_event,
    is_multi_story_roundup,
    normalize_category,
)
from core.pipeline import ArticlePipeline
from monitoring.health import HealthMonitor
from monitoring.metrics import Metrics
from processing.background_selector import THEME_KURDISTAN_BASHUR, select_background_theme
from processing.content_formatter import (
    ContentFormatter,
    normalize_neutral_terminology,
    strip_rss_boilerplate,
)
from processing.extractor import ArticleExtractor
from processing.translator import GeminiTranslationProvider
from sources.base import RawArticle
from sources.rss_source import parse_rss_entry
from storage.database import Database
from storage.models import Article


def _make_article(
    title: str,
    summary: str = "",
    source_name: str = "Reuters",
    category: str = "politics_world",
    region: str = "",
    url: str = "https://example.com/news/1",
    score: float = 8.2,
) -> Article:
    pub_iso = datetime.now(timezone.utc).isoformat()
    art = Article(
        original_url=url,
        canonical_url=Article.canonicalize_url(url),
        original_title=title,
        normalized_title=Article.normalize_title(title),
        original_summary=summary or (
            f"{title}. Diplomatic officials issued statements regarding ongoing international "
            "consultations and government policy measures across major capitals."
        ),
        source_name=source_name,
        category=category,
        region=region,
        importance_score=score,
        publication_date=pub_iso,
        discovered_at=pub_iso,
    )
    art.compute_content_hash()
    art.compute_event_fingerprint()
    return art


class TestBug1ContentBasedKurdistanClassification:
    @pytest.mark.asyncio
    async def test_kurdistan24_china_us_trade_not_classified_as_kurdistan_or_bashur(self, tmp_path):
        title = "بەیژینگ داوا لە واشنتن دەکات سەقامگیریی پەیوەندییە بازرگانییەکانی نێوانیان بپارێزێت"
        summary = (
            "وەزارەتی بازرگانی چین داوای لە حکوومەتی ئەمریکا کرد ناکۆکییە گومرگییەکان چارەسەر بکەن "
            "و سەقامگیریی پەیوەندییە ئابووری و بازرگانییەکانی نێوان هەردوو وڵات بپارێزن. "
            "کوردستان24"
        )
        art = _make_article(
            title=title,
            summary=summary,
            source_name="Kurdistan24 Sorani",
            category="kurdistan",
            region="South Kurdistan",
            url="https://www.kurdistan24.net/ckb/story/china-us-trade-1",
        )

        # Must NEVER treat Beijing-Washington trade from Kurdistan24 as Kurdistan/Bashur
        assert is_genuine_kurdistan_event(art) is False
        assert is_bashur_article(art) is False
        inferred = normalize_category(art.category, text_hint=f"{title} {summary}")
        assert inferred in {"economy_markets", "politics_world"}
        assert inferred != "kurdistan"
        assert select_background_theme(art) != THEME_KURDISTAN_BASHUR

        # QuotaBalancer must count it as Global, NOT consuming a 5-of-8 Bashur slot
        balancer = QuotaBalancer(batch_limit=8, bashur_per_8=5)
        res = balancer.select_batch([art], limit=8)
        assert res.bashur_selected == 0
        assert res.global_selected == 1

        # Even if Gemini fails over to fallback or mistakenly suggests "kurdistan", pipeline fixes it
        db = Database(str(tmp_path / "k24_trade.db"))
        await db.connect()
        try:
            art.id = await db.insert_article(art)
            pipeline = ArticlePipeline(db, AsyncMock(), HealthMonitor(), Metrics())

            ranker = ImportanceRanker(api_key="")
            fallback_eval = ranker._rule_based_fallback(art, "Test fallback")
            assert fallback_eval.category in {"economy_markets", "politics_world"}
            assert fallback_eval.category != "kurdistan"

            # Even if Gemini output claims category="kurdistan", _build_evaluation_from_payload overrides it
            built = ranker._build_evaluation_from_payload(
                {"score": 8, "is_breaking": False, "category": "kurdistan", "reason": "Trade talks"},
                art,
                threshold=7,
            )
            assert built.category in {"economy_markets", "politics_world"}
            assert built.category != "kurdistan"

            ok = await pipeline._apply_relevance_evaluation(art, fallback_eval)
            assert ok is True
            assert art.category in {"economy_markets", "politics_world"}
            assert art.category != "kurdistan"
            assert art.region == ""
        finally:
            await db.close()

    def test_kurdistan24_erbil_peshmerga_is_classified_as_kurdistan_and_bashur(self):
        art = _make_article(
            title="حکوومەتی هەرێمی کوردستان و هێزەکانی پێشمەرگە لە هەولێر کۆبوونەوە",
            summary=(
                "لە شاری هەولێر کۆبوونەوەیەکی ئەمنی سەبارەت بە دۆخی سنووری کەرکووک و ئەرکی پێشمەرگە "
                "و مووچەی فەرمانبەرانی هەرێمی کوردستان بەڕێوەچوو."
            ),
            source_name="Kurdistan24 Sorani",
            category="kurdistan",
            region="South Kurdistan",
            url="https://www.kurdistan24.net/ckb/story/erbil-krg-2",
        )
        assert is_genuine_kurdistan_event(art) is True
        assert is_bashur_article(art) is True
        assert normalize_category(art.category, text_hint=f"{art.original_title} {art.original_summary}") == "kurdistan"
        assert select_background_theme(art) == THEME_KURDISTAN_BASHUR


class TestBug2CelebrityHollywoodRejection:
    @pytest.mark.parametrize(
        "title,summary",
        [
            (
                "ئەکتەری دێرین لە هۆلیوودی ئەمریکا لە تەمەنی ٧٩ ساڵیدا کۆچی دوایی کرد",
                "ئەکتەرە بەناوبانگەکەی سینەمای هۆلیوود دوای چەندین ساڵ کارکردن لە فیلمە جیهانییەکان کۆچی دوایی کرد.",
            ),
            (
                "Veteran Hollywood actor dies at 79 after decades of acclaimed films",
                "Friends and co-stars from cinema and Broadway paid tribute to the Oscar-winning movie star.",
            ),
            (
                "وفاة ممثل هوليوود الشهير عن عمر ناهز 79 عاما",
                "نعت الأوساط الفنية والسينمائية النجم الذي شارك في عشرات الأفلام الأمريكية.",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_hollywood_actor_obituary_rejected_in_all_languages(self, title: str, summary: str):
        art = _make_article(title=title, summary=summary, source_name="Kurdistan24 Sorani")
        assert check_blocked_topic(art) is not None
        assert check_pre_filter_off_topic(art) is not None

        ranker = ImportanceRanker(api_key="dummy-key")
        assert ranker.is_eligible(0.95, 0.90, article=art) is False
        eval_res = await ranker.evaluate_global_relevance(art)
        assert eval_res.passed is False

    def test_cyber_threat_actor_is_not_false_positive_blocked(self):
        art = _make_article(
            title="State-sponsored threat actor targets European defense ministry networks",
            summary="Cybersecurity agencies attributed the ransomware intrusion to a foreign intelligence unit.",
            source_name="BleepingComputer",
            category="cybersecurity",
        )
        assert check_blocked_topic(art) is None


class TestBug3RssBoilerplateAndMultiStoryRoundups:
    def test_strip_rss_boilerplate_cleans_guardian_and_kurdish_footers(self):
        raw_en = (
            "<p>European leaders reached an agreement on defense spending after midnight talks.</p>"
            " <a href='https://theguardian.com/x'>Continue reading...</a>"
        )
        entry = SimpleNamespace(
            link="https://www.theguardian.com/world/2026/nov/25/eu-defense",
            title="European leaders reach defense spending accord",
            summary=raw_en,
            published="Wed, 25 Nov 2026 10:00:00 GMT",
        )
        parsed = parse_rss_entry(entry, "The Guardian World", "politics_world")
        assert parsed is not None
        assert "Continue reading" not in parsed.summary
        assert "..." not in parsed.summary
        assert "defense spending" in parsed.summary

        # Translated Sorani output boilerplate also stripped cleanly
        dirty_ku = (
            "سەرکردەکانی ئەورووپا لەسەر بودجەی بەرگری ڕێککەوتن، و، هاوکاری سەربازییان زیاد کرد. "
            "بەردەوام بە لە خوێندنەوە..."
        )
        cleaned_ku = normalize_neutral_terminology(strip_rss_boilerplate(dirty_ku))
        assert "بەردەوام بە لە خوێندنەوە" not in cleaned_ku
        assert ", و," not in cleaned_ku
        assert "، و،" not in cleaned_ku

        # ContentFormatter caption/message also strips trailing Continue reading / بەردەوام بە لە خوێندنەوە
        art = _make_article(
            title="EU defense accord",
            url="https://www.theguardian.com/world/2026/nov/25/eu-defense",
        )
        art.translated_title = "سەرکردەکانی ئەورووپا ڕێککەوتنی بەرگرییان پەسەند کرد"
        art.translated_summary = "سەرکردەکانی ئەورووپا ڕێککەوتنێکی نوێیان واژۆ کرد. بەردەوام بە لە خوێندنەوە..."
        formatter = ContentFormatter()
        caption = formatter.format_caption(art)
        assert "بەردەوام بە لە خوێندنەوە" not in caption

    @pytest.mark.asyncio
    async def test_gemini_translation_strips_boilerplate_and_awkward_punctuation(self):
        provider = GeminiTranslationProvider(api_key="fake-key")
        with patch.object(
            provider._client,
            "generate_json",
            new=AsyncMock(return_value={
                "title": "بەریتانیا و فەرەنسا، و، ئەڵمانیا کۆبوونەوە",
                "summary": (
                    "وەزیرانی دەرەوەی بەریتانیا، و، فەرەنسا لە لەندەن کۆبوونەوە و باسی ئاسایشیان کرد. "
                    "بەردەوام بە لە خوێندنەوە..."
                ),
            }),
        ):
            tr_title, tr_summary = await provider.translate_article(
                title="UK, France and Germany hold defense talks",
                summary="Foreign ministers met in London for defense consultations. Continue reading...",
            )

        assert ", و," not in tr_title and "، و،" not in tr_title
        assert ", و," not in tr_summary and "، و،" not in tr_summary
        assert "بەردەوام بە لە خوێندنەوە" not in tr_summary

    @pytest.mark.parametrize(
        "title,url,summary",
        [
            (
                "Army set to detonate World War II munitions AND Abrego Garcia leaves US",
                "https://www.npr.org/2026/11/25/nx-s1-9999/up-first-briefing",
                "Today on Up First: engineers clear wartime ordnance, and separately a deportation ruling takes effect.",
            ),
            (
                "Morning Briefing: Middle East talks resume; Senate weighs defense budget",
                "https://www.npr.org/podcasts/510318/up-first",
                "Plus, other top headlines in today's news roundup.",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_npr_multi_story_roundup_rejected(self, title: str, url: str, summary: str):
        art = _make_article(title=title, url=url, summary=summary, source_name="NPR News")
        assert is_multi_story_roundup(art) is True
        assert check_blocked_topic(art) is not None
        assert check_pre_filter_off_topic(art) is not None

        ranker = ImportanceRanker(api_key="dummy-key")
        assert ranker.is_eligible(0.95, 0.90, article=art) is False
        eval_res = await ranker.evaluate_global_relevance(art)
        assert eval_res.passed is False

    def test_single_person_immigration_case_blocked_while_diplomatic_headlines_pass(self):
        indiv_case = _make_article(
            title="Kilmar Abrego Garcia ordered held ahead of US immigration court hearing",
            summary="A federal immigration judge scheduled a deportation hearing in Maryland.",
            source_name="NPR News",
        )
        assert check_blocked_topic(indiv_case) is not None

        diplomacy = _make_article(
            title="Israel and Iran exchange warnings ahead of UN Security Council session",
            summary="Ambassadors at the United Nations held emergency consultations on regional escalation.",
            source_name="Reuters World",
        )
        assert is_multi_story_roundup(diplomacy) is False
        assert check_blocked_topic(diplomacy) is None


class TestBugTartousSyriaNotClassifiedAsKurdistan:
    """
    Ensure category classification ("کوردستان") is strictly determined by
    verified news content/location (Southern Kurdistan focus), NEVER falsely
    tagging non-Kurdistan Syrian locations (e.g., Tartous, Latakia, Damascus) as "کوردستان".
    """

    def test_tartous_airstrike_from_north_press_not_classified_as_kurdistan(self):
        title = "Airstrikes hit military depot near Tartous port in western Syria"
        summary = (
            "QAMISHLI, Syria (North Press) - Explosions were reported on Wednesday near the port of Tartous "
            "following suspected airstrikes targeting coastal ammunition warehouses."
        )
        art = _make_article(
            title=title,
            summary=summary,
            source_name="North Press Agency",
            category="kurdistan",
            region="West Kurdistan",
            url="https://npasyria.com/en/tartous-strike-1/",
        )

        assert is_genuine_kurdistan_event(art) is False
        assert is_bashur_article(art) is False
        inferred = normalize_category(art.category, text_hint=f"{title} {summary}")
        assert inferred == "military_conflict"

    def test_tartous_syrian_naval_dockyard_news_normalizes_to_military_or_world(self):
        title = "Russia expands naval facilities at Syrian port of Tartus"
        summary = "Satellite images reveal new docking piers being constructed at Tartous naval base on the Mediterranean."
        art = _make_article(
            title=title,
            summary=summary,
            source_name="North Press Agency",
            category="kurdistan",
            region="West Kurdistan",
            url="https://npasyria.com/en/tartus-dock-2/",
        )

        assert is_genuine_kurdistan_event(art) is False
        assert is_bashur_article(art) is False
        inferred = normalize_category(art.category, text_hint=f"{title} {summary}")
        assert inferred in {"military_conflict", "politics_world"}

    def test_damascus_latakia_strikes_rejected_from_kurdistan_category(self):
        title = "هێرشێکی ئاسمانی کۆگایەکی لە لازقیە و دەوروبەری دیمەشق کردە ئامانج"
        summary = "تەقینەوەی بەهێز لە بەندەری لازقیە و ناوچەکانی دەوروبەری دیمەشق بیسترا بەهۆی بۆردومانەوە."
        art = _make_article(
            title=title,
            summary=summary,
            source_name="Kurdistan24 Sorani",
            category="kurdistan",
            region="South Kurdistan",
            url="https://www.kurdistan24.net/ckb/latakia-strike",
        )

        assert is_genuine_kurdistan_event(art) is False
        assert is_bashur_article(art) is False
        inferred = normalize_category("kurdistan", text_hint=f"{title} {summary}")
        assert inferred == "military_conflict"

    def test_genuine_kurdistan_events_remain_verified_as_kurdistan(self):
        title = "Erbil and Baghdad sign agreement on regional civil service salaries"
        summary = "KRG Prime Minister Masrour Barzani announced the finalization of the budget mechanism in Erbil."
        art = _make_article(
            title=title,
            summary=summary,
            source_name="Rudaw English",
            category="kurdistan",
            region="South Kurdistan",
            url="https://www.rudaw.net/english/kurdistan/salary-agreement",
        )

        assert is_genuine_kurdistan_event(art) is True
        assert is_bashur_article(art) is True
        assert normalize_category("kurdistan", text_hint=f"{title} {summary}") == "kurdistan"

    def test_genuine_rojava_sdf_events_remain_genuine_kurdistan(self):
        title = "SDF repels infiltration attempt near Kobani countryside"
        summary = "Syrian Democratic Forces in Rojava engaged hostile fighters near Kobani and Qamishli."
        art = _make_article(
            title=title,
            summary=summary,
            source_name="North Press Agency",
            category="kurdistan",
            region="West Kurdistan",
            url="https://npasyria.com/en/kobani-sdf-1",
        )

        assert is_genuine_kurdistan_event(art) is True
        assert is_bashur_article(art) is False
        assert normalize_category("kurdistan", text_hint=f"{title} {summary}") == "kurdistan"

    def test_pipeline_and_formatter_clear_tartous_kurdistan_tag(self):
        art = _make_article(
            title="Airstrikes target Tartous military site",
            summary="Explosions shook the coastal city of Tartous in Syria.",
            source_name="North Press Agency",
            category="kurdistan",
            region="West Kurdistan",
            url="https://npasyria.com/en/tartous-site",
        )
        art.translated_title = "بۆردومانی ئاسمانی بنکەیەکی سەربازی لە تەرتووس کردە ئامانج"
        art.translated_summary = "تەقینەوەی بەهێز لە کەناراوەکانی تەرتووس لە سووریا ڕوویدا."

        relevance = RelevanceEvaluation(
            score=8,
            is_breaking=False,
            category="kurdistan",  # LLM erroneously classified as kurdistan
            reason="Coastal strike in Tartous",
            passed=True,
        )

        ArticlePipeline._populate_relevance_fields(art, relevance)

        # Region must be stripped and category coerced to military_conflict
        assert art.region == ""
        assert art.category == "military_conflict"

        formatter = ContentFormatter()
        caption = formatter.format_caption(art)
        assert "📂 کوردستان" not in caption
        assert "شەڕ و سەربازی" in caption
        assert "⚔️" in caption
