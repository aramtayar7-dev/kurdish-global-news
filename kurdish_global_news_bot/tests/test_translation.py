"""Tests for Gemini-first single-pass translation, native Kurdish rewrite, and Google Translate ckb fallback."""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from processing.translator import (
    TranslationEngine,
    GeminiTranslationProvider,
    GoogleTranslateProvider,
    TranslationError,
    is_kurdish_sorani,
    normalize_sorani_chars,
    trim_at_sentence_boundary,
    _split_text,
    GOOGLE_TRANSLATE_ENDPOINT,
    GEMINI_TRANSLATION_SYSTEM_PROMPT,
    GEMINI_KURDISH_REWRITE_SYSTEM_PROMPT,
)


class _MockResponseContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class TestGeminiTranslationProvider:
    """Test Tier 1 Gemini single-pass translation & Kurdish rewriting."""

    def test_provider_properties(self):
        provider = GeminiTranslationProvider(api_key="gem_k1, gem_k2", model="gemini-2.5-flash")
        assert provider.name == "Gemini"
        assert provider.model == "gemini-2.5-flash"
        assert provider.api_keys == ["gem_k1", "gem_k2"]
        assert provider.current_key == "gem_k1"
        assert provider.has_keys is True

    @pytest.mark.asyncio
    async def test_single_pass_translation_success(self):
        provider = GeminiTranslationProvider(api_key="gem_test_123", base_backoff=0.0)
        payload_data = {
            "title": "ڕێککەوتنی نوێی بازرگانی لە جنێڤ واژۆکرا",
            "summary": (
                "نوێنەرانی بیست و پێنج وڵات دوای سێ ڕۆژ لە دانوستان گەیشتنە ڕێککەوتنێکی گشتگیر. "
                "بڕیارەکە کەمکردنەوەی باجی گومرگی بە ڕێژەی لە سەدا ١٥ لەخۆدەگرێت. "
                "ئەم هەنگاوە دەبێتە هۆی بووژانەوەی ئابووریی نێودەوڵەتی."
            ),
        }
        mock_resp = AsyncMock(
            status=200,
            json=AsyncMock(return_value={
                "candidates": [{"content": {"parts": [{"text": json.dumps(payload_data)}]}}]
            }),
        )
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=_MockResponseContext(mock_resp))

        with patch.object(provider._client, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            title, summary = await provider.translate_article(
                title="New Trade Agreement Signed in Geneva",
                summary="Officials from 25 countries agreed on new tariff reductions of 15 percent after three days of talks.",
            )

        assert title == payload_data["title"]
        assert summary == payload_data["summary"]
        mock_session.post.assert_called_once()
        call_args, call_kwargs = mock_session.post.call_args
        assert "gemini-2.5-flash" in call_args[0]
        assert call_kwargs["headers"]["x-goog-api-key"] == "gem_test_123"
        assert call_kwargs["json"]["generationConfig"]["maxOutputTokens"] == 800
        assert call_kwargs["json"]["systemInstruction"]["parts"][0]["text"] == GEMINI_TRANSLATION_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_native_kurdish_article_uses_rewrite_prompt(self):
        """Articles already in Sorani Kurdish skip English-to-Kurdish translation and use editorial rewrite."""
        provider = GeminiTranslationProvider(api_key="gem_test_123", base_backoff=0.0)
        mock_resp = AsyncMock(
            status=200,
            json=AsyncMock(return_value={
                "candidates": [{"content": {"parts": [{"text": json.dumps({
                    "title": "هەولێر و بەغدا لەسەر هەناردەکردنی نەوت کۆدەبنەوە",
                    "summary": "شاندێکی باڵای هەرێمی کوردستان گەیشتە بەغدا بۆ گفتوگۆکردن لەسەر دۆسیەی وزە. کۆبوونەوەکە بە ئامادەبوونی وەزیرانی پەیوەندیدار بەڕێوەدەچێت. بڕیارە ڕێککەوتنی کۆتایی لە چەند ڕۆژی داهاتوودا واژۆ بکرێت.",
                })}]}}]
            }),
        )
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=_MockResponseContext(mock_resp))

        with patch.object(provider._client, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            title, summary = await provider.translate_article(
                title="شاندی هەرێمی کوردستان بۆ پرسی نەوت سەردانی بەغدا دەکات",
                summary="شاندێکی حکومەتی هەرێم بە مەبەستی چارەسەرکردنی کێشەکانی هەناردەی نەوت گەیشتە بەغدا.",
            )

        assert "هەولێر" in title
        sent_system_prompt = mock_session.post.call_args[1]["json"]["systemInstruction"]["parts"][0]["text"]
        assert sent_system_prompt == GEMINI_KURDISH_REWRITE_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_key_rotation_and_exponential_backoff_on_429(self):
        provider = GeminiTranslationProvider(api_key="gem_k1, gem_k2", base_backoff=0.05)

        resp_429 = AsyncMock(status=429, text=AsyncMock(return_value="Quota exceeded"))
        resp_200 = AsyncMock(
            status=200,
            json=AsyncMock(return_value={
                "candidates": [{"content": {"parts": [{"text": '{"title": "سەردێر", "summary": "پوختەی هەواڵ."}'}]}}]
            }),
        )
        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=[
            _MockResponseContext(resp_429),
            _MockResponseContext(resp_429),
            _MockResponseContext(resp_200),
        ])

        with patch.object(provider._client, "_get_session", new_callable=AsyncMock, return_value=mock_session), \
             patch("processing.gemini_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            title, summary = await provider.translate_article("English Title", "English Body text.")

        assert title == "سەردێر"
        assert summary == "پوختەی هەواڵ."
        assert mock_session.post.call_count == 3
        # Key rotation happened before exponential backoff on 2nd 429
        assert mock_session.post.call_args_list[0][1]["headers"]["x-goog-api-key"] == "gem_k1"
        assert mock_session.post.call_args_list[1][1]["headers"]["x-goog-api-key"] == "gem_k2"
        mock_sleep.assert_awaited()

    @pytest.mark.asyncio
    async def test_missing_key_raises_translation_error(self):
        provider = GeminiTranslationProvider(api_key="")
        with pytest.raises(TranslationError, match="GEMINI_API_KEY"):
            await provider.translate_article("Title", "Body")


class TestGoogleTranslateFallback:
    """Test lightweight Google Translate ckb fallback provider."""

    def test_provider_name(self):
        provider = GoogleTranslateProvider()
        assert provider.name == "GoogleTranslate"

    @pytest.mark.asyncio
    async def test_translate_empty_string(self):
        provider = GoogleTranslateProvider()
        assert await provider.translate("") == ""

    @pytest.mark.asyncio
    async def test_translate_chunk_calls_ckb_endpoint(self):
        provider = GoogleTranslateProvider(delay=0.0)
        mock_response = AsyncMock(
            status=200,
            json=AsyncMock(return_value=[[["سڵاو جیهان", "Hello world", None, None]]]),
        )
        mock_session = MagicMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response

        with patch.object(provider, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            res = await provider._translate_chunk("Hello world", source_lang="auto", target_lang="ckb")

        assert res == "سڵاو جیهان"
        call_args, call_kwargs = mock_session.get.call_args
        assert call_args[0] == GOOGLE_TRANSLATE_ENDPOINT
        assert call_kwargs["params"]["tl"] == "ckb"


class TestTranslationEngineNoGroq:
    """Test Gemini-first TranslationEngine without Groq."""

    def test_providers_no_groq(self):
        engine_with_gem = TranslationEngine(gemini_api_key="gem_abc")
        names = [p.name for p in engine_with_gem.providers]
        assert names == ["Gemini", "GoogleTranslate"]
        assert "Groq" not in names

        engine_no_key = TranslationEngine(gemini_api_key="")
        assert [p.name for p in engine_no_key.providers] == ["GoogleTranslate"]

    @pytest.mark.asyncio
    async def test_gemini_single_pass_avoids_redundant_calls(self):
        engine = TranslationEngine(gemini_api_key="gem_abc")
        with patch.object(
            engine._gemini_provider,
            "translate_article",
            new_callable=AsyncMock,
            return_value=("سەردێری کوردی", "ڕستەی یەکەم. ڕستەی دووەم. ڕستەی سێیەم."),
        ) as mock_gem, patch.object(
            engine._google_fallback,
            "translate",
            new_callable=AsyncMock,
        ) as mock_google:
            title, summary = await engine.translate_article("Eng Title", "Eng Summary")

        assert title == "سەردێری کوردی"
        assert "ڕستەی سێیەم." in summary
        mock_gem.assert_awaited_once()
        mock_google.assert_not_called()

    @pytest.mark.asyncio
    async def test_google_fallback_used_only_when_all_gemini_keys_fail(self):
        engine = TranslationEngine(gemini_api_key="gem_abc", delay=0.0)
        with patch.object(
            engine._gemini_provider,
            "translate_article",
            new_callable=AsyncMock,
            side_effect=TranslationError("All Gemini keys exhausted"),
        ), patch.object(
            engine._google_fallback,
            "translate",
            new_callable=AsyncMock,
            side_effect=["سەردێری گووگڵ", "پوختەی گووگڵ بە تەواوی."],
        ) as mock_google:
            title, summary = await engine.translate_article("Eng Title", "Eng Summary")

        assert title == "سەردێری گووگڵ"
        assert summary == "پوختەی گووگڵ بە تەواوی."
        assert mock_google.await_count == 2

    @pytest.mark.asyncio
    async def test_native_kurdish_fallback_skips_google_translate(self):
        engine = TranslationEngine(gemini_api_key="", delay=0.0)
        with patch.object(engine._google_fallback, "translate", new_callable=AsyncMock) as mock_google:
            title, summary = await engine.translate_article(
                title="ئاسایشی هەولێر بەیاننامەیەکی بڵاوکردەوە",
                summary="بەرپرسانی ئەمنی هەولێر زانیاری نوێیان لەسەر پاراستنی ئارامیی شارەکە خستە ڕوو.",
            )
        assert "ئاسایشی هەولێر" in title
        assert "پاراستنی ئارامیی" in summary
        mock_google.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_translation(self):
        engine = TranslationEngine(gemini_api_key="")
        assert await engine.validate_translation("هەواڵی گرنگ لە کوردستان", "Important news") is True
        assert await engine.validate_translation("", "Important news") is False


class TestKurdishUtilitiesAndSplitter:
    """Test Sorani detection, normalization, and clean sentence-boundary trimming."""

    def test_is_kurdish_sorani(self):
        assert is_kurdish_sorani("هەولێر پایتەختی هەرێمی کوردستانە") is True
        assert is_kurdish_sorani("US Secretary of State meets Iraqi officials in Baghdad") is False

    def test_normalize_sorani_chars(self):
        assert normalize_sorani_chars("كوردستاني باشوور") == "کوردستانی باشوور"

    def test_trim_at_sentence_boundary_no_ellipsis(self):
        text = "ڕستەی یەکەم تەواو بوو. ڕستەی دووەمیش لێرەدایە. ڕستەی سێیەم درێژە و لە سنووری دیاریکراو تێپەڕی دەکات بۆ تاقیکردنەوە."
        trimmed = trim_at_sentence_boundary(text, max_len=65)
        assert "..." not in trimmed
        assert trimmed.endswith(".")
        assert len(trimmed) <= 65

    def test_text_splitter(self):
        text = ". ".join([f"Sentence {i}" for i in range(200)])
        chunks = _split_text(text, max_len=100)
        assert len(chunks) > 1
        assert _split_text("", max_len=4500) == [""]
