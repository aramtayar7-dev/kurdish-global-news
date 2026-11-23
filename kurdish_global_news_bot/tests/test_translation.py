"""Tests for the 3-tier translation engine, fallback providers, and Gemini proofreader."""

import asyncio
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from processing.translator import (
    TranslationEngine,
    GoogleTranslateProvider,
    GroqTranslationProvider,
    DuckDuckGoAIProvider,
    GeminiProofreader,
    TranslationError,
    _split_text,
    GOOGLE_TRANSLATE_ENDPOINT,
    GROQ_API_ENDPOINT,
    DUCKDUCKGO_CHAT_ENDPOINT,
    DUCKDUCKGO_STATUS_ENDPOINT,
    GEMINI_API_BASE_URL,
)


class TestGroqTranslationProvider:
    """Test the Groq LLM translation provider with multi-key rotation."""

    def test_provider_name_and_model(self):
        provider = GroqTranslationProvider(api_key="test_key", model="llama-3.3-70b-versatile")
        assert provider.name == "Groq"
        assert provider.model == "llama-3.3-70b-versatile"
        assert provider.api_keys == ["test_key"]
        assert provider.current_key == "test_key"

    def test_multi_key_parsing(self):
        provider = GroqTranslationProvider(api_key="key1, key2 ,  key3  ")
        assert provider.api_keys == ["key1", "key2", "key3"]
        assert provider.current_key == "key1"

    @pytest.mark.asyncio
    async def test_translate_empty_string(self):
        provider = GroqTranslationProvider(api_key="test_key")
        result = await provider.translate("")
        assert result == ""

    @pytest.mark.asyncio
    async def test_translate_missing_api_key_raises(self):
        provider = GroqTranslationProvider(api_key="")
        with pytest.raises(TranslationError, match="GROQ_API_KEY"):
            await provider.translate("Some text")

    @pytest.mark.asyncio
    async def test_translate_success(self):
        """Verify endpoint, payload with llama-3.3-70b-versatile, headers, and parsing."""
        provider = GroqTranslationProvider(
            api_key="gsk_test123",
            model="llama-3.3-70b-versatile",
            delay=0.0,
        )

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "سەرکردەکانی جیهان بۆ تاوتوێکردنی ئاگربەستی بەپەلە کۆدەبنەوە.",
                    }
                }
            ]
        })

        mock_session = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response

        with patch.object(provider, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            result = await provider.translate("Global leaders gather to discuss emergency ceasefire.")
            assert result == "سەرکردەکانی جیهان بۆ تاوتوێکردنی ئاگربەستی بەپەلە کۆدەبنەوە."

            mock_session.post.assert_called_once()
            call_args, call_kwargs = mock_session.post.call_args
            assert call_args[0] == GROQ_API_ENDPOINT
            assert call_kwargs["headers"]["Authorization"] == "Bearer gsk_test123"
            assert call_kwargs["json"]["model"] == "llama-3.3-70b-versatile"
            messages = call_kwargs["json"]["messages"]
            assert len(messages) == 2
            assert "Central Kurdish" in messages[0]["content"] or "Sorani" in messages[0]["content"]
            assert messages[1]["content"] == "Global leaders gather to discuss emergency ceasefire."

    @pytest.mark.asyncio
    async def test_key_rotation_on_rate_limit_429(self):
        """When key 1 returns 429, key 2 is automatically rotated to and succeeds."""
        provider = GroqTranslationProvider(
            api_key="gsk_key1, gsk_key2",
            delay=0.0,
        )

        mock_resp_429 = AsyncMock()
        mock_resp_429.status = 429
        mock_resp_429.text = AsyncMock(return_value="Rate limit exceeded")

        mock_resp_200 = AsyncMock()
        mock_resp_200.status = 200
        mock_resp_200.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "وەرگێڕان بە سەرکەوتوویی تەواو بوو"}}]
        })

        # Sequence of responses: first 429, then 200
        mock_session = MagicMock()
        mock_session.post.return_value.__aenter__.side_effect = [mock_resp_429, mock_resp_200]

        with patch.object(provider, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            result = await provider.translate("Text to translate")
            assert result == "وەرگێڕان بە سەرکەوتوویی تەواو بوو"
            assert mock_session.post.call_count == 2
            # Second call used key 2
            second_call_headers = mock_session.post.call_args_list[1][1]["headers"]
            assert second_call_headers["Authorization"] == "Bearer gsk_key2"

    @pytest.mark.asyncio
    async def test_all_keys_exhausted_raises_translation_error(self):
        provider = GroqTranslationProvider(
            api_key="key1, key2",
            delay=0.0,
        )

        mock_resp_429 = AsyncMock()
        mock_resp_429.status = 429
        mock_resp_429.text = AsyncMock(return_value="Quota exhausted")

        mock_session = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = mock_resp_429

        with patch.object(provider, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            with pytest.raises(TranslationError, match="exhausted or rate-limited"):
                await provider.translate("Text")


class TestDuckDuckGoAIProvider:
    """Test Tier 2 DuckDuckGo AI Chat translation provider."""

    def test_provider_name_and_model(self):
        provider = DuckDuckGoAIProvider(model="gpt-4o-mini", delay=0.0)
        assert provider.name == "DuckDuckGoAI"
        assert provider.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_translate_empty_string(self):
        provider = DuckDuckGoAIProvider(delay=0.0)
        res = await provider.translate("")
        assert res == ""

    @pytest.mark.asyncio
    async def test_translate_success_via_stream(self):
        provider = DuckDuckGoAIProvider(delay=0.0)

        mock_session = MagicMock()
        # Mock _fetch_vqd
        with patch.object(provider, "_fetch_vqd", new_callable=AsyncMock, return_value="test_vqd_123"), \
             patch.object(provider, "_get_session", new_callable=AsyncMock, return_value=mock_session):

            mock_response = AsyncMock()
            mock_response.status = 200

            async def _fake_content():
                yield b'data: {"message": "\xd8\xb3\xda\xb5\xd8\xa7\xd9\x88 "}\n'
                yield b'data: {"message": "\xd8\xac\xdb\x8c\xd9\x87\xd8\xa7\xd9\x86"}\n'
                yield b'data: [DONE]\n'

            mock_response.content = _fake_content()
            mock_session.post.return_value.__aenter__.return_value = mock_response

            res = await provider.translate("Hello world")
            assert "سڵاو" in res
            assert "جیهان" in res

    @pytest.mark.asyncio
    async def test_translate_raises_on_418_challenge(self):
        """When DDG returns 418 challenge, it raises TranslationError cleanly for Tier 3."""
        provider = DuckDuckGoAIProvider(delay=0.0)
        mock_session = MagicMock()

        with patch.object(provider, "_fetch_vqd", new_callable=AsyncMock, return_value="test_vqd"), \
             patch.object(provider, "_get_session", new_callable=AsyncMock, return_value=mock_session):

            mock_response = AsyncMock()
            mock_response.status = 418
            mock_response.text = AsyncMock(return_value='{"action":"error","status":418}')
            mock_session.post.return_value.__aenter__.return_value = mock_response

            with pytest.raises(TranslationError, match="418"):
                await provider.translate("Breaking news")


class TestGoogleTranslateProvider:
    """Test Tier 3 Google Translate ckb safety net provider."""

    def test_provider_name(self):
        provider = GoogleTranslateProvider()
        assert provider.name == "GoogleTranslate"

    @patch("processing.translator.GoogleTranslateProvider.translate")
    @pytest.mark.asyncio
    async def test_translate_basic(self, mock_translate):
        """TEST 11: Translation produces Sorani Kurdish text."""
        mock_translate.return_value = "هەواڵی گرنگ لە جیهان"
        provider = GoogleTranslateProvider()
        result = await provider.translate("Important world news", target_lang="ckb")
        assert result == "هەواڵی گرنگ لە جیهان"

    @pytest.mark.asyncio
    async def test_translate_empty_string(self):
        provider = GoogleTranslateProvider()
        result = await provider.translate("")
        assert result == ""

    @pytest.mark.asyncio
    async def test_translate_chunk_calls_endpoint(self):
        """Verify endpoint, query params with ckb, and headers."""
        provider = GoogleTranslateProvider(delay=0.0)
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[
            [["سڵاو جیهان", "Hello world", None, None]]
        ])

        mock_session = MagicMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response

        with patch.object(provider, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            res = await provider._translate_chunk("Hello world", source_lang="auto", target_lang="ckb")
            assert res == "سڵاو جیهان"
            mock_session.get.assert_called_once()
            call_args, call_kwargs = mock_session.get.call_args
            assert call_args[0] == GOOGLE_TRANSLATE_ENDPOINT
            params = call_kwargs["params"]
            assert params["client"] == "gtx"
            assert params["sl"] == "auto"
            assert params["tl"] == "ckb"
            assert params["dt"] == "t"
            assert params["q"] == "Hello world"
            assert "User-Agent" in call_kwargs["headers"]


class TestGeminiProofreader:
    """Test Stage 4 Gemini Pre-Publish Proofreader & Editor."""

    def test_gemini_init_and_properties(self):
        proofreader = GeminiProofreader(api_key="gem_key1, gem_key2", model="gemini-2.5-flash", enabled=True)
        assert proofreader.model == "gemini-2.5-flash"
        assert proofreader.api_keys == ["gem_key1", "gem_key2"]
        assert proofreader.is_enabled is True

    @pytest.mark.asyncio
    async def test_proofread_success_polishing(self):
        proofreader = GeminiProofreader(api_key="gem_key1")
        mock_session = MagicMock()

        polished_data = {
            "title": "سەرۆکی ئەمریکا بەیاننامەیەکی بەپەلەی بڵاوکردەوە",
            "summary": "لە کۆبوونەوەکەدا باس لە بارودۆخی ناوچەکە کرا.",
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": json.dumps(polished_data)}
                        ]
                    }
                }
            ]
        })
        mock_session.post.return_value.__aenter__.return_value = mock_response

        with patch.object(proofreader, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            title, summary = await proofreader.proofread(
                original_title="US President issued an urgent statement",
                draft_title="سەرۆک ئەمریکا بەیاننامە بەپەلە بڵاوکردەوە",
                original_summary="In the meeting, the regional situation was discussed.",
                draft_summary="لە کۆبوونەوە بارودۆخ ناوچە باس کرا.",
            )
            assert title == polished_data["title"]
            assert summary == polished_data["summary"]
            mock_session.post.assert_called_once()
            call_args, call_kwargs = mock_session.post.call_args
            assert "gemini-2.5-flash" in call_args[0]
            assert call_kwargs["headers"]["x-goog-api-key"] == "gem_key1"

    @pytest.mark.asyncio
    async def test_proofread_multi_key_rotation_on_429(self):
        proofreader = GeminiProofreader(api_key="gem_key1, gem_key2")
        mock_session = MagicMock()

        mock_resp_429 = AsyncMock()
        mock_resp_429.status = 429
        mock_resp_429.text = AsyncMock(return_value="Resource exhausted: quota exceeded")

        mock_resp_200 = AsyncMock()
        mock_resp_200.status = 200
        mock_resp_200.json = AsyncMock(return_value={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": json.dumps({"title": "سەردێری چاککراو", "summary": "پوختەی چاککراو"})}
                        ]
                    }
                }
            ]
        })

        mock_session.post.return_value.__aenter__.side_effect = [mock_resp_429, mock_resp_200]

        with patch.object(proofreader, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            title, summary = await proofreader.proofread(
                original_title="Original",
                draft_title="سەردێر",
                original_summary="Summary",
                draft_summary="پوختە",
            )
            assert title == "سەردێری چاککراو"
            assert mock_session.post.call_count == 2
            # Key 2 was used on retry
            second_key = mock_session.post.call_args_list[1][1]["headers"]["x-goog-api-key"]
            assert second_key == "gem_key2"

    @pytest.mark.asyncio
    async def test_proofread_failsafe_when_disabled(self):
        proofreader = GeminiProofreader(api_key="gem_key1", enabled=False)
        title, summary = await proofreader.proofread(
            original_title="English title",
            draft_title="سەردێری ڕەشنووس",
            original_summary="English summary",
            draft_summary="پوختەی ڕەشنووس",
        )
        assert title == "سەردێری ڕەشنووس"
        assert summary == "پوختەی ڕەشنووس"

    @pytest.mark.asyncio
    async def test_proofread_failsafe_when_no_api_key(self):
        proofreader = GeminiProofreader(api_key="")
        title, summary = await proofreader.proofread(
            original_title="English title",
            draft_title="سەردێری ڕەشنووس",
        )
        assert title == "سەردێری ڕەشنووس"

    @pytest.mark.asyncio
    async def test_proofread_failsafe_on_invalid_json(self):
        """When Gemini returns non-JSON text, fail-safe returns draft without raising."""
        proofreader = GeminiProofreader(api_key="gem_key1")
        mock_session = MagicMock()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Sorry, I cannot format this as JSON."}]
                    }
                }
            ]
        })
        mock_session.post.return_value.__aenter__.return_value = mock_response

        with patch.object(proofreader, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            title, summary = await proofreader.proofread(
                original_title="Original",
                draft_title="سەردێری ڕەشنووس",
                original_summary="Summary",
                draft_summary="پوختە",
            )
            assert title == "سەردێری ڕەشنووس"
            assert summary == "پوختە"


class TestTranslationEngine:
    """Test the translation engine with 3-tier fallback and Gemini proofreading."""

    def test_setup_providers_with_groq_key(self):
        engine = TranslationEngine(groq_api_key="gsk_test123")
        assert len(engine.providers) == 3
        assert engine.providers[0].name == "Groq"
        assert engine.providers[1].name == "DuckDuckGoAI"
        assert engine.providers[2].name == "GoogleTranslate"

    def test_setup_providers_without_groq_key(self):
        engine = TranslationEngine(groq_api_key="")
        assert len(engine.providers) == 2
        assert engine.providers[0].name == "DuckDuckGoAI"
        assert engine.providers[1].name == "GoogleTranslate"

    @pytest.mark.asyncio
    async def test_fallback_chain_groq_fails_ddg_succeeds(self):
        engine = TranslationEngine(groq_api_key="gsk_test123")
        groq_prov = engine.providers[0]
        ddg_prov = engine.providers[1]

        with patch.object(groq_prov, "translate", new_callable=AsyncMock, side_effect=TranslationError("Groq 429")), \
             patch.object(ddg_prov, "translate", new_callable=AsyncMock, return_value="وەرگێڕانی دەکدەکگۆ"):
            result = await engine._translate_with_fallback("Test text", target_lang="ckb")
            assert result == "وەرگێڕانی دەکدەکگۆ"

    @pytest.mark.asyncio
    async def test_fallback_chain_groq_and_ddg_fail_google_succeeds(self):
        engine = TranslationEngine(groq_api_key="gsk_test123")
        groq_prov = engine.providers[0]
        ddg_prov = engine.providers[1]
        google_prov = engine.providers[2]

        with patch.object(groq_prov, "translate", new_callable=AsyncMock, side_effect=TranslationError("Groq 429")), \
             patch.object(ddg_prov, "translate", new_callable=AsyncMock, side_effect=TranslationError("DDG 418 Challenge")), \
             patch.object(google_prov, "translate", new_callable=AsyncMock, return_value="وەرگێڕانی گووگڵ"):
            result = await engine._translate_with_fallback("Test text", target_lang="ckb")
            assert result == "وەرگێڕانی گووگڵ"

    @pytest.mark.asyncio
    async def test_translate_article_runs_gemini_proofreading(self):
        """Verify translate_article translates draft and passes to Gemini proofreading."""
        engine = TranslationEngine(groq_api_key="", enable_gemini_proofread=True)

        with patch.object(engine, "translate_title", new_callable=AsyncMock, return_value="سەردێری خاو"), \
             patch.object(engine, "translate_summary", new_callable=AsyncMock, return_value="پوختەی خاو"), \
             patch.object(engine._proofreader, "proofread", new_callable=AsyncMock, return_value=("سەردێری پوختەکراو", "پوختەی پوختەکراو")) as mock_proof:

            final_title, final_summary = await engine.translate_article(
                title="English Title",
                summary="English Summary",
            )
            assert final_title == "سەردێری پوختەکراو"
            assert final_summary == "پوختەی پوختەکراو"
            mock_proof.assert_awaited_once_with(
                original_title="English Title",
                draft_title="سەردێری خاو",
                original_summary="English Summary",
                draft_summary="پوختەی خاو",
            )

    @pytest.mark.asyncio
    async def test_validate_translation_valid(self):
        engine = TranslationEngine()
        is_valid = await engine.validate_translation(
            "هەواڵی گرنگ لە جیهان",
            "Important world news",
        )
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_translation_empty(self):
        engine = TranslationEngine()
        is_valid = await engine.validate_translation("", "Some original text")
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_translation_failure_raises(self):
        engine = TranslationEngine(groq_api_key="")
        for p in engine.providers:
            p.translate = AsyncMock(side_effect=TranslationError("Fail"))

        with pytest.raises(TranslationError, match="All translation providers failed"):
            await engine._translate_with_fallback("Text", target_lang="ckb")


class TestTextSplitter:
    """Test the text splitting utility."""

    def test_short_text_no_split(self):
        result = _split_text("Short text.", max_len=4500)
        assert len(result) == 1
        assert result[0] == "Short text."

    def test_long_text_splits(self):
        text = ". ".join([f"Sentence {i}" for i in range(200)])
        result = _split_text(text, max_len=100)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 200

    def test_empty_text(self):
        result = _split_text("", max_len=4500)
        assert result == [""]
