"""
Tests for OpenRouter Kurdish-Optimized Free Fallback Engine.

Verifies:
  1. Config loading & model priority hierarchy.
  2. OpenRouterClient standalone unit tests (multi-key, failover across Kurdish models,
     response_format 400 retry, <think> block stripping, markdown fence parsing).
  3. GeminiClient transparent failover on 429 / quota exhaustion.
  4. TranslationEngine & ImportanceRanker seamless failover with zero extra Gemini API calls.
  5. Watchdog suppression when OpenRouter rescues cycles & secret redaction for OpenRouter/Groq keys.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import config
from core.importance_ranker import ImportanceRanker
from monitoring.watchdog import (
    ALERT_GEMINI_EXHAUSTED,
    AdminWatchdog,
    WatchdogCycleMetrics,
    redact_secrets,
)
from processing.gemini_client import GeminiClient, GeminiQuotaExhaustedError
from processing.openrouter_client import (
    OpenRouterClient,
    OpenRouterError,
    _parse_api_keys,
)
from processing.translator import TranslationEngine
from sources.base import RawArticle
from storage.models import Article


class _MockResponseContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


# ═══════════════════════════════════════════════════════════════════
# 1. Config & Key Parsing Tests
# ═══════════════════════════════════════════════════════════════════

def test_openrouter_config_defaults():
    """Verify default Kurdish-capable free models list and priority order."""
    assert "google/gemma-3-27b-it:free" in config.OPENROUTER_KURDISH_FREE_MODELS
    assert "meta-llama/llama-3.3-70b-instruct:free" in config.OPENROUTER_KURDISH_FREE_MODELS
    assert "deepseek/deepseek-chat-v3-0324:free" in config.OPENROUTER_KURDISH_FREE_MODELS
    # Gemma 3 is first priority for Kurdish Sorani quality
    assert config.OPENROUTER_KURDISH_FREE_MODELS[0] == "google/gemma-3-27b-it:free"
    assert config.OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1/chat/completions"


def test_openrouter_parse_api_keys():
    """Verify comma-separated key normalization and deduplication."""
    keys = _parse_api_keys("sk-or-key1, sk-or-key2,sk-or-key1, ,sk-or-key3")
    assert keys == ["sk-or-key1", "sk-or-key2", "sk-or-key3"]


# ═══════════════════════════════════════════════════════════════════
# 2. OpenRouterClient Standalone Unit Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_openrouter_client_success_primary_model():
    """Primary model succeeds with valid JSON."""
    client = OpenRouterClient(
        api_key="sk-or-testkey-12345",
        models=["google/gemma-3-27b-it:free"],
    )

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(
        return_value={
            "choices": [
                {
                    "message": {
                        "content": '{"title": "سەردێڕی هەواڵ", "summary": "کورتەی هەواڵ بە سۆرانی."}'
                    }
                }
            ]
        }
    )

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post = MagicMock(return_value=_MockResponseContext(mock_resp))
    client._session = mock_session

    result = await client.generate_json("System prompt", "User article")
    assert result["title"] == "سەردێڕی هەواڵ"
    assert "کورتەی هەواڵ" in result["summary"]


@pytest.mark.asyncio
async def test_openrouter_model_failover_chain():
    """Fails over from Gemma 3 (HTTP 429) to Llama 3.3 (HTTP 200)."""
    client = OpenRouterClient(
        api_key="sk-or-testkey-12345",
        models=["google/gemma-3-27b-it:free", "meta-llama/llama-3.3-70b-instruct:free"],
    )

    resp_429 = AsyncMock()
    resp_429.status = 429
    resp_429.text = AsyncMock(return_value="Rate limited on free tier")

    resp_200 = AsyncMock()
    resp_200.status = 200
    resp_200.json = AsyncMock(
        return_value={
            "choices": [
                {
                    "message": {
                        "content": '{"title": "هەواڵی دووەم", "summary": "سەرکەوت لە مۆدێلی دووەم."}'
                    }
                }
            ]
        }
    )

    call_count = 0

    def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _MockResponseContext(resp_429)
        return _MockResponseContext(resp_200)

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post = MagicMock(side_effect=mock_post)
    client._session = mock_session

    result = await client.generate_json("System prompt", "User article")
    assert result["title"] == "هەواڵی دووەم"
    assert call_count == 2


@pytest.mark.asyncio
async def test_openrouter_retry_without_response_format_on_400():
    """If free model rejects response_format with HTTP 400, client retries without it."""
    client = OpenRouterClient(
        api_key="sk-or-testkey-12345",
        models=["deepseek/deepseek-chat-v3-0324:free"],
    )

    resp_400 = AsyncMock()
    resp_400.status = 400
    resp_400.text = AsyncMock(return_value="response_format json_object is not supported")

    resp_200 = AsyncMock()
    resp_200.status = 200
    resp_200.json = AsyncMock(
        return_value={
            "choices": [
                {
                    "message": {
                        "content": '{"title": "سەرکەوت بەبێ فۆرمات", "summary": "ناوەڕۆک"}'
                    }
                }
            ]
        }
    )

    payloads = []

    def mock_post(url, headers=None, json=None, timeout=None):
        payloads.append(dict(json))
        if len(payloads) == 1:
            return _MockResponseContext(resp_400)
        return _MockResponseContext(resp_200)

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post = MagicMock(side_effect=mock_post)
    client._session = mock_session

    result = await client.generate_json("System prompt", "User article")
    assert result["title"] == "سەرکەوت بەبێ فۆرمات"
    assert len(payloads) == 2
    # First attempt had response_format
    assert "response_format" in payloads[0]
    # Second attempt omitted response_format
    assert "response_format" not in payloads[1]


@pytest.mark.asyncio
async def test_openrouter_strips_think_blocks_and_markdown():
    """Extracts valid JSON even when surrounded by reasoning <think> tags and code fences."""
    client = OpenRouterClient(api_key="sk-or-testkey-12345")

    raw_output = """
<think>
The user wants a Central Kurdish translation of this international report.
Let's check the terms and ensure BBC Sorani style.
</think>
```json
{
  "title": "ڕێککەوتنی نوێ لە نێوان لایەنەکان واژۆ کرا",
  "summary": "لە کۆبوونەوەکەدا بڕیار لەسەر دەستپێکردنەوەی دانوستانەکان درا."
}
```
"""
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(
        return_value={"choices": [{"message": {"content": raw_output}}]}
    )

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post = MagicMock(return_value=_MockResponseContext(mock_resp))
    client._session = mock_session

    result = await client.generate_json("System prompt", "User article")
    assert result["title"] == "ڕێککەوتنی نوێ لە نێوان لایەنەکان واژۆ کرا"
    assert "<think>" not in str(result)


# ═══════════════════════════════════════════════════════════════════
# 3. GeminiClient Transparent Failover Integration Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_gemini_client_normal_success_does_not_call_openrouter():
    """When Gemini succeeds, OpenRouter is untouched."""
    gemini = GeminiClient(api_key="gemini-key-1", base_backoff=0.0)

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(
        return_value={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": '{"title": "Gemini Title", "summary": "Gemini Summary"}'}
                        ]
                    }
                }
            ]
        }
    )

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post = MagicMock(return_value=_MockResponseContext(mock_resp))
    gemini._session = mock_session

    with patch.object(OpenRouterClient, "generate_json", new=AsyncMock()) as mock_or:
        res = await gemini.generate_json("Sys", "User")
        assert res["title"] == "Gemini Title"
        mock_or.assert_not_called()
        stats = gemini.consume_cycle_stats()
        assert stats["openrouter_rescued"] == 0


@pytest.mark.asyncio
async def test_gemini_client_fails_over_to_openrouter_when_all_keys_in_cooldown(monkeypatch):
    """When all Gemini keys are in 429 cooldown, GeminiClient transparently delegates to OpenRouter."""
    monkeypatch.setattr(config, "ENABLE_OPENROUTER_FALLBACK", True)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-or-mock-key")

    gemini = GeminiClient(api_key="gemini-key-1", base_backoff=0.0)
    # Put key into cooldown
    gemini._cooldowns["gemini-key-1"] = 9999999999.0

    mock_or_result = {
        "title": "سەردێڕی ڕزگارکراو بە OpenRouter",
        "summary": "ناوەڕۆکی هەواڵ بەبێ دابڕان.",
    }

    with patch.object(OpenRouterClient, "generate_json", new=AsyncMock(return_value=mock_or_result)) as mock_or:
        result = await gemini.generate_json("Sys", "User")
        assert result["title"] == "سەردێڕی ڕزگارکراو بە OpenRouter"
        mock_or.assert_called_once()
        stats = gemini.consume_cycle_stats()
        assert stats["openrouter_rescued"] == 1


@pytest.mark.asyncio
async def test_gemini_client_fails_over_to_openrouter_on_gemini_http_429_exhaustion(monkeypatch):
    """When Gemini API calls hit HTTP 429, GeminiClient catches it and falls over to OpenRouter."""
    monkeypatch.setattr(config, "ENABLE_OPENROUTER_FALLBACK", True)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-or-mock-key")

    gemini = GeminiClient(api_key="gemini-key-1", base_backoff=0.0, max_rate_limit_retries=0)

    resp_429 = AsyncMock()
    resp_429.status = 429
    resp_429.text = AsyncMock(return_value="Resource exhausted: quota exceeded")

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post = MagicMock(return_value=_MockResponseContext(resp_429))
    gemini._session = mock_session

    mock_or_result = {
        "title": "ڕزگارکراو لە پاش 429",
        "summary": "سەرکەوتووانە ئەنجامدرا.",
    }

    with patch.object(OpenRouterClient, "generate_json", new=AsyncMock(return_value=mock_or_result)) as mock_or:
        result = await gemini.generate_json("Sys", "User")
        assert result["title"] == "ڕزگارکراو لە پاش 429"
        mock_or.assert_called_once()
        stats = gemini.consume_cycle_stats()
        assert stats["openrouter_rescued"] == 1


# ═══════════════════════════════════════════════════════════════════
# 4. TranslationEngine & ImportanceRanker End-to-End Fallback Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_translation_engine_uses_openrouter_when_gemini_cooling(monkeypatch):
    """TranslationEngine automatically gets OpenRouter fallback with proper Sorani schema."""
    monkeypatch.setattr(config, "ENABLE_OPENROUTER_FALLBACK", True)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-or-mock-key")

    engine = TranslationEngine()
    # Mark gemini keys in cooldown
    for k in engine._gemini_provider._client.api_keys:
        engine._gemini_provider._client._cooldowns[k] = 9999999999.0

    mock_or_article = {
        "title": "سەردانێکی گرنگ بۆ ڕۆژهەڵاتی ناوەڕاست",
        "summary": "وەزیری دەرەوەی ئەمریکا سەردانی ناوچەکەی کرد. لە میانەی گەشتەکەیدا لەگەڵ بەرپرسانی باڵا کۆبووەوە.",
        "event_key": "mideast_visit",
        "event_entities": ["mideast", "diplomacy"],
    }

    with patch.object(OpenRouterClient, "generate_json", new=AsyncMock(return_value=mock_or_article)) as mock_or:
        title, summary = await engine.translate_article(
            title="Important Visit to Middle East",
            summary="US official visited the region for high-level meetings.",
        )
        assert title == "سەردانێکی گرنگ بۆ ڕۆژهەڵاتی ناوەڕاست"
        assert "وەزیری دەرەوەی ئەمریکا" in summary
        mock_or.assert_called_once()


@pytest.mark.asyncio
async def test_importance_ranker_uses_openrouter_when_gemini_cooling(monkeypatch):
    """ImportanceRanker automatically gets OpenRouter fallback for classification and scoring."""
    monkeypatch.setattr(config, "ENABLE_OPENROUTER_FALLBACK", True)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-or-mock-key")

    ranker = ImportanceRanker()
    # Put gemini keys in cooldown
    for k in ranker._gemini.api_keys:
        ranker._gemini._cooldowns[k] = 9999999999.0

    mock_ranking = {
        "items": [
            {
                "idx": 0,
                "score": 9,
                "category": "military_conflict",
                "is_breaking": False,
                "reason": "Major regional military event",
            }
        ]
    }

    with patch.object(OpenRouterClient, "generate_json", new=AsyncMock(return_value=mock_ranking)) as mock_or:
        art = Article(
            original_title="Major Missile Base Targeted in Strike",
            original_summary="A key base was hit overnight causing significant damage.",
            original_url="https://apnews.com/article/2",
            source_name="AP News",
        )
        evals = await ranker.evaluate_batch_relevance([art])
        assert len(evals) == 1
        assert evals[0].score == 9
        assert evals[0].category == "military_conflict"
        mock_or.assert_called_once()


# ═══════════════════════════════════════════════════════════════════
# 5. Watchdog Suppression & Secret Redaction Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_watchdog_suppresses_gemini_exhausted_alert_when_rescued_by_openrouter():
    """
    If gemini_exhausted > 0 BUT openrouter_rescued > 0, watchdog does NOT send
    GEMINI_EXHAUSTED alarm spam to the admin chat.
    """
    mock_publisher = MagicMock()
    mock_publisher.send_admin_message = AsyncMock(return_value=True)

    watchdog = AdminWatchdog(
        db=None,
        publisher=mock_publisher,
        admin_chat_id="12345678",
        public_channel_id="@kurdishglobalnews",
    )

    metrics = WatchdogCycleMetrics(
        gemini_429_hits=5,
        gemini_exhausted=1,
        openrouter_rescued=1,  # Rescued!
        discovery_ran=True,
    )

    result = await watchdog.run_cycle_checks(metrics)
    assert ALERT_GEMINI_EXHAUSTED not in result["alerts_sent"]
    mock_publisher.send_admin_message.assert_not_called()


@pytest.mark.asyncio
async def test_watchdog_sends_gemini_exhausted_alert_when_openrouter_not_rescued():
    """
    If gemini_exhausted > 0 and openrouter_rescued == 0, watchdog sends
    GEMINI_EXHAUSTED alert to the admin chat.
    """
    mock_publisher = MagicMock()
    mock_publisher.send_admin_message = AsyncMock(return_value=True)

    watchdog = AdminWatchdog(
        db=None,
        publisher=mock_publisher,
        admin_chat_id="12345678",
        public_channel_id="@kurdishglobalnews",
    )

    metrics = WatchdogCycleMetrics(
        gemini_429_hits=5,
        gemini_exhausted=1,
        openrouter_rescued=0,  # NOT rescued!
        discovery_ran=True,
    )

    result = await watchdog.run_cycle_checks(metrics)
    assert ALERT_GEMINI_EXHAUSTED in result["alerts_sent"]
    assert mock_publisher.send_admin_message.called


def test_watchdog_redacts_openrouter_and_groq_secrets(monkeypatch):
    """Confirm OpenRouter and Groq API keys are completely scrubbed from error logs and alerts."""
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-or-v1-abcdef1234567890abcdef1234567890")
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk_1234567890abcdefghijklmnopqrstuvwxyz")

    raw_text = (
        "HTTP 401 error with bearer token: Bearer sk-or-v1-abcdef1234567890abcdef1234567890 "
        "and Groq token gsk_1234567890abcdefghijklmnopqrstuvwxyz failed."
    )
    scrubbed = redact_secrets(raw_text)
    assert "sk-or-v1-abcdef" not in scrubbed
    assert "gsk_1234567890" not in scrubbed
    assert "[REDACTED]" in scrubbed


@pytest.mark.asyncio
async def test_openrouter_enforces_min_tokens_retries_length_truncation_and_recovers_json():
    """Ensures OpenRouter enforces max_tokens >= 1024, retries on finish_reason=length, and cleans wrapped/truncated JSON."""
    client = OpenRouterClient(
        api_key="sk-or-testkey-12345",
        models=["openrouter/free"],
    )

    payloads: list[dict] = []

    resp_clipped = AsyncMock()
    resp_clipped.status = 200
    resp_clipped.json = AsyncMock(
        return_value={
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": '{"title":"بەرگریی ئاسمانی چەند مووشەکێکی خستە خوارەوە","summary":"وەزارەتە'
                    },
                }
            ]
        }
    )

    resp_complete = AsyncMock()
    resp_complete.status = 200
    resp_complete.json = AsyncMock(
        return_value={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            'Here is the output:\n```json\n'
                            '{"title": "بەرگریی ئاسمانی چەند مووشەکێکی خستە خوارەوە", '
                            '"summary": "وەزارەتی بەرگری ڕایگەیاند هێزەکان بەرپەرچی هێرشەکانیان داوەتەوە."}\n```'
                        )
                    },
                }
            ]
        }
    )

    def fake_post(url, headers=None, json=None, timeout=None):
        payloads.append(dict(json))
        if len(payloads) == 1:
            return _MockResponseContext(resp_clipped)
        return _MockResponseContext(resp_complete)

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post = MagicMock(side_effect=fake_post)
    client._session = mock_session

    res = await client.generate_json("Sys", "User", max_tokens=800)
    assert len(payloads) == 2
    assert payloads[0]["max_tokens"] >= 1024
    assert payloads[1]["max_tokens"] >= 2048
    assert res["title"] == "بەرگریی ئاسمانی چەند مووشەکێکی خستە خوارەوە"
    assert "وەزارەتی بەرگری" in res["summary"]

    # Also verify safe JSON recovery if a truncated Sorani payload hits hard model limits after retry
    recovered = OpenRouterClient._parse_json_safely(
        '{"title":"کۆبوونەوەی ئەمنی لە هەولێر","summary":"بەرپرسانی سەربازی کۆبوونەوەیەکی بەپەلەیان ئەنجام دا. دواتر هێزەکان'
    )
    assert recovered["title"] == "کۆبوونەوەی ئەمنی لە هەولێر"
    assert "بەرپرسانی سەربازی" in recovered["summary"]
