from __future__ import annotations

import ast
import json
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from api.webhook import handle_webhook_request


def _make_mock_bot() -> MagicMock:
    bot = MagicMock()
    bot.defaults = None
    bot.send_photo = AsyncMock(return_value=SimpleNamespace(message_id=777))
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=888))
    return bot


def test_vercel_json_routes_api_webhook():
    vercel_cfg = json.loads(
        (Path(__file__).resolve().parent.parent / "vercel.json").read_text(encoding="utf-8")
    )
    assert vercel_cfg["version"] == 2
    assert any(
        r.get("source") == "/api/webhook" and r.get("destination") == "/api/webhook.py"
        for r in vercel_cfg.get("rewrites", [])
    )


def test_vercel_webhook_rejects_bad_secret_and_invalid_json():
    bot = _make_mock_bot()

    status_403, body_403 = handle_webhook_request(
        b"{}",
        headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"},
        webhook_secret="expected",
        bot=bot,
    )
    assert status_403 == 403
    assert body_403["error"] == "forbidden"

    status_400, body_400 = handle_webhook_request(
        b"not-json",
        headers={"X-Telegram-Bot-Api-Secret-Token": "expected"},
        webhook_secret="expected",
        bot=bot,
    )
    assert status_400 == 400
    assert body_400["error"] == "invalid_json"


def test_vercel_webhook_publishes_admin_post_and_blocks_non_admin(tmp_path: Path):
    bot = _make_mock_bot()
    fake_card = tmp_path / "vercel_card.jpg"
    fake_card.write_bytes(b"JPEGDATA")

    card_gen = MagicMock()
    card_gen.generate_card_file = AsyncMock(return_value=fake_card)

    non_admin_payload = json.dumps(
        {
            "update_id": 100,
            "message": {
                "message_id": 1,
                "date": 1700000000,
                "chat": {"id": 99999, "type": "private"},
                "from": {"id": 99999, "is_bot": False, "first_name": "Other"},
                "text": "پۆستی نایاسایی",
            },
        }
    ).encode("utf-8")

    st_ignored, res_ignored = handle_webhook_request(
        non_admin_payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "sec"},
        webhook_secret="sec",
        bot=bot,
        card_generator=card_gen,
        channel_id="@KurdishGlobalNews",
        admin_chat_id=111222333,
    )
    assert st_ignored == 200
    assert res_ignored["status"] == "ignored"
    bot.send_photo.assert_not_called()

    admin_payload = json.dumps(
        {
            "update_id": 101,
            "message": {
                "message_id": 2,
                "date": 1700000001,
                "chat": {"id": 111222333, "type": "private"},
                "from": {"id": 111222333, "is_bot": False, "first_name": "Admin"},
                "text": "بەپەلە | کۆبوونەوەی گرنگ لە هەولێر",
            },
        }
    ).encode("utf-8")

    st_ok, res_ok = handle_webhook_request(
        admin_payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "sec"},
        webhook_secret="sec",
        bot=bot,
        card_generator=card_gen,
        channel_id="@KurdishGlobalNews",
        admin_chat_id=111222333,
    )
    assert st_ok == 200
    assert res_ok["status"] == "published"
    assert res_ok["is_breaking"] is True
    bot.send_photo.assert_awaited_once()
    assert "✅ بڵاوکرایەوە" in bot.send_message.await_args.kwargs["text"]


def test_pyproject_covers_full_webhook_import_graph_without_trailing_commas():
    root = Path(__file__).resolve().parent.parent
    raw_toml = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert ",\n]" not in raw_toml and ",]" not in raw_toml

    cfg = tomllib.loads(raw_toml)
    assert cfg["tool"]["vercel"]["entrypoint"] == "api.webhook:handler"
    declared = {
        dep.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip().lower()
        for dep in cfg["project"]["dependencies"]
    }

    module_to_dist = {
        "PIL": "pillow",
        "aiohttp": "aiohttp",
        "aiosqlite": "aiosqlite",
        "arabic_reshaper": "arabic-reshaper",
        "bidi": "python-bidi",
        "bs4": "beautifulsoup4",
        "dateutil": "python-dateutil",
        "dotenv": "python-dotenv",
        "httpx": "httpx",
        "rapidfuzz": "rapidfuzz",
        "telegram": "python-telegram-bot",
    }

    stdlib = set(sys.stdlib_module_names)
    visited: set[Path] = set()
    queue: list[Path] = [root / "api" / "webhook.py"]
    external: set[str] = set()

    def _resolve(mod: str) -> Path | None:
        parts = mod.split(".")
        p_file = root.joinpath(*parts).with_suffix(".py")
        if p_file.exists():
            return p_file
        p_pkg = root.joinpath(*parts) / "__init__.py"
        return p_pkg if p_pkg.exists() else None

    while queue:
        current = queue.pop(0).resolve()
        if current in visited or not current.exists():
            continue
        visited.add(current)
        if current.name != "__init__.py":
            init_file = current.parent / "__init__.py"
            if init_file.exists() and init_file != root / "__init__.py":
                queue.append(init_file)
        tree = ast.parse(current.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
            for mod in mods:
                top = mod.split(".")[0]
                target = _resolve(mod) or _resolve(top)
                if target:
                    queue.append(target)
                elif top not in stdlib:
                    external.add(top)

    for top_mod in sorted(external):
        dist = module_to_dist.get(top_mod, top_mod.lower())
        assert dist in declared, f"Missing dependency for imported module {top_mod} ({dist})"

