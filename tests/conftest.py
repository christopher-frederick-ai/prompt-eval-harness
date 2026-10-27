"""Keep the test suite hermetic: never read a developer's real .env or shell settings."""

import pytest

import prompt_eval.config as config

# Applies at import time too, because some test modules build a Config at module level.
config._load_dotenv = lambda: None

_CONFIG_VARS = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_CHAT_DEPLOYMENT",
    "AZURE_API_BASE",
    "AZURE_API_KEY",
    "JUDGE_DEPLOYMENT",
    "REASONING_EFFORT",
    "PRICE_IN_PER_M",
    "PRICE_OUT_PER_M",
    "JUDGE_PRICE_IN_PER_M",
    "JUDGE_PRICE_OUT_PER_M",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Each test starts with no config variables set and no .env loading."""
    for name in _CONFIG_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "_load_dotenv", lambda: None)
