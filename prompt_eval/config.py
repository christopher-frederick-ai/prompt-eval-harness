"""Configuration loaded from environment variables (optionally a local .env file).

Secrets are never passed around in code: the Azure SDK reads AZURE_OPENAI_API_KEY from
the environment itself, so this module only checks that it is present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


# Fallback names, so an existing .env from other projects works unchanged.
# (AZURE_API_VERSION is deliberately NOT aliased: older values such as 2023-05-15 predate
# gpt-5 models and would fail. Set AZURE_OPENAI_API_VERSION explicitly if you need to change it.)
_ALIASES = {
    "AZURE_OPENAI_ENDPOINT": ("AZURE_API_BASE",),
    "AZURE_OPENAI_API_KEY": ("AZURE_API_KEY",),
    "AZURE_CHAT_DEPLOYMENT": (),
}


def _apply_aliases() -> None:
    for canonical, alts in _ALIASES.items():
        if not os.getenv(canonical):
            for alt in alts:
                if os.getenv(alt):
                    os.environ[canonical] = os.environ[alt]
                    break


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # python-dotenv is optional
        pass


@dataclass(frozen=True)
class Config:
    endpoint: str
    api_version: str
    chat_deployment: str
    judge_deployment: str
    reasoning_effort: str | None
    # USD per 1M tokens. These are ASSUMED list prices: verify against your Azure pricing
    # page and override via PRICE_IN_PER_M / PRICE_OUT_PER_M. Reasoning tokens bill as output.
    price_in_per_m: float
    price_out_per_m: float
    # Prices for the JUDGE deployment (may differ from the generator's). If unset they fall back
    # to the generator prices and judge_prices_set is False, which the report flags.
    judge_price_in_per_m: float = 0.0
    judge_price_out_per_m: float = 0.0
    judge_prices_set: bool = False

    @classmethod
    def from_env(cls, dry_run: bool = False) -> "Config":
        _load_dotenv()
        _apply_aliases()
        if dry_run:
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://dry-run.invalid/")
            deployment = os.getenv("AZURE_CHAT_DEPLOYMENT", "mock-model")
        else:
            missing = [
                name
                for name in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_CHAT_DEPLOYMENT")
                if not os.getenv(name)
            ]
            if missing:
                raise SystemExit(
                    "Missing environment variables: "
                    + ", ".join(missing)
                    + "\nCopy .env.example to .env and fill them in (or use --dry-run)."
                )
            endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
            deployment = os.environ["AZURE_CHAT_DEPLOYMENT"]
        price_in = float(os.getenv("PRICE_IN_PER_M", "0.25"))
        price_out = float(os.getenv("PRICE_OUT_PER_M", "2.00"))
        jin, jout = os.getenv("JUDGE_PRICE_IN_PER_M"), os.getenv("JUDGE_PRICE_OUT_PER_M")
        return cls(
            endpoint=endpoint,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            chat_deployment=deployment,
            judge_deployment=os.getenv("JUDGE_DEPLOYMENT", deployment),
            reasoning_effort=os.getenv("REASONING_EFFORT") or None,
            price_in_per_m=price_in,
            price_out_per_m=price_out,
            judge_price_in_per_m=float(jin) if jin else price_in,
            judge_price_out_per_m=float(jout) if jout else price_out,
            judge_prices_set=bool(jin and jout),
        )

    def snapshot(self) -> dict:
        """Non-secret settings, safe to write into the run manifest."""
        return {
            "api_version": self.api_version,
            "chat_deployment": self.chat_deployment,
            "judge_deployment": self.judge_deployment,
            "reasoning_effort": self.reasoning_effort,
            "price_in_per_m": self.price_in_per_m,
            "price_out_per_m": self.price_out_per_m,
            "judge_price_in_per_m": self.judge_price_in_per_m,
            "judge_price_out_per_m": self.judge_price_out_per_m,
            "judge_prices_set": self.judge_prices_set,
        }
