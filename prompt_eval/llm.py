"""LLM access: a real Azure OpenAI client and an offline mock with the same interface.

Design notes
- Reasoning models (e.g. gpt-5-mini) reject `temperature` and `max_tokens`, so neither is set.
  `reasoning_effort` is only passed when REASONING_EFFORT is set.
- Latency is measured around the successful attempt only; retries are counted separately.
- Every call is wrapped with LangSmith's @traceable (a no-op if tracing is off or langsmith
  is not installed), tagged with variant / input / rep so runs can be filtered in the UI.
- Set LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY / LANGCHAIN_PROJECT BEFORE the first call.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from typing import Protocol

from .config import Config
from .data import VARIANTS

try:
    from langsmith import traceable
except ImportError:  # keep the harness usable without langsmith installed

    def traceable(*dargs, **dkwargs):  # type: ignore[no-redef]
        def deco(fn):
            def wrapper(*args, **kwargs):
                kwargs.pop("langsmith_extra", None)
                return fn(*args, **kwargs)

            return wrapper

        return deco


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    latency_s: float
    retries: int = 0


class LLM(Protocol):
    def invoke(self, prompt: str, meta: dict | None = None) -> LLMResult: ...


class AzureLLM:
    def __init__(self, cfg: Config, deployment: str | None = None, max_attempts: int = 5, trace_name: str = "PromptVariantRun"):
        from langchain_openai import AzureChatOpenAI

        kwargs = dict(
            azure_endpoint=cfg.endpoint,
            api_version=cfg.api_version,
            azure_deployment=deployment or cfg.chat_deployment,
            max_retries=0,  # we handle retries ourselves so latency stays honest
        )
        if cfg.reasoning_effort:
            kwargs["reasoning_effort"] = cfg.reasoning_effort
        self._model = AzureChatOpenAI(**kwargs)
        self._max_attempts = max_attempts
        self._call = traceable(name=trace_name)(self._raw_call)

    def _raw_call(self, prompt: str) -> dict:
        from langchain_core.messages import HumanMessage

        start = time.perf_counter()
        resp = self._model.invoke([HumanMessage(content=prompt)])
        latency = time.perf_counter() - start
        usage = getattr(resp, "usage_metadata", None) or {}
        details = usage.get("output_token_details") or {}
        return {
            "text": resp.content if isinstance(resp.content, str) else str(resp.content),
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
            "reasoning_tokens": int(details.get("reasoning", 0)),
            "latency_s": latency,
        }

    def invoke(self, prompt: str, meta: dict | None = None) -> LLMResult:
        import openai

        non_retryable = (
            openai.AuthenticationError,
            openai.NotFoundError,
            openai.BadRequestError,
            openai.PermissionDeniedError,
        )
        meta = meta or {}
        extra = {"metadata": meta, "tags": [f"variant:{meta.get('variant_id', '?')}"]}
        last_exc: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                out = self._call(prompt, langsmith_extra=extra)
                return LLMResult(retries=attempt, **out)
            except non_retryable:
                raise
            except Exception as exc:  # rate limits, timeouts, transient 5xx
                last_exc = exc
                time.sleep(min(30.0, 2**attempt + random.random()))
        raise RuntimeError(f"gave up after {self._max_attempts} attempts: {type(last_exc).__name__}") from last_exc


# --------------------------------------------------------------------------------------
# Offline mock: lets the whole pipeline run (and be tested) with no credentials.
# Its numbers are SYNTHETIC and only exercise the plumbing. Never report them as results.
# --------------------------------------------------------------------------------------

_MOCK_WORDS = {  # (mean, sd) of summary length per variant
    "v1": (23, 1.0),
    "v2": (21, 3.0),
    "v3": (16, 3.0),
    "v4": (120, 45.0),
    "v5": (22, 2.0),
    "v6": (20, 1.5),
}


class MockLLM:
    def __init__(self, seed: int = 0):
        self._seed = seed
        self._n = 0

    def _rng(self, prompt: str) -> random.Random:
        self._n += 1
        h = hashlib.sha256(f"{self._seed}|{self._n}|{prompt}".encode()).hexdigest()
        return random.Random(int(h[:12], 16))

    def invoke(self, prompt: str, meta: dict | None = None) -> LLMResult:
        rng = self._rng(prompt)
        if "Respond with only JSON" in prompt:
            text, vis, reasoning = self._judge(prompt, rng), 40, rng.choice([0, 64])
        else:
            text, vis, reasoning = self._summarize(prompt, rng)
        in_tok = int(len(prompt.split()) * 1.3)
        out_tok = vis + reasoning
        latency = 0.8 + out_tok * 0.006 + rng.lognormvariate(-1.5, 0.5)
        return LLMResult(text, in_tok, out_tok, reasoning, latency)

    def _summarize(self, prompt: str, rng: random.Random):
        vid, prefix = next(((v, p) for v, p in VARIANTS.items() if prompt.startswith(p)), ("v1", ""))
        source = prompt[len(prefix):]
        mean, sd = _MOCK_WORDS[vid]
        n = max(5, int(rng.gauss(mean, sd)))
        words = source.replace(". ", ", ").replace(".", "").split()
        picked = (words * 3)[:n]
        if n > 60:  # long answers look like bullet lists
            chunks = [" ".join(picked[i : i + 12]) for i in range(0, len(picked), 12)]
            text = "\n".join("- " + c + "." for c in chunks)
        else:
            text = " ".join(picked) + "."
        vis = int(len(picked) * 1.35)
        reasoning = rng.choice([192, 256, 320, 384]) if vid == "v4" else rng.choice([0, 64, 64, 128])
        return text, vis, reasoning

    def _judge(self, prompt: str, rng: random.Random) -> str:
        m = re.search(r"SUMMARY:\n<<<\n(.*?)\n>>>", prompt, re.DOTALL)
        w = len(m.group(1).split()) if m else 10
        faith = 5 if rng.random() > 0.2 else 4
        cover = max(1, min(5, 2 + w // 6 + rng.choice([-1, 0, 0, 1])))
        clarity = rng.choice([4, 4, 5])
        return json.dumps(
            {"faithfulness": faith, "coverage": cover, "clarity": clarity, "rationale": "mock judgement"}
        )
