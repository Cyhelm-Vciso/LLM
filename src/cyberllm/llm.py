"""Minimal chat backends.

  openai-compatible  Any server exposing POST /v1/chat/completions: vLLM, Ollama,
                     LM Studio, llama.cpp server, TGI, LiteLLM, and hosted
                     inference providers. Used for the generator (teacher) and for
                     serving your fine-tuned model during evaluation.
  anthropic          Claude via the official Anthropic SDK. Used as an evaluation
                     judge or as a baseline to compare your model against.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"


class ChatBackend(Protocol):
    name: str

    def chat(self, system: str, messages: list[dict], max_tokens: int = 4096, json_schema: dict | None = None) -> str: ...


class BackendRefusal(RuntimeError):
    """The model declined the request (e.g. a safety classifier)."""


@dataclass
class OpenAICompatible:
    base_url: str
    model: str
    api_key: str | None = None
    temperature: float = 0.7
    timeout: float = 600.0
    retries: int = 3
    name: str = "openai-compatible"

    def chat(self, system: str, messages: list[dict], max_tokens: int = 4096, json_schema: dict | None = None) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        key = self.api_key or os.environ.get("OPENAI_COMPAT_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload = {
            "model": self.model,
            "messages": ([{"role": "system", "content": system}] if system else []) + messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        if json_schema is not None:
            # Widely supported (vLLM, Ollama, LM Studio). Servers that don't support it ignore or reject it;
            # callers still parse defensively.
            payload["response_format"] = {"type": "json_object"}
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
                if r.status_code == 400 and "response_format" in payload:
                    payload.pop("response_format")  # server doesn't support it; retry without
                    continue
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            except (httpx.HTTPError, KeyError, IndexError) as e:
                last_err = e
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500 and e.response.status_code != 429:
                    break
                time.sleep(2**attempt)
        raise RuntimeError(f"{url}: request failed: {last_err}")


@dataclass
class Claude:
    model: str = DEFAULT_CLAUDE_MODEL
    effort: str = "medium"
    name: str = "anthropic"

    def __post_init__(self) -> None:
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise SystemExit("Install the Claude extra: pip install -e '.[claude]'") from e
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY / `ant auth login` profile

    def chat(self, system: str, messages: list[dict], max_tokens: int = 16000, json_schema: dict | None = None) -> str:
        output_config: dict = {"effort": self.effort}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        kwargs: dict = {}
        if self.model in ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"):
            # Server-side fallback: if a safety classifier declines (e.g. a benign security
            # question misread as harmful), the API retries on Anthropic's recommended model.
            kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        resp = self._client.beta.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            output_config=output_config,
            **kwargs,
        )
        if resp.stop_reason == "refusal":
            category = getattr(resp.stop_details, "category", None) if resp.stop_details else None
            raise BackendRefusal(f"Claude declined (category={category})")
        return "".join(b.text for b in resp.content if b.type == "text")


def make_backend(spec: str, base_url: str | None = None, api_key: str | None = None, temperature: float = 0.7) -> ChatBackend:
    """spec examples: 'openai:qwen2.5:14b' (needs base_url), 'anthropic', 'anthropic:claude-sonnet-5-5'."""
    kind, _, model = spec.partition(":")
    if kind == "anthropic":
        return Claude(model=model or DEFAULT_CLAUDE_MODEL)
    if kind == "openai":
        if not base_url:
            raise SystemExit("--base-url is required for openai-compatible backends (e.g. http://localhost:11434/v1)")
        if not model:
            raise SystemExit("model name required, e.g. openai:qwen2.5:14b")
        return OpenAICompatible(base_url=base_url, model=model, api_key=api_key, temperature=temperature)
    raise SystemExit(f"unknown backend '{spec}'. Use 'openai:<model>' or 'anthropic[:<model>]'")


def extract_json(text: str) -> dict:
    """Parse the first JSON object in a model reply (tolerates code fences and preamble)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    raise ValueError("no JSON object found in model output")
