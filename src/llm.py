"""Minimal LLM client over plain HTTPS (no vendor SDK needed).

Set ONE of these environment variables to enable LLM generation:
  GEMINI_API_KEY     (Google AI Studio, has a free tier)
  OPENAI_API_KEY
  ANTHROPIC_API_KEY
  GROQ_API_KEY       (free tier, OpenAI-compatible)
Optionally set LLM_MODEL to override the default model for that provider.
Without a key the system uses the extractive generator in generation.py.
"""
from __future__ import annotations

import os

import requests

TIMEOUT = 60


def provider() -> str | None:
    for name, var in [("gemini", "GEMINI_API_KEY"), ("openai", "OPENAI_API_KEY"),
                      ("anthropic", "ANTHROPIC_API_KEY"), ("groq", "GROQ_API_KEY")]:
        if os.getenv(var):
            return name
    return None


def model_name() -> str | None:
    p = provider()
    defaults = {"gemini": "gemini-2.5-flash", "openai": "gpt-4o-mini",
                "anthropic": "claude-haiku-4-5", "groq": "llama-3.3-70b-versatile"}
    return os.getenv("LLM_MODEL", defaults.get(p)) if p else None


def complete(system: str, user: str, max_tokens: int = 800) -> str:
    p, model = provider(), model_name()
    if p is None:
        raise RuntimeError("No LLM API key set")
    if p == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        r = requests.post(url, params={"key": os.environ["GEMINI_API_KEY"]}, timeout=TIMEOUT, json={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens}})
        r.raise_for_status()
        return "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
    if p == "anthropic":
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=TIMEOUT, headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
            json={"model": model, "max_tokens": max_tokens, "temperature": 0, "system": system,
                  "messages": [{"role": "user", "content": user}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])
    base = "https://api.openai.com/v1" if p == "openai" else "https://api.groq.com/openai/v1"
    key = os.environ["OPENAI_API_KEY" if p == "openai" else "GROQ_API_KEY"]
    r = requests.post(f"{base}/chat/completions", timeout=TIMEOUT,
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": model, "temperature": 0, "max_tokens": max_tokens,
                            "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": user}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
