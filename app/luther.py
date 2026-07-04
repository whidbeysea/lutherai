"""Core chat logic: retrieve relevant passages, assemble the system prompt, and call
an LLM to generate Luther's response. This module is the shared core used by both the
CLI test harness (scripts/chat_cli.py) and the FastAPI app (app/main.py).

Supports two backends, chosen via the LLM_BACKEND env var ("anthropic" | "ollama"),
so a local open-source model can be tried and compared against Claude Haiku without
touching retrieval, the system prompt, or any guardrail logic -- only this module's
generation call differs between them. Defaults to "anthropic"; switching back is a
one-line .env edit and a service restart, no code change.
"""
import os
from pathlib import Path

import anthropic
import requests
from dotenv import load_dotenv

from app.retrieval import retrieve

load_dotenv()

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.md"

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

_base_system_prompt = None
_anthropic_client = None


def _client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def _base_prompt() -> str:
    global _base_system_prompt
    if _base_system_prompt is None:
        _base_system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    return _base_system_prompt


def _format_context(passages: list[dict]) -> str:
    if not passages:
        return "(No relevant passages were found in your writings for this question.)"
    blocks = []
    for p in passages:
        meta = p["metadata"]
        blocks.append(f"[{meta['source']}, {meta['year']}]\n{p['text']}")
    return "\n\n---\n\n".join(blocks)


def _context_block(passages: list[dict]) -> str:
    return "## Passages from your own writings, relevant to this question\n\n" + _format_context(passages)


def _generate_anthropic(passages: list[dict], messages: list[dict]) -> dict:
    """Base prompt and retrieved-context are sent as separate system blocks, with the
    base prompt marked cache_control so Anthropic caches it (~90% cheaper on repeat
    calls within the 5-minute cache window) once it's long enough to clear Haiku's
    4096-token minimum for caching to activate."""
    system_blocks = [
        {"type": "text", "text": _base_prompt(), "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": _context_block(passages)},
    ]

    response = _client().messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_TOKENS,
        system=system_blocks,
        messages=messages,
    )
    text = "".join(block.text for block in response.content if block.type == "text")

    return {
        "response": text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_creation_input_tokens": response.usage.cache_creation_input_tokens,
        "cache_read_input_tokens": response.usage.cache_read_input_tokens,
    }


def _generate_ollama(passages: list[dict], messages: list[dict]) -> dict:
    """Ollama has no prompt-caching concept, so the base prompt and retrieved context
    are just concatenated into a single system message alongside conversation history."""
    system_text = _base_prompt() + "\n\n" + _context_block(passages)
    ollama_messages = [{"role": "system", "content": system_text}] + messages

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": OLLAMA_MODEL, "messages": ollama_messages, "stream": False},
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()

    return {
        "response": data["message"]["content"],
        "input_tokens": data.get("prompt_eval_count", 0),
        "output_tokens": data.get("eval_count", 0),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


BACKENDS = {
    "anthropic": _generate_anthropic,
    "ollama": _generate_ollama,
}


def ask_luther(question: str, history: list[dict] | None = None) -> dict:
    """Returns {"response": str, "retrieved": list[dict], ...usage fields} for a single
    user question.

    history, if given, is a list of {"role": "user"|"assistant", "content": str} from
    earlier turns in the same session -- passed through for conversational continuity,
    but retrieval is always keyed off the latest question only (v1; a fancier version
    might rewrite the query using conversation context).
    """
    passages = retrieve(question)

    messages = list(history or [])
    messages.append({"role": "user", "content": question})

    backend_name = os.environ.get("LLM_BACKEND", "anthropic")
    backend = BACKENDS.get(backend_name)
    if backend is None:
        raise RuntimeError(f"Unknown LLM_BACKEND: {backend_name!r} (expected one of {list(BACKENDS)})")

    result = backend(passages, messages)
    result["retrieved"] = passages
    return result
