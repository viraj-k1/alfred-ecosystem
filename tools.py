"""
tools.py — Custom CrewAI Tools
==============================
- Mike (Mem0) tools: save_preference, search_memory
- Web scraping placeholder (Playwright)

All tools are registered as @tool decorated functions so CrewAI agents can
call them natively.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from crewai.tools import tool
from mem0 import Memory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mike — Mem0 Memory Layer
# ---------------------------------------------------------------------------
# We initialise Mem0 once at module load time and reuse it across all tool
# calls. The local config uses ChromaDB so no external service is needed.

_MEM0_USER_ID = "primary_user"  # single-user system

def _build_mem0_config() -> dict:
    """Return a Mem0 config dict.

    Uses local ChromaDB so nothing external is required on Windows.
    If MEM0_API_KEY is set, it switches to the Mem0 cloud platform instead.
    """
    if os.getenv("MEM0_API_KEY"):
        # Cloud mode
        return {}

    # Local mode — ChromaDB persisted in ./mike_db/
    return {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "alfred_memory",
                "path": "./mike_db",
            },
        },
        "llm": {
            "provider": "litellm",
            "config": {
                "model": "gemini/gemini-3.8-flash",
                "api_key": os.getenv("GEMINI_API_KEY", ""),
            },
        },
        "embedder": {
            "provider": "gemini",
            "config": {
                "model": "models/gemini-embedding-001",
                "api_key": os.getenv("GEMINI_API_KEY", ""),
            },
        },
    }


_mike: Optional[Memory] = None


def get_mike() -> Memory:
    """Return the singleton Mem0 Memory instance (lazy initialisation)."""
    global _mike
    if _mike is None:
        config = _build_mem0_config()
        _mike = Memory.from_config(config) if config else Memory()
        logger.info("Mike (Mem0) initialised successfully.")
    return _mike


# ---------------------------------------------------------------------------
# Tool: save_preference
# ---------------------------------------------------------------------------

@tool("save_preference")
def save_preference(preference: str) -> str:
    """Save a user preference or personal fact to Mike (Mem0 memory layer).

    Use this whenever the user states a preference, personal fact, or
    constraint (e.g., "I prefer remote roles", "I study at NYU").

    Args:
        preference: The plain-text preference or fact to persist.

    Returns:
        Confirmation string.
    """
    try:
        mike = get_mike()
        result = mike.add(
            messages=[{"role": "user", "content": preference}],
            user_id=_MEM0_USER_ID,
        )
        logger.info("Preference saved to Mike: %s | result=%s", preference, result)
        return f"✅ Preference saved to memory: \"{preference}\""
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save preference to Mike: %s", exc)
        return f"⚠️ Could not save preference. Error: {exc}"


# ---------------------------------------------------------------------------
# Tool: search_memory
# ---------------------------------------------------------------------------

@tool("search_memory")
def search_memory(query: str) -> str:
    """Search Mike (Mem0) for stored user preferences relevant to a query.

    Use this before executing any task to retrieve context about the user's
    preferences, constraints, or personal facts.

    Args:
        query: A natural-language search query, e.g. "internship preferences"
               or "user location and university".

    Returns:
        A formatted string of relevant memories, or a notice if none found.
    """
    try:
        mike = get_mike()
        results = mike.search(query=query, user_id=_MEM0_USER_ID, limit=10)

        if not results or not results.get("results"):
            return "ℹ️ No relevant memories found for this query."

        memories = results["results"]
        lines = ["📋 **Relevant User Preferences:**"]
        for i, mem in enumerate(memories, 1):
            memory_text = mem.get("memory", mem.get("text", str(mem)))
            score = mem.get("score", 0)
            lines.append(f"{i}. {memory_text} (relevance: {score:.2f})")

        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to search Mike: %s", exc)
        return f"⚠️ Memory search failed. Error: {exc}"


# ---------------------------------------------------------------------------
# Tool: get_all_memories
# ---------------------------------------------------------------------------

@tool("get_all_memories")
def get_all_memories() -> str:
    """Retrieve ALL stored preferences and facts for the user from Mike.

    Use this for daily briefings or when you need a complete picture of
    the user's profile.

    Returns:
        A formatted string of all memories.
    """
    try:
        mike = get_mike()
        results = mike.get_all(user_id=_MEM0_USER_ID)

        if not results or not results.get("results"):
            return "ℹ️ No memories stored yet."

        memories = results["results"]
        lines = ["📋 **All User Preferences & Facts:**"]
        for i, mem in enumerate(memories, 1):
            memory_text = mem.get("memory", mem.get("text", str(mem)))
            lines.append(f"{i}. {memory_text}")

        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to retrieve all memories from Mike: %s", exc)
        return f"⚠️ Could not retrieve memories. Error: {exc}"


# ---------------------------------------------------------------------------
# Tool: web_search_placeholder (Playwright stub)
# ---------------------------------------------------------------------------

@tool("web_search_placeholder")
def web_search_placeholder(query: str) -> str:
    """[PLACEHOLDER] Perform a web search for job listings or other information.

    This tool is a stub. To activate real scraping, install Playwright
    (``playwright install chromium``) and implement the search logic below.

    Args:
        query: The search query string.

    Returns:
        A placeholder notice with the query echoed back.
    """
    logger.warning(
        "web_search_placeholder called with query=%r — "
        "real Playwright scraping not yet implemented.",
        query,
    )
    # -----------------------------------------------------------------------
    # IMPLEMENTATION GUIDE (uncomment and fill in when ready):
    # -----------------------------------------------------------------------
    # from playwright.sync_api import sync_playwright
    #
    # with sync_playwright() as p:
    #     browser = p.chromium.launch(headless=True)
    #     page = browser.new_page()
    #     page.goto(f"https://www.google.com/search?q={query}")
    #     # extract results…
    #     results_text = page.inner_text("body")
    #     browser.close()
    # return results_text
    # -----------------------------------------------------------------------

    return (
        f"🔍 [Placeholder] Web search for: \"{query}\"\n\n"
        "Real Playwright scraping is not yet implemented. "
        "To enable it, install Playwright (``playwright install chromium``)\n"
        "and implement the search logic in tools.py → web_search_placeholder.\n\n"
        "For now, Bob will use his trained knowledge to answer this query."
    )
