"""
orchestrator.py — CrewAI Orchestration Layer
=============================================
Loads agents/tasks from YAML, wires up LiteLLM routing with fallbacks,
and exposes a single async function `handle_message()` that main.py calls.
"""

from __future__ import annotations

import asyncio
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import litellm
import yaml
from crewai import Agent, Crew, Process, Task
from litellm import Router

from tools import (
    get_all_memories,
    save_preference,
    search_memory,
    web_search_placeholder,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# YAML Loaders
# ---------------------------------------------------------------------------

_BASE = Path(__file__).parent


def _load_yaml(filename: str) -> dict[str, Any]:
    with open(_BASE / filename, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def _agents_cfg() -> dict[str, Any]:
    return _load_yaml("agents.yaml")


@lru_cache(maxsize=1)
def _tasks_cfg() -> dict[str, Any]:
    return _load_yaml("tasks.yaml")


# ---------------------------------------------------------------------------
# LiteLLM Router — Model Routing + Fallback Chain
# ---------------------------------------------------------------------------

def build_litellm_router() -> Router:
    """Build a LiteLLM Router with per-agent model assignments.

    Bob's heavy model (gemini-3.1-pro-preview) falls back to
    gemini-3.8-flash if it hits a rate-limit or timeout.
    """
    primary_key = os.environ["GEMINI_API_KEY"]
    bob_key = os.getenv("GEMINI_API_KEY_BOB", primary_key)

    model_list = [
        # ── Alfred / Edna — fast model ─────────────────────────────────────
        {
            "model_name": "fast-model",
            "litellm_params": {
                "model": "gemini/gemini-3.8-flash",
                "api_key": primary_key,
            },
        },
        # ── Bob — primary heavy model ──────────────────────────────────────
        {
            "model_name": "heavy-model",
            "litellm_params": {
                "model": "gemini/gemini-3.1-pro-preview",
                "api_key": bob_key,
            },
        },
        # ── Bob — fallback to fast model if heavy hits limits ──────────────
        {
            "model_name": "heavy-model",
            "litellm_params": {
                "model": "gemini/gemini-3.8-flash",
                "api_key": primary_key,
            },
        },
    ]

    router = Router(
        model_list=model_list,
        # Retry up to 3 times before falling to next model in group
        num_retries=3,
        # Wait up to 60 s per call before counting as timeout
        timeout=60,
        # Fall back to the next entry with the same model_name on these errors
        allowed_fails=2,
        retry_after=5,
    )

    logger.info("LiteLLM Router initialised with fast + heavy (+ fallback) models.")
    return router


# Singleton router — created once at import time
_router: Router | None = None


def get_router() -> Router:
    global _router
    if _router is None:
        _router = build_litellm_router()
    return _router


# ---------------------------------------------------------------------------
# LiteLLM completion helper (used by agents via CrewAI's llm= parameter)
# ---------------------------------------------------------------------------

def _make_llm_config(model_alias: str) -> dict[str, Any]:
    """Return a dict suitable for passing to CrewAI Agent(llm=...)."""
    # CrewAI accepts a dict with 'model' and optional keys that map to
    # litellm.completion kwargs. We patch through the router by overriding
    # litellm's completion with the router's completion at startup.
    return {"model": model_alias}


# ---------------------------------------------------------------------------
# Patch litellm.completion to route through our Router
# ---------------------------------------------------------------------------

def _patch_litellm_with_router(router: Router) -> None:
    """Monkey-patch litellm.completion so CrewAI calls route through our Router."""
    original_completion = litellm.completion

    def patched_completion(*args: Any, **kwargs: Any):  # noqa: ANN401
        model = kwargs.get("model", args[0] if args else "fast-model")
        # Only intercept our named aliases; pass others through unchanged
        if model in {"fast-model", "heavy-model"}:
            logger.debug("Router intercept → model=%s", model)
            return router.completion(*args, **kwargs)
        return original_completion(*args, **kwargs)

    litellm.completion = patched_completion
    logger.info("litellm.completion patched to use LiteLLM Router.")


# ---------------------------------------------------------------------------
# Agent Factory
# ---------------------------------------------------------------------------

def _build_alfred(cfg: dict) -> Agent:
    return Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        tools=[save_preference, search_memory, get_all_memories],
        llm="fast-model",
        verbose=cfg.get("verbose", False),
        allow_delegation=cfg.get("allow_delegation", True),
        max_iter=cfg.get("max_iterations", 5),
    )


def _build_edna(cfg: dict) -> Agent:
    return Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        tools=[search_memory, get_all_memories],
        llm="fast-model",
        verbose=cfg.get("verbose", False),
        allow_delegation=cfg.get("allow_delegation", False),
        max_iter=cfg.get("max_iterations", 3),
    )


def _build_bob(cfg: dict) -> Agent:
    return Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        tools=[search_memory, web_search_placeholder],
        llm="heavy-model",   # Router will fall back to fast-model automatically
        verbose=cfg.get("verbose", True),
        allow_delegation=cfg.get("allow_delegation", False),
        max_iter=cfg.get("max_iterations", 8),
    )


# ---------------------------------------------------------------------------
# Crew Assembler
# ---------------------------------------------------------------------------

def _build_crew(
    user_message: str,
    user_preferences: str,
    task_type: str = "route_and_respond",
    extra_context: dict | None = None,
) -> Crew:
    """Assemble a Crew dynamically for a single user request."""
    agents_cfg = _agents_cfg()
    tasks_cfg = _tasks_cfg()

    alfred = _build_alfred(agents_cfg["alfred"])
    edna = _build_edna(agents_cfg["edna"])
    bob = _build_bob(agents_cfg["bob"])

    # Fill in YAML task template placeholders
    extra = extra_context or {}
    placeholders = {
        "user_message": user_message,
        "user_preferences": user_preferences,
        "bob_summary": extra.get("bob_summary", "No summary available yet."),
        "bob_output": extra.get("bob_output", ""),
        "preference_text": extra.get("preference_text", user_message),
    }

    task_cfg = tasks_cfg[task_type]
    task_description = task_cfg["description"].format(**placeholders)
    task_expected_output = task_cfg["expected_output"]

    # Map task type to responsible agent
    agent_map = {"alfred": alfred, "edna": edna, "bob": bob}
    responsible_agent = agent_map.get(task_cfg.get("agent", "alfred"), alfred)

    main_task = Task(
        description=task_description,
        expected_output=task_expected_output,
        agent=responsible_agent,
    )

    return Crew(
        agents=[alfred, edna, bob],
        tasks=[main_task],
        process=Process.sequential,
        verbose=False,
        memory=False,   # We manage memory ourselves via Mem0/Mike
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def initialise_orchestrator() -> None:
    """Call once at startup to warm up the router and patch litellm."""
    router = get_router()
    _patch_litellm_with_router(router)
    # Pre-load YAML configs
    _agents_cfg()
    _tasks_cfg()
    logger.info("Orchestrator ready.")


async def handle_message(user_message: str) -> str:
    """Process a Telegram message through the multi-agent crew.

    This is the single entry point called by main.py.

    Args:
        user_message: Raw text from the Telegram user.

    Returns:
        Final response string to send back via Telegram.
    """
    from tools import search_memory as _search_memory  # avoid circular at module level

    # 1. Retrieve relevant memories before spinning up the crew
    loop = asyncio.get_event_loop()
    user_preferences = await loop.run_in_executor(
        None, _search_memory, user_message
    )

    # 2. Determine task type heuristically
    msg_lower = user_message.lower()
    if any(k in msg_lower for k in ("job", "intern", "career", "role", "position", "hiring", "recruit")):
        task_type = "job_search"
    elif any(k in msg_lower for k in ("briefing", "brief", "summary", "today", "agenda")):
        task_type = "daily_briefing"
    else:
        task_type = "route_and_respond"

    logger.info("Routing message to task_type=%s", task_type)

    # 3. Run the Crew in a thread (CrewAI is synchronous)
    crew = _build_crew(user_message, user_preferences, task_type)
    result = await loop.run_in_executor(None, crew.kickoff)

    # CrewAI returns a CrewOutput object; extract the string
    response_text = str(result).strip() if result else "I'm sorry, I wasn't able to process that request."
    return response_text
