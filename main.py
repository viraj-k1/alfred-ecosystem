"""
main.py — Alfred Multi-Agent Ecosystem Entry Point
===================================================
Initialises:
  1. Logging
  2. .env loading
  3. LiteLLM Router + CrewAI Orchestrator
  4. Telegram Bot (python-telegram-bot v20+, async)

The bot only exposes Alfred to the user. All routing and delegation to
Edna, Bob, and Mike happens transparently inside orchestrator.py.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.logging import RichHandler
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ---------------------------------------------------------------------------
# Bootstrap .env before importing anything else that reads env vars
# ---------------------------------------------------------------------------
load_dotenv(Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("alfred")

# Silence noisy third-party loggers
for noisy in ("httpx", "httpcore", "telegram", "crewai", "litellm", "chromadb"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Validate required env vars early
# ---------------------------------------------------------------------------
_REQUIRED_VARS = ["TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY"]

def _validate_env() -> None:
    missing = [v for v in _REQUIRED_VARS if not os.getenv(v)]
    if missing:
        logger.error(
            "❌ Missing required environment variables: %s\n"
            "   Copy .env.example → .env and fill in your keys.",
            ", ".join(missing),
        )
        sys.exit(1)

_validate_env()

# ---------------------------------------------------------------------------
# Import orchestrator AFTER env vars are loaded
# ---------------------------------------------------------------------------
from orchestrator import handle_message, initialise_orchestrator  # noqa: E402

# ---------------------------------------------------------------------------
# Access Control (optional)
# ---------------------------------------------------------------------------

def _get_allowed_ids() -> set[int]:
    raw = os.getenv("ALLOWED_USER_IDS", "").strip()
    if not raw:
        return set()   # empty = allow all
    try:
        return {int(uid.strip()) for uid in raw.split(",") if uid.strip()}
    except ValueError:
        logger.warning("ALLOWED_USER_IDS contains non-integer values; ignoring.")
        return set()

_ALLOWED_IDS = _get_allowed_ids()


def _is_allowed(user_id: int) -> bool:
    if not _ALLOWED_IDS:
        return True   # no allowlist → open to all
    return user_id in _ALLOWED_IDS


# ---------------------------------------------------------------------------
# Telegram Handlers
# ---------------------------------------------------------------------------

async def _cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    user = update.effective_user
    if not _is_allowed(user.id):
        await update.message.reply_text("🚫 Access denied.")
        return

    greeting = (
        f"*Good day, {user.first_name}.*\n\n"
        "I'm Alfred — your chief of staff and AI orchestrator.\n\n"
        "I have access to:\n"
        "• 🧠 *Mike* — your personal memory vault\n"
        "• 📋 *Edna* — your executive monitor & briefing agent\n"
        "• 💼 *Bob* — your career intelligence & headhunting agent\n\n"
        "How may I assist you today?"
    )
    await update.message.reply_text(greeting, parse_mode=ParseMode.MARKDOWN)


async def _cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    help_text = (
        "*Alfred — Command Reference*\n\n"
        "/start — Introduction and status\n"
        "/help — This help message\n"
        "/briefing — Request a daily briefing from Edna\n"
        "/jobs — Trigger Bob's internship search\n"
        "/memory — Show all stored preferences\n"
        "/clear — [Admin] Clear all stored memories\n\n"
        "Or just *send any message* and I'll route it to the right agent."
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)


async def _cmd_briefing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Trigger a daily briefing from Edna."""
    if not _is_allowed(update.effective_user.id):
        await update.message.reply_text("🚫 Access denied.")
        return
    await _process_message(update, context, override_message="Give me my daily briefing.")


async def _cmd_jobs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Trigger Bob's internship search."""
    if not _is_allowed(update.effective_user.id):
        await update.message.reply_text("🚫 Access denied.")
        return
    await _process_message(
        update, context,
        override_message="Search for Summer 2027 internship opportunities for me."
    )


async def _cmd_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show all stored preferences."""
    if not _is_allowed(update.effective_user.id):
        await update.message.reply_text("🚫 Access denied.")
        return
    from tools import get_all_memories
    loop = asyncio.get_event_loop()
    memories = await loop.run_in_executor(None, get_all_memories)
    await update.message.reply_text(memories, parse_mode=ParseMode.MARKDOWN)


async def _cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear all stored memories (admin action)."""
    if not _is_allowed(update.effective_user.id):
        await update.message.reply_text("🚫 Access denied.")
        return
    try:
        from tools import get_mike
        mike = get_mike()
        mike.delete_all(user_id="primary_user")
        await update.message.reply_text("🗑️ All memories cleared.")
    except Exception as exc:  # noqa: BLE001
        await update.message.reply_text(f"⚠️ Could not clear memories: {exc}")


async def _process_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    override_message: str | None = None,
) -> None:
    """Core handler: send user message through Alfred and reply."""
    user = update.effective_user

    if not _is_allowed(user.id):
        await update.message.reply_text("🚫 Access denied.")
        return

    user_message = override_message or (update.message.text or "").strip()
    if not user_message:
        return

    logger.info("Incoming message from %s (%d): %r", user.first_name, user.id, user_message[:80])

    # Show typing indicator while processing
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING,
    )

    try:
        response = await handle_message(user_message)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error in handle_message: %s", exc)
        response = (
            "⚠️ I encountered an error processing your request. "
            "Please try again in a moment."
        )

    # Telegram has a 4096-character limit; split if needed
    max_len = 4000
    if len(response) <= max_len:
        await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
    else:
        chunks = [response[i : i + max_len] for i in range(0, len(response), max_len)]
        for i, chunk in enumerate(chunks):
            parse = ParseMode.MARKDOWN if i == 0 else None
            await update.message.reply_text(chunk, parse_mode=parse)


async def _handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for plain text messages."""
    await _process_message(update, context)


# ---------------------------------------------------------------------------
# Error Handler
# ---------------------------------------------------------------------------

async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Telegram update caused exception: %s", context.error, exc_info=context.error)
    if isinstance(update, Update) and update.message:
        await update.message.reply_text(
            "⚠️ Something went wrong internally. Alfred is investigating."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Start the bot."""
    logger.info("🚀 Alfred Ecosystem starting up…")

    # 1. Warm up the orchestrator (loads YAML, patches litellm, warms Mike)
    initialise_orchestrator()

    # 2. Build Telegram application
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = (
        Application.builder()
        .token(token)
        .read_timeout(60)
        .write_timeout(60)
        .connect_timeout(30)
        .pool_timeout(60)
        .build()
    )

    # 3. Register handlers
    app.add_handler(CommandHandler("start", _cmd_start))
    app.add_handler(CommandHandler("help", _cmd_help))
    app.add_handler(CommandHandler("briefing", _cmd_briefing))
    app.add_handler(CommandHandler("jobs", _cmd_jobs))
    app.add_handler(CommandHandler("memory", _cmd_memory))
    app.add_handler(CommandHandler("clear", _cmd_clear))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_text))
    app.add_error_handler(_error_handler)

    # 4. Run (long-polling — no webhook needed)
    logger.info("✅ Alfred is online. Polling for messages…")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
