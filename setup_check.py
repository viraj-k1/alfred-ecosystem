"""
setup_check.py — Pre-flight Environment Checker
================================================
Run this BEFORE starting main.py to verify your setup is correct.
Usage: python setup_check.py
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def check(label: str, ok: bool, detail: str = "") -> bool:
    icon = f"{GREEN}✅{RESET}" if ok else f"{RED}❌{RESET}"
    suffix = f" — {detail}" if detail else ""
    print(f"  {icon}  {label}{suffix}")
    return ok


def section(title: str) -> None:
    print(f"\n{BOLD}{BLUE}{'─'*50}{RESET}")
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(f"{BOLD}{BLUE}{'─'*50}{RESET}")


all_ok = True

# ── Python Version ──────────────────────────────────────────────────────────
section("Python Version")
py_ok = sys.version_info >= (3, 10)
all_ok &= check(
    f"Python {sys.version_info.major}.{sys.version_info.minor}",
    py_ok,
    "Need 3.10+" if not py_ok else "",
)

# ── Required Files ──────────────────────────────────────────────────────────
section("Project Files")
required_files = [
    "main.py", "orchestrator.py", "tools.py",
    "agents.yaml", "tasks.yaml", "requirements.txt", ".env",
]
for fname in required_files:
    exists = Path(fname).exists()
    all_ok &= check(fname, exists, "MISSING" if not exists else "")

# ── Environment Variables ───────────────────────────────────────────────────
section("Environment Variables")
env_vars = {
    "TELEGRAM_BOT_TOKEN": True,   # required
    "GEMINI_API_KEY": True,        # required
    "GEMINI_API_KEY_BOB": False,   # optional (falls back to GEMINI_API_KEY)
    "ALLOWED_USER_IDS": False,     # optional
    "LOG_LEVEL": False,            # optional
}
for var, required in env_vars.items():
    val = os.getenv(var, "")
    is_set = bool(val)
    label = var + (" (required)" if required else " (optional)")
    if required:
        all_ok &= check(label, is_set, "NOT SET — add to .env" if not is_set else "✓ set")
    else:
        check(label, True, f"= {val!r}" if is_set else "not set (OK)")

# ── Python Packages ─────────────────────────────────────────────────────────
section("Python Packages")
packages = {
    "telegram": "python-telegram-bot",
    "crewai": "crewai",
    "litellm": "litellm",
    "mem0": "mem0ai",
    "chromadb": "chromadb",
    "dotenv": "python-dotenv",
    "yaml": "pyyaml",
    "rich": "rich",
    "tenacity": "tenacity",
}
for mod, pkg in packages.items():
    try:
        importlib.import_module(mod)
        check(pkg, True)
    except ImportError:
        all_ok &= check(pkg, False, "NOT INSTALLED — run: pip install -r requirements.txt")

# ── Mem0 / ChromaDB Local DB ────────────────────────────────────────────────
section("Mem0 Local Database")
mike_db = Path("mike_db")
check(
    "mike_db/ directory",
    mike_db.exists(),
    "Will be created on first run" if not mike_db.exists() else "exists",
)

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
if all_ok:
    print(f"{GREEN}{BOLD}✅ All checks passed! You're ready to run: python main.py{RESET}")
else:
    print(f"{RED}{BOLD}❌ Some checks failed. Fix the issues above before running main.py.{RESET}")
print()
