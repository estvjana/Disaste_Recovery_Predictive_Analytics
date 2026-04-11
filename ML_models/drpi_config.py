"""
=============================================================================
DRPI — Shared Config & Secret Loader
=============================================================================
Loads environment variables from config/.env (never committed to git).
Import this in any DRPI script:
    from drpi_config import NLP_API_KEY, SUPABASE_URL, SUPABASE_KEY
=============================================================================
"""
import os
from pathlib import Path

# Locate config/.env relative to project root (works from ML_models/ or root)
_ROOT = Path(__file__).parent.parent
_ENV_FILE = _ROOT / "config" / ".env"


def _load_env(path: Path):
    """Simple .env loader — no external dependencies needed."""
    if not path.exists():
        print(f"  ⚠  config/.env not found at {path}")
        print("     Copy config/.env.example → config/.env and fill in your keys.")
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            # Only set if not already in environment (real env vars take priority)
            if key not in os.environ:
                os.environ[key] = val


_load_env(_ENV_FILE)

# ── Exported constants ────────────────────────────────────────────────────────
NLP_API_KEY = os.environ.get("NLP_API_KEY", "")
SUPABASE_URL      = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY      = os.environ.get("SUPABASE_KEY", "")
BM25_INDEX_PATH   = os.environ.get("BM25_INDEX_PATH", "./data/BM25.pkl")

PROJECT_ROOT = str(_ROOT)


def check_keys():
    """Print which keys are loaded (masked) — useful for debugging."""
    def mask(s):
        return s[:8] + "..." + s[-4:] if len(s) > 14 else ("✓ set" if s else "✗ MISSING")

    print("  NLP_API_KEY :", mask(NLP_API_KEY))
    print("  SUPABASE_URL      :", SUPABASE_URL if SUPABASE_URL else "✗ MISSING")
    print("  SUPABASE_KEY      :", mask(SUPABASE_KEY))


if __name__ == "__main__":
    print("DRPI Config Check:")
    check_keys()
