"""List the models a Moonshot (or OpenAI-compatible) key can actually call.

Moonshot's model names differ from the Bedrock IDs, and they change over time,
so guessing MOONSHOT_EXTRACTION_MODEL is a common source of 404s.

Run:  python -m scripts.list_moonshot_models
"""
from __future__ import annotations

import sys

from backend.config import get_settings


def main() -> int:
    s = get_settings()
    if not s.moonshot_enabled:
        print("MOONSHOT_API_KEY is not set in .env — nothing to query.")
        return 1

    from openai import OpenAI

    client = OpenAI(api_key=s.moonshot_api_key, base_url=s.moonshot_base_url)
    try:
        models = sorted(m.id for m in client.models.list().data)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {s.moonshot_base_url}: {exc}")
        return 1

    print(f"{len(models)} model(s) available at {s.moonshot_base_url}:\n")
    for m in models:
        print(f"  {m}")

    configured = {s.moonshot_extraction_model, s.moonshot_synthesis_model}
    missing = configured - set(models)
    if missing:
        print(f"\n[WARN] configured but not offered: {', '.join(sorted(missing))}")
        print("       Update MOONSHOT_EXTRACTION_MODEL / MOONSHOT_SYNTHESIS_MODEL.")
    else:
        print(f"\n[PASS] configured models are available: {', '.join(sorted(configured))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
