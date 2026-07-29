"""Probe candidate Bedrock model IDs for (a) basic Converse and (b) structured output."""
from __future__ import annotations

import sys

from langchain_aws import ChatBedrockConverse
from pydantic import BaseModel, Field

REGION = "us-east-2"
CANDIDATES = ["deepseek.v3.2", "zai.glm-4.7-flash"]


class Joke(BaseModel):
    setup: str = Field(description="setup")
    punchline: str = Field(description="punchline")


def probe(model_id: str) -> None:
    print(f"\n=== {model_id} ===")
    llm = ChatBedrockConverse(model=model_id, region_name=REGION, temperature=0, max_tokens=512)

    # 1) basic invoke
    try:
        r = llm.invoke("Reply with exactly: ok")
        print(f"  [basic]      OK -> {str(r.content)[:60]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [basic]      FAIL -> {type(exc).__name__}: {str(exc)[:160]}")
        return

    # 2) structured output (tool-calling based)
    try:
        s = llm.with_structured_output(Joke)
        j = s.invoke("Tell me a short joke about databases.")
        print(f"  [structured] OK -> {j}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [structured] FAIL -> {type(exc).__name__}: {str(exc)[:160]}")

    # 3) structured output via json_schema method (fallback path)
    try:
        s2 = llm.with_structured_output(Joke, method="json_schema")
        j2 = s2.invoke("Tell me a short joke about caching.")
        print(f"  [json_schema] OK -> {j2}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [json_schema] FAIL -> {type(exc).__name__}: {str(exc)[:160]}")


if __name__ == "__main__":
    for m in CANDIDATES:
        probe(m)
    sys.exit(0)
