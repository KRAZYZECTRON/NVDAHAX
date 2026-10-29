"""List the models your Nebius key can use (and spot Cosmos / NVIDIA ones).

    python -m ai.list_models

Uses NEBIUS_API_KEY and NEBIUS_BASE_URL (or cosmos.base_url in config.yaml).
"""
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from common import load_config

base_url = os.environ.get("NEBIUS_BASE_URL") or load_config().get("cosmos", {}).get("base_url")
key = os.environ.get("NEBIUS_API_KEY")
print(f"base_url = {base_url}")
if not key:
    sys.exit("NEBIUS_API_KEY is not set in this terminal.")
print(f"key      = {key[:5]}...{key[-4:]}  ({len(key)} chars)")

try:
    ids = sorted(m.id for m in OpenAI(base_url=base_url, api_key=key, timeout=30).models.list())
except Exception as e:
    sys.exit(f"\nRequest failed: {type(e).__name__}: {e}")

print(f"\n{len(ids)} models available.")
cosmos = [i for i in ids if "cosmos" in i.lower()]
nvidia = [i for i in ids if "nvidia" in i.lower() or "nemotron" in i.lower()]
vision = [i for i in ids if any(k in i.lower() for k in ("vl", "vision", "omni", "gemma-3", "llama-4"))]
print("\nCosmos models:", cosmos or "NONE")
print("NVIDIA models:", nvidia or "none")
print("Likely vision models:", vision or "none")
if "--all" in sys.argv:
    print("\nAll:", *ids, sep="\n  ")
