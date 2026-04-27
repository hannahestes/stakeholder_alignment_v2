---
name: Prefer Ollama over Claude API
description: User wants to use local Ollama models instead of the Claude/Anthropic API for persona simulation
type: feedback
---

Use Ollama (local, free) for LLM inference in the tree_to_persona study, not the Claude/Anthropic API.

**Why:** User explicitly said "I want to keep using the free ollama model" when the anthropic package was about to be installed.

**How to apply:** Default to Ollama in evaluate.py and related scripts. Do not default to or push Claude API usage for this project's persona simulation workflow. Keep `anthropic` in requirements.txt only if it's optional/behind a flag — or remove it.
