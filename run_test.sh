# uv run pytest tests/test_pipeline.py -m unit
# uv run pytest tests/test_embed.py -m embed

# time uv run pytest tests/test_qwen35.py -m pipeline > qwen35.test
# ollama stop qwen3.5:4b

# time uv run pytest tests/test_gemma3.py -m pipeline > gemma3.test
# ollama stop gemma3:4b

# time uv run pytest tests/test_qwen25.py -m pipeline > qwen25.test
# ollama stop qwen2.5:3b

# time uv run pytest tests/test_phi4mini.py -m pipeline > phi4mini.test
# ollama stop phi4-mini

# time uv run pytest tests/test_mistral.py -m pipeline > ministral3.test
# ollama stop ministral-3:3b

time uv run pytest tests/test_mistral_semantic.py -m pipeline > ministral3.semantic.test2
ollama stop ministral-3:3b

# time uv run pytest tests/test_qwen359.py -m pipeline > qwen359.test
# ollama stop qwen3.5:9b