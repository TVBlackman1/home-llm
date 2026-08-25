time uv run pytest tests/test_qwen35.py -v > qwen35.test
ollama stop qwen3.5:4b

time uv run pytest tests/test_gemma3.py -v > gemma3.test
ollama stop gemma3:4b

time uv run pytest tests/test_qwen25.py > qwen25.test
ollama stop qwen2.5:3b

time uv run pytest tests/test_phi4mini.py > phi4mini.test
ollama stop phi4-mini

time uv run pytest tests/test_mistral.py > ministral3.test
ollama stop ministral-3:3b